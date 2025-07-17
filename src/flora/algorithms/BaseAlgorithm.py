# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import Any, Optional

import rich.repr
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..communicator.BaseCommunicator import Communicator, ReductionType
from ..communicator.grpc_communicator import GrpcCommunicator
from ..communicator.TorchDistCommunicator import TorchDistCommunicator
from ..data.DataModule import DataModule
from ..helper.RoundMetrics import RoundMetrics
from ..mixins import SetupMixin
from . import utils
from .utils import log_param_changes

# ======================================================================================


class AggLevel(str, Enum):
    """
    Aggregation levels granularity.

    - ROUND: Aggregate after each global round
    - EPOCH: Aggregate after each local epoch
    - ITER: Aggregate after each local batch iteration

    CONFIGURATION NOTE: Use uppercase values in config files:
    - ✓ agg_level: ROUND (correct)
    - ✗ agg_level: round (incorrect - will cause aggregation to be skipped)

    FUTURE ENHANCEMENTS:
    - TODO: Analyze algorithm incompatibilities with agg_freq > 1 (some algorithms may require specific frequencies)
    - TODO: Design declarative validation framework for algorithm subclasses to specify aggregation requirements
    """

    ROUND = "ROUND"
    EPOCH = "EPOCH"
    ITER = "ITER"  # Batch-level aggregation
    # TODO: Maybe add another level based on the number of samples processed?


@rich.repr.auto
class Algorithm(SetupMixin):
    """
    Base class for federated learning algorithms with hooks-based architecture.

    **Required implementations:**
    - `_configure_local_optimizer()`: Return optimizer (e.g., SGD, Adam)
    - `_train_step()`: Forward pass, return (loss, batch_size)
    - `_aggregate()`: Federated aggregation using self.comm

    **Optional lifecycle hooks:**
    - `_on_setup()`, `_on_round_start()`, `_on_round_end()` for round lifecycle
    - `_on_epoch_start()`, `_on_epoch_end()` for epoch lifecycle
    - `_on_batch_start()`, `_on_batch_end()` for batch lifecycle
    - `_backward_pass()`, `_optimizer_step()` for training customization

    **Available infrastructure:**
    `self.metrics`, `self.comm`, `self.local_model`, hyperparameters, and state tracking.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "_aggregate"):
            cls._aggregate = log_param_changes(cls._aggregate)
        if hasattr(cls, "_setup"):
            cls._setup = log_param_changes(cls._setup)

    def __init__(
        self,
        # Core FL components
        comm: Communicator,
        local_model: nn.Module,
        agg_level: AggLevel,
        agg_freq: int,
        # Training hyperparameters
        local_lr: float,
        max_epochs_per_round: int,
        # Infrastructure components
        tb_writer: Optional[SummaryWriter],
        # Miscellaneous
        **kwargs: Any,
    ):
        """
        Initialize the Algorithm instance.

        Args:
            comm: Communicator for distributed operations
            local_model: The neural network model for this algorithm instance
            agg_level: Level at which aggregation occurs (ROUND, EPOCH, or ITER)
            agg_freq: Frequency of aggregation operations
            local_lr: Learning rate for local training
            max_epochs: Maximum number of epochs per round
            tb_writer: TensorBoard writer for logging metrics
            **kwargs: Additional algorithm-specific parameters
        """
        # Core federated learning components
        self.comm: Communicator = comm
        self.local_model: nn.Module = local_model
        self.agg_level: AggLevel = agg_level
        self.agg_freq: int = agg_freq

        # Training hyperparameters
        if local_lr <= 0:
            raise ValueError(f"local_lr must be positive, got {local_lr}")
        if max_epochs_per_round <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs_per_round}")
        if agg_freq <= 0:
            raise ValueError(f"agg_freq must be positive, got {agg_freq}")

        self.local_lr: float = local_lr
        self.local_epochs_per_round: int = max_epochs_per_round

        # Infrastructure components
        # self.tb_writer: SummaryWriter = tb_writer
        # self.tb_writer: Optional[SummaryWriter] = None
        self.tb_writer = None

        # Initialize state
        self.__round_local_optimizer: Optional[torch.optim.Optimizer] = None
        self.__round_metrics: Optional[RoundMetrics] = None
        self.__round_idx: int = 0
        self.__epoch_idx: int = 0
        self.__batch_idx: int = 0
        self.__local_sample_count: int = 0

        # Lazy-initialized distributed training parameters
        self._local_iters_per_epoch: Optional[int] = None
        self._global_max_iters_per_epoch: Optional[int] = None
        self._global_max_epochs_per_round: Optional[int] = None

    # =============================================================================
    # PROPERTIES
    # =============================================================================

    @property
    def local_optimizer(self) -> torch.optim.Optimizer:
        """Current optimizer for local training.

        Created during round initialization in __reset_round_state().
        """
        if self.__round_local_optimizer is None:
            raise RuntimeError("local_optimizer accessed before round initialization")
        return self.__round_local_optimizer

    @local_optimizer.setter
    def local_optimizer(self, value: torch.optim.Optimizer) -> None:
        if not isinstance(value, torch.optim.Optimizer):
            raise ValueError(
                f"local_optimizer must be torch.optim.Optimizer, got {type(value)}"
            )
        self.__round_local_optimizer = value

    @property
    def metrics(self) -> RoundMetrics:
        """Current round metrics for tracking training statistics.

        Created during round initialization in __reset_round_state().
        """
        if self.__round_metrics is None:
            raise RuntimeError("metrics accessed before round initialization")
        return self.__round_metrics

    @metrics.setter
    def metrics(self, value: RoundMetrics) -> None:
        if not isinstance(value, RoundMetrics):
            raise ValueError(f"metrics must be RoundMetrics, got {type(value)}")
        self.__round_metrics = value

    @property
    def round_idx(self) -> int:
        """Current federated learning round index.

        Tracks which global training round is currently being executed.
        Must be non-negative. Used for aggregation frequency calculations.
        """
        return self.__round_idx

    @round_idx.setter
    def round_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"round_idx must be non-negative, got {value}")
        self.__round_idx = value

    @property
    def epoch_idx(self) -> int:
        """Current local training epoch index within the current round.

        Tracks which epoch is currently being executed in the current round.
        Must be non-negative. Reset to 0 at the start of each round.
        """
        return self.__epoch_idx

    @epoch_idx.setter
    def epoch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"epoch_idx must be non-negative, got {value}")
        self.__epoch_idx = value

    @property
    def batch_idx(self) -> int:
        """Current batch index within the current epoch.

        Tracks which batch is currently being processed in the current epoch.
        Must be non-negative. Reset to 0 at the start of each epoch.
        """
        return self.__batch_idx

    @batch_idx.setter
    def batch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"batch_idx must be non-negative, got {value}")
        self.__batch_idx = value

    @property
    def tb_global_epoch(self) -> int:
        """Global epoch count across all rounds for TensorBoard logging.

        Computed as round_idx * max_epochs + epoch_idx to provide a monotonically
        increasing epoch count suitable for TensorBoard visualization.
        """
        return self.round_idx * self.local_epochs_per_round + self.epoch_idx

    @property
    def local_sample_count(self) -> int:
        """Local samples processed by this node in current round.

        This count represents the number of training samples this node has processed
        and is used for computing aggregation weights in federated learning.

        Validation rules:
        - Must be non-negative
        - Can only be reset to 0 or increased (prevents double-counting)
        - Reset to 0 at the start of each round
        """
        return self.__local_sample_count

    @local_sample_count.setter
    def local_sample_count(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"local_sample_count must be non-negative, got {value}")
        elif value < self.__local_sample_count and value != 0:
            raise ValueError(
                f"local_sample_count can only be reset to 0 or increased (current: {self.__local_sample_count}, got {value})"
            )
        self.__local_sample_count = value

    @property
    def local_iters_per_epoch(self) -> int:
        """Number of local iterations per epoch for this node.

        Computed as len(dataloader) for the local DataLoader. This value is used
        along with global_max_iters_per_epoch to ensure all nodes participate in
        synchronized training loops, even if they have different amounts of local data.

        Available only after round_exec() begins and distributed parameter discovery
        completes during __reset_round_state().
        """
        if self._local_iters_per_epoch is None:
            raise RuntimeError(
                "local_iters_per_epoch accessed before distributed training initialization. "
                "This value is computed from the local DataLoader and is only available after "
                "round_exec() begins and distributed parameter discovery completes."
            )
        return self._local_iters_per_epoch

    @property
    def global_max_iters_per_epoch(self) -> int:
        """Global maximum iterations per epoch across all nodes.

        Computed using MAX reduction across all nodes' local_iters_per_epoch.
        This value ensures all nodes participate in synchronized training loops
        for the same number of iterations, preventing nodes with less data from
        finishing early and causing synchronization issues.

        Requires distributed communication and is only available after round_exec()
        begins and distributed parameter discovery completes during __reset_round_state().
        """
        if self._global_max_iters_per_epoch is None:
            raise RuntimeError(
                "global_max_iters_per_epoch accessed before distributed training initialization. "
                "This value requires communication across all nodes to determine the global maximum "
                "and is only available after round_exec() begins and distributed parameter discovery completes."
            )
        return self._global_max_iters_per_epoch

    @property
    def global_max_epochs_per_round(self) -> int:
        """Global maximum epochs per round across all nodes.

        Computed using MAX reduction across all nodes' max_epochs. This value ensures
        all nodes participate in synchronized training loops for the same number of epochs,
        maintaining consistency in the federated learning protocol even if nodes have
        different max_epochs settings.

        Requires distributed communication and is only available after round_exec()
        begins and distributed parameter discovery completes during __reset_round_state().
        """
        if self._global_max_epochs_per_round is None:
            raise RuntimeError(
                "global_max_epochs_per_round accessed before distributed training initialization. "
                "This value requires communication across all nodes to determine the global maximum "
                "and is only available after round_exec() begins and distributed parameter discovery completes."
            )
        return self._global_max_epochs_per_round

    # =============================================================================
    # SETUP
    # =============================================================================

    def _setup(self, device: torch.device) -> None:
        """
        Default setup for federated learning algorithms (OPTIONAL override).

        Performs standard federated learning initialization:
        - Moves model to target device
        - Broadcasts initial model from rank 0 to all participants

        Override this method if your algorithm needs custom setup behavior.
        When overriding, ALWAYS call super()._setup(device) first to ensure
        the standard model setup happens before your custom logic.

        Args:
            device: Target device for model placement

        Common override use-cases:
        - Initialize algorithm-specific state (control variates, momentum buffers)
        - Create copies of the model for reference (global model, previous models)
        - Set up auxiliary networks or transformations

        Example:
            def _setup(self, device: torch.device) -> None:
                super()._setup(device)  # Standard model setup
                self.global_model = copy.deepcopy(self.local_model)
                self.momentum_buffers = {name: torch.zeros_like(param)
                                       for name, param in self.local_model.named_parameters()}
        """
        # Move model to target device (one-time operation)
        self.local_model = self.local_model.to(device)

        # Standard federated learning setup: broadcast initial model from server
        self.local_model = self.comm.broadcast(self.local_model, src=0)
        # TODO: maybe require this to return the local model or perhaps a whole state object? (should we do the same to aggregate?)

    # =============================================================================
    # MINIMAL OVERRIDES
    # =============================================================================

    @abstractmethod
    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        Configure optimizer for local training (REQUIRED override).

        Create and return the optimizer that will be used for local model updates.
        This is called once per round when the optimizer is first accessed.

        Args:
            local_lr: Learning rate for local training

        Returns:
            Configured PyTorch optimizer (e.g., SGD, Adam, AdamW)

        Example:
            return torch.optim.SGD(self.local_model.parameters(), lr=local_lr, momentum=0.9)
        """
        pass

    @abstractmethod
    def _train_step(self, batch: Any, batch_idx: int) -> tuple[torch.Tensor, int]:
        """
        Compute loss for a single batch (REQUIRED override).

        Perform forward pass and compute loss tensor. This method only handles
        the forward pass - backward pass, optimizer step, and metrics are
        handled automatically by the base class.

        Args:
            batch: Single batch from DataLoader (already on device)
            batch_idx: Current batch index within epoch

        Returns:
            loss: Scalar tensor for backward pass
            batch_size: Number of samples in batch (for metrics)

        Example:
            inputs, targets = batch
            outputs = self.local_model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            return loss, inputs.size(0)
        """
        pass

    @abstractmethod
    def _aggregate(self) -> None:
        """
        Coordinate model updates across clients (REQUIRED override).

        This method is called after local training when aggregation should occur
        (controlled by agg_level and agg_freq). Use self.comm for communication.

        Common patterns:
        - FedAvg: self.local_model = self.comm.aggregate(self.local_model, ReductionType.MEAN)
        - Custom: Aggregate gradients, apply server-side optimization, broadcast result

        Available communication operations:
        - self.comm.aggregate(model, ReductionType.MEAN/SUM): All-reduce aggregation
        - self.comm.broadcast(model, src=0): Broadcast from specific rank
        - self.comm.all_gather(tensors): Gather tensors from all ranks

        Example:
            # Simple FedAvg aggregation
            self.local_model = self.comm.aggregate(self.local_model, ReductionType.MEAN)
        """
        pass

    # =============================================================================
    # =============================================================================

    def __reset_round_state(
        self, round_idx: int, dataloader: Optional[DataLoader]
    ) -> None:
        # Discover distributed training parameters to ensure all nodes participate in the same loop structure
        self._local_iters_per_epoch = len(dataloader) if dataloader is not None else 0

        # Find global maximum iterations and epochs with single batched collective operation
        # Combine both MAX reductions to reduce communication overhead
        discovery_tensor = torch.tensor(
            [self._local_iters_per_epoch, self.local_epochs_per_round], dtype=torch.int
        )
        global_maxes = self.comm.aggregate(discovery_tensor, ReductionType.MAX)

        self._global_max_iters_per_epoch = int(global_maxes[0].item())
        self._global_max_epochs_per_round = int(global_maxes[1].item())

        # ---
        # Create new instances for this round
        self.__round_local_optimizer = self._configure_local_optimizer(self.local_lr)
        self.__round_metrics = RoundMetrics()
        # ---
        self.round_idx = round_idx
        self.epoch_idx = 0
        self.batch_idx = 0
        # ---
        self.local_sample_count = 0

    def round_exec(self, datamodule: DataModule, round_idx: int) -> dict[str, float]:
        """
        Execute federated round computation across multiple epochs.

        DEFAULT: Sequential epoch processing with automatic timing and metrics.

        Override Use-Cases:
        - Early stopping: break epoch loop based on validation metrics
        - Dynamic epochs: adjust epoch count based on convergence
        - Learning rate schedules: step schedulers between epochs
        - Cross-epoch state: maintain state across epochs (e.g., momentum buffers)
        """
        # Future: could select datamodule.val or datamodule.test based on phase
        dataloader = datamodule.train

        _t_start = time.time()
        self.__reset_round_state(round_idx, dataloader)

        print(
            f"[START] Round {round_idx + 1} | "
            f"global_max_epochs_per_round={self.global_max_epochs_per_round} | "
            f"global_max_iters_per_epoch={self.global_max_iters_per_epoch}",
            flush=True,
        )
        # Overridable hook for algorithm-specific logic
        self._round_start(round_idx)

        # ---
        # All nodes enter synchronized epoch loop structure
        self.local_model.train()  # Future: .eval() for val/test

        for epoch_idx in range(self.global_max_epochs_per_round):
            self.__run_epoch(
                epoch_idx,
                dataloader,
            )

        # ---
        # Check if aggregation should occur based on level & frequency
        if self.agg_level == AggLevel.ROUND:
            if self.round_idx % self.agg_freq == 0:
                self._aggregate()
            else:
                next_agg = ((self.round_idx // self.agg_freq) + 1) * self.agg_freq + 1
                print(
                    f"[AGG-SKIP] round={self.round_idx + 1} freq={self.agg_freq} next={next_agg}",
                    flush=True,
                )

        # Overridable hook for algorithm-specific logic
        self._round_end(round_idx)
        _t_end = time.time()
        self.metrics.update_mean("time/round", _t_end - _t_start)

        print(
            f"[END] Round {self.round_idx + 1} |",
            {
                k: round(v, 4)
                if isinstance(v, float) and k.startswith("time/")
                else round(v, 2)
                if isinstance(v, float)
                else v
                for k, v in self.metrics.to_dict().items()
            },
            flush=True,
        )

        return self.metrics.to_dict()

    def __run_epoch(
        self,
        epoch_idx: int,
        dataloader: Optional[DataLoader],
    ) -> None:
        """
        PRIVATE internal method to run a single epoch.
        """
        self.epoch_idx = epoch_idx

        print(
            f"[START] Round {self.round_idx + 1} Epoch {epoch_idx + 1}",
            flush=True,
        )

        _t_epoch_start = time.time()

        # Overridable hook for algorithm-specific logic
        self._epoch_start(epoch_idx)

        # Initialize dataloader iterator for sequential batch processing
        dataloader_iter = iter(dataloader) if dataloader is not None else None

        # All nodes participate in synchronized batch loop
        for batch_idx in range(self.global_max_iters_per_epoch):
            # Set batch index and start batch processing
            self.batch_idx = batch_idx
            _t_batch_start = time.time()
            # Overridable hook for algorithm-specific logic
            self._batch_start(batch_idx)

            # Data preparation: fetch and transfer batch
            batch = None
            _t_batch_data_start = time.time()
            if (
                dataloader_iter is not None
                and self.epoch_idx < self.local_epochs_per_round
            ):
                try:
                    batch = next(dataloader_iter)
                    batch = self._transfer_batch_to_device(
                        batch, next(self.local_model.parameters()).device
                    )
                except StopIteration:
                    # Node has exhausted its data - continue with None batch for synchronization
                    pass
            _t_batch_data_end = time.time()

            # Execute batch computation
            _t_batch_compute_start = time.time()
            if batch is not None:
                self._batch_exec(batch, batch_idx)
            _t_batch_compute_end = time.time()

            # Batch-level aggregation
            if self.agg_level == AggLevel.ITER:
                if self.batch_idx % self.agg_freq == 0:
                    self._aggregate()
                else:
                    next_agg = (
                        (self.batch_idx // self.agg_freq) + 1
                    ) * self.agg_freq + 1
                    print(
                        f"[AGG-SKIP] iter={self.batch_idx + 1} freq={self.agg_freq} next={next_agg}",
                        flush=True,
                    )

            # Overridable hook for algorithm-specific logic
            self._batch_end(batch_idx)

            # Update timing metrics
            _t_batch_end = time.time()

            self.metrics.update_mean(
                "time/step_data", _t_batch_data_end - _t_batch_data_start
            )
            self.metrics.update_mean(
                "time/step_compute", _t_batch_compute_end - _t_batch_compute_start
            )

            # Overall batch timing for all nodes
            self.metrics.update_mean("time/step", _t_batch_end - _t_batch_start)

        # ---
        # Epoch boundary synchronization
        print("[EPOCH-SYNC-START]", flush=True)
        _t_start = time.time()
        sync_signal = torch.tensor([1.0])
        total_signals = self.comm.aggregate(sync_signal, ReductionType.SUM)
        sync_time = time.time() - _t_start
        print(
            f"[EPOCH-SYNC-END] time={sync_time:.4f}s signals={int(total_signals.item())}",
            flush=True,
        )

        # Epoch-level aggregation
        if self.agg_level == AggLevel.EPOCH:
            if self.epoch_idx % self.agg_freq == 0:
                print(
                    f"[AGG-START] epoch={self.epoch_idx + 1} samples={self.local_sample_count}",
                    flush=True,
                )
                _t_start = time.time()
                self._aggregate()
                agg_time = time.time() - _t_start
                print(f"[AGG-END] time={agg_time:.4f}s", flush=True)
            else:
                next_agg = ((self.epoch_idx // self.agg_freq) + 1) * self.agg_freq + 1
                print(
                    f"[AGG-SKIP] epoch={self.epoch_idx + 1} freq={self.agg_freq} next={next_agg}",
                    flush=True,
                )

        # Overridable hook for algorithm-specific logic
        self._epoch_end(epoch_idx)
        _t_epoch_end = time.time()

        self.metrics.update_mean("time/epoch", _t_epoch_end - _t_epoch_start)

        print(
            f"[END] Round {self.round_idx + 1} Epoch {epoch_idx + 1} |",
            {
                k: round(v, 4)
                if isinstance(v, float) and k.startswith("time/")
                else round(v, 2)
                if isinstance(v, float)
                else v
                for k, v in self.metrics.to_dict().items()
            },
            flush=True,
        )

        # Log epoch metrics to TensorBoard
        if self.tb_writer:
            for key, value in self.metrics.to_dict().items():
                self.tb_writer.add_scalar(key, value, self.tb_global_epoch)

    def _batch_exec(self, batch: Any, batch_idx: int) -> None:
        """
        Execute computation for a single batch.

        Override Use-Cases:
        - Multiple optimizer steps: perform several gradient steps per batch
        - Custom backward passes: manual gradient computation or accumulation
        - Mixed precision: automatic or manual scaling for float16 training
        - Gradient clipping: apply gradient norm clipping before _optimizer_step
        """
        # Forward pass hook (implemented by subclasses)
        loss, batch_size = self._train_step(batch, batch_idx)
        # Automatic sample tracking
        self.local_sample_count = self.local_sample_count + batch_size
        self.metrics.update_sum("local/sample_count", batch_size)
        self.metrics.update_sum("local/batch_count", 1)

        # Automatic loss tracking
        self.metrics.update_mean("local/loss", loss.detach().item(), batch_size)

        self.local_optimizer.zero_grad()
        # Backward pass hook
        self._backward_pass(loss, batch_idx)

        # Automatic gradient tracking
        self.metrics.update_mean(
            "local/grad_norm", utils.get_grad_norm(self.local_model)
        )

        # Optimizer step hook
        self._optimizer_step(batch_idx)

    # =============================================================================

    def _backward_pass(self, loss: torch.Tensor, batch_idx: int) -> None:
        """
        Compute gradients from loss tensor.
        PROTECTED hook - can override in subclasses.

        DEFAULT: Standard loss.backward() for automatic differentiation.

        Override Use-Cases:
        - Manual gradients: torch.autograd.grad() for specific parameters
        - Gradient penalty: add regularization terms during backward pass
        - Mixed precision: scale loss before backward pass
        """
        loss.backward()

    def _optimizer_step(self, batch_idx: int) -> None:
        """
        Apply parameter updates using computed gradients.
        PROTECTED hook - override in subclasses.

        DEFAULT: Standard optimizer.step() with current gradients.

        Override Use-Cases:
        - Conditional updates: skip updates based on gradient norms or loss values
        - Per-parameter learning rates: apply different step sizes to different layers
        - Momentum modifications: adjust momentum based on training progress
        """
        self.local_optimizer.step()

    # =============================================================================
    # LIFECYCLE HOOKS
    # =============================================================================

    def _round_start(self, round_idx: int) -> None:
        """
        Algorithm-specific round start hook. Override for model sync and state reset.
        """
        pass

    def _round_end(self, round_idx: int) -> None:
        """
        Algorithm-specific round end hook (PROTECTED - override in subclasses).

        Called after framework aggregation logic. Use for:
        - Custom post-training processing: model validation, state finalization
        - Algorithm-specific aggregation enhancements: custom model updates
        - Round metrics collection: algorithm-specific statistics gathering

        Example:
            def _round_end(self, round_idx: int) -> None:
                self.compute_algorithm_metrics()
                self.save_round_checkpoint()
        """
        pass

    def _epoch_start(self, epoch_idx: int) -> None:
        """
        Algorithm-specific epoch start hook (PROTECTED - override in subclasses).

        Called at the start of each local training epoch. Use for:
        - Learning rate scheduling: step LR scheduler based on epoch progress
        - Epoch state reset: clear per-epoch counters or loss accumulators
        - Dynamic configuration: adjust dropout rates or data augmentation per epoch

        Example:
            def _epoch_start(self, epoch_idx: int) -> None:
                self.scheduler.step()
                self.epoch_loss_accumulator = 0.0
        """
        pass

    def _epoch_end(self, epoch_idx: int) -> None:
        """
        Algorithm-specific epoch end hook (PROTECTED - override in subclasses).

        Called after framework aggregation logic. Use for:
        - Local validation: evaluate model on local validation set
        - Epoch metrics: compute and log per-epoch training statistics
        - Early stopping: check local convergence criteria

        Example:
            def _epoch_end(self, epoch_idx: int) -> None:
                val_loss = self.validate_local_model()
                self.metrics.update("validation/loss", val_loss)
        """
        pass

    def _batch_start(self, batch_idx: int) -> None:
        """
        Algorithm-specific batch start hook (PROTECTED - override in subclasses).

        Called before processing each batch. Use for:
        - Optimizer state: modify learning rates or momentum per batch
        - Batch tracking: initialize batch-specific counters or flags
        - Debug logging: log batch indices or data samples for debugging

        Example:
            def _batch_start(self, batch_idx: int) -> None:
                if batch_idx % 100 == 0:
                    self.adjust_learning_rate(batch_idx)
        """
        pass

    def _batch_end(self, batch_idx: int) -> None:
        """
        Algorithm-specific batch end hook (PROTECTED - override in subclasses).

        Called after framework aggregation logic. Use for:
        - Batch metrics: log loss values or gradient norms per batch
        - Memory cleanup: clear temporary tensors or cached computations
        - Progress tracking: update training progress indicators

        Example:
            def _batch_end(self, batch_idx: int) -> None:
                if batch_idx % 50 == 0:
                    self.log_batch_metrics(batch_idx)
                torch.cuda.empty_cache()  # Memory cleanup
        """
        pass

    # =============================================================================
    # MISC UTILITY METHODS
    # =============================================================================

    def _transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        Move batch tensors to algorithm's compute device.
        PROTECTED utility - can override in subclasses.

        DEFAULT: Handles tensor, tuple, list, dict batch formats automatically.

        Override Use-Cases:
        - Nested structures: recursive transfer for complex batch hierarchies
        - Selective transfer: move only specific tensors to GPU, keep others on CPU
        - Memory optimization: transfer tensors individually to reduce peak memory
        """
        # Single tensor
        if isinstance(batch, torch.Tensor):
            return batch.to(device)

        # Tuple/list of tensors (most common case)
        if isinstance(batch, (tuple, list)):
            transferred = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    transferred.append(item.to(device))
                else:
                    transferred.append(item)  # Keep non-tensors as-is
            return tuple(transferred) if isinstance(batch, tuple) else transferred

        # Dictionary with tensor values
        if isinstance(batch, dict):
            transferred = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    transferred[key] = value.to(device)
                else:
                    transferred[key] = value  # Keep non-tensors as-is
            return transferred

        # Unsupported type - warn user
        logging.warning(
            f"Batch type '{type(batch).__name__}' not handled by _transfer_batch_to_device(). "
            "Override this method for custom batch formats."
        )
        return batch
