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
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

import rich.repr
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..communicator.BaseCommunicator import Communicator, ReductionType
from ..communicator.grpc_communicator import GrpcCommunicator
from ..communicator.TorchDistCommunicator import TorchDistCommunicator
from ..helper.RoundMetrics import RoundMetrics
from . import utils
from .utils import log_param_changes
from ..mixins import SetupMixin

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
    ITER = "ITER"


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

    def __init__(
        self,
        # Core FL components
        comm: Communicator,
        local_model: nn.Module,
        agg_level: AggLevel,
        agg_freq: int,
        # Training hyperparameters
        local_lr: float,
        max_epochs: int,
        # Infrastructure components
        tb_writer: Optional[SummaryWriter],
        # Extensibility
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
        self.local_lr: float = local_lr
        self.max_epochs: int = max_epochs

        # Infrastructure components
        # self.tb_writer: SummaryWriter = tb_writer
        # self.tb_writer: Optional[SummaryWriter] = None
        self.tb_writer = None

        # Initialize state tracking
        self._local_optimizer: Optional[torch.optim.Optimizer] = None
        self._metrics: Optional[RoundMetrics] = None
        self._curr_tr_sample_ct: int = 0
        self._round_idx: int = 0
        self._epoch_idx: int = 0
        self._batch_idx: int = 0

    # =============================================================================
    # PROPERTIES
    # =============================================================================

    @property
    def local_optimizer(self) -> torch.optim.Optimizer:
        """Current optimizer - created lazily when first accessed."""
        if self._local_optimizer is None:
            self._local_optimizer = self._configure_local_optimizer(self.local_lr)
        return self._local_optimizer

    @property
    def metrics(self) -> RoundMetrics:
        """Current round metrics - created on first access."""
        if self._metrics is None:
            self._metrics = RoundMetrics()
        return self._metrics

    @property
    def round_idx(self) -> int:
        """Current federated round index with validation."""
        return self._round_idx

    @round_idx.setter
    def round_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"round_idx must be non-negative, got {value}")
        self._round_idx = value

    @property
    def epoch_idx(self) -> int:
        """Current local epoch index with validation."""
        return self._epoch_idx

    @epoch_idx.setter
    def epoch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"epoch_idx must be non-negative, got {value}")
        self._epoch_idx = value

    @property
    def batch_idx(self) -> int:
        """Current local batch index with validation."""
        return self._batch_idx

    @batch_idx.setter
    def batch_idx(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"batch_idx must be non-negative, got {value}")
        self._batch_idx = value

    @property
    def tb_global_step(self) -> int:
        """Current global step for TensorBoard logging."""
        return self.round_idx * self.max_epochs + self.epoch_idx

    @property
    def local_sample_count(self) -> int:
        """Local samples processed by this node in current round - this node's contribution."""
        return self._curr_tr_sample_ct

    @local_sample_count.setter
    def local_sample_count(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"local_samples must be >= 0 (got {value})")
        if value == 0:
            logging.info("NOTE: Local sample count reset to 0.")
        elif value < self._curr_tr_sample_ct:
            raise ValueError(
                f"local_samples can only be reset to 0 or increased (current: {self._curr_tr_sample_ct}, got {value})"
            )
        self._curr_tr_sample_ct = value

    # =============================================================================
    # SETUP
    # =============================================================================

    @log_param_changes
    def _setup(self) -> None:
        """
        Default setup for federated learning algorithms (OPTIONAL override).

        Performs standard federated learning initialization:
        - Broadcasts initial model from rank 0 to all participants

        Override this method if your algorithm needs custom setup behavior.
        When overriding, ALWAYS call super()._setup() first to ensure
        the standard model broadcast happens before your custom logic.

        Common override use-cases:
        - Initialize algorithm-specific state (control variates, momentum buffers)
        - Create copies of the model for reference (global model, previous models)
        - Set up auxiliary networks or transformations

        Example:
            def _setup(self) -> None:
                super()._setup()  # Standard model broadcast
                self.global_model = copy.deepcopy(self.local_model)
                self.momentum_buffers = {name: torch.zeros_like(param)
                                       for name, param in self.local_model.named_parameters()}
        """
        # Standard federated learning setup: broadcast initial model from server
        self.local_model = self.comm.broadcast(self.local_model, src=0)

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
    # TRAINING LOGIC
    # =============================================================================

    def train_round(
        self,
        dataloader: DataLoader[Any],
        round_idx: int,
    ):
        """
        Execute federated round computation across multiple epochs.

        DEFAULT: Sequential epoch processing with automatic timing and metrics.

        Override Use-Cases:
        - Early stopping: break epoch loop based on validation metrics
        - Dynamic epochs: adjust epoch count based on convergence
        - Learning rate schedules: step schedulers between epochs
        - Cross-epoch state: maintain state across epochs (e.g., momentum buffers)
        """
        self.local_model.train()

        for epoch_idx in range(self.max_epochs):
            # Update current epoch index
            self.epoch_idx = epoch_idx
            print(
                f"[EPOCH-START] R{round_idx + 1} E{epoch_idx + 1}",
                flush=True,
            )
            # Epoch timing
            epoch_start_time = time.time()
            self.__epoch_start(epoch_idx)

            # Epoch computation
            self.__train_epoch(dataloader, epoch_idx)

            # Overridable epoch end hook
            self.__epoch_end(epoch_idx)
            epoch_time = time.time() - epoch_start_time

            self.metrics.update_mean("time/epoch", epoch_time)

            print(
                f"[EPOCH-END] R{round_idx + 1} E{epoch_idx + 1} |",
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
                    self.tb_writer.add_scalar(key, value, self.tb_global_step)

    def __train_epoch(
        self,
        dataloader: DataLoader[Any],
        epoch_idx: int,
    ) -> None:
        """
        Execute computation for all batches in a single epoch.

        DEFAULT: Sequential batch processing with automatic device transfer and timing.

        Override Use-Cases:
        - Gradient accumulation: accumulate gradients over multiple batches before optimizer step
        - Custom sampling: dynamic batch selection or weighted sampling
        - Multi-dataloader: alternate between multiple data sources
        - Batch preprocessing: custom transformations before device transfer
        """
        data_iter = iter(dataloader)

        for batch_idx in range(len(dataloader)):
            # Update current batch index
            self.batch_idx = batch_idx
            # Batch timing
            batch_start_time = time.time()
            # Overridable batch start hook
            self.__batch_start(batch_idx)

            # ---
            # Sample batch and transfer to device
            data_start_time = time.time()
            batch = self._transfer_batch_to_device(
                next(data_iter),
                next(self.local_model.parameters()).device,
            )
            data_time = time.time() - data_start_time

            # ---
            # Batch computation
            compute_start_time = time.time()
            self.__train_batch(batch, batch_idx)
            compute_time = time.time() - compute_start_time

            # ---
            # Overridable batch end hook
            self.__batch_end(batch_idx)
            batch_time = time.time() - batch_start_time

            # ---
            self.metrics.update_mean("time/step_data", data_time)
            self.metrics.update_mean("time/step_compute", compute_time)
            self.metrics.update_mean("time/step", batch_time)

    def __train_batch(self, batch: Any, batch_idx: int) -> None:
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
        self.metrics.update_sum("train/num_samples", batch_size)
        self.metrics.update_sum("train/num_batches", 1)

        # Automatic loss tracking
        self.metrics.update_mean("train/loss", loss.detach().item(), batch_size)

        self.local_optimizer.zero_grad()
        # Backward pass hook
        self._backward_pass(loss, batch_idx)

        # Automatic gradient tracking
        self.metrics.update_mean(
            "train/grad_norm", utils.get_grad_norm(self.local_model)
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
    # INTERNAL LOGIC
    # =============================================================================

    def __should_aggregate(self):
        match self.agg_level:
            case AggLevel.ROUND:
                return self.round_idx % self.agg_freq == 0
            case AggLevel.EPOCH:
                return self.epoch_idx % self.agg_freq == 0
            case AggLevel.ITER:
                return self.batch_idx % self.agg_freq == 0
            case _:
                raise ValueError(f"Invalid coord_level: {self.agg_level}")

    # =============================================================================
    # LIFECYCLE HOOKS
    # =============================================================================

    @log_param_changes
    def round_start(self, round_idx: int) -> None:
        """
        Initialize a new federated round by resetting framework state and calling algorithm-specific setup.

        Automatically resets optimizer, metrics, and indices, then calls _round_start() for
        algorithm-specific initialization like model synchronization.

        Override _round_start() instead of this method.
        """
        # Framework state reset (always happens, can't be forgotten by algorithms)
        self._local_optimizer = None
        self._metrics = None
        self.round_idx = round_idx
        self.epoch_idx = 0
        self.batch_idx = 0
        self.local_sample_count = 0

        # Call algorithm-specific round start logic
        self._round_start(round_idx)

    def _round_start(self, round_idx: int) -> None:
        """
        Algorithm-specific round start hook. Override for model sync and state reset.
        """
        pass

    @log_param_changes
    def round_end(self, round_idx: int) -> None:
        """
        Finalize federated round by handling aggregation and calling algorithm-specific cleanup.

        Automatically handles ROUND-level aggregation, then calls _round_end() for
        algorithm-specific finalization.

        Override _round_end() instead of this method.
        """
        # Framework aggregation logic (always happens when appropriate)
        if self.agg_level == AggLevel.ROUND:
            should_agg = self.__should_aggregate()
            if should_agg:
                self._aggregate()
            else:
                print(
                    f"[AGG-SKIP] R{round_idx + 1} | agg_freq={self.agg_freq} | next aggregation at R{((self.round_idx // self.agg_freq) + 1) * self.agg_freq + 1}",
                    flush=True,
                )

        # Call algorithm-specific round end logic
        self._round_end(round_idx)

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

    def __epoch_start(self, epoch_idx: int) -> None:
        """
        Private epoch initialization. Calls _epoch_start() for algorithm-specific setup.
        """
        # Framework epoch initialization (currently minimal, but could expand)
        # Future: epoch-level metrics reset, scheduler steps, etc.

        # Call algorithm-specific epoch start logic
        self._epoch_start(epoch_idx)

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

    def __epoch_end(self, epoch_idx: int) -> None:
        """
        Private epoch finalization. Handles EPOCH-level aggregation and calls _epoch_end().
        """
        # Framework aggregation logic (always happens when appropriate)
        if self.agg_level == AggLevel.EPOCH:
            should_agg = self.__should_aggregate()
            if should_agg:
                self._aggregate()
            else:
                print(
                    f"[AGG-SKIP] E{epoch_idx + 1} | agg_freq={self.agg_freq} | next aggregation at E{((self.epoch_idx // self.agg_freq) + 1) * self.agg_freq + 1}",
                    flush=True,
                )

        # Call algorithm-specific epoch end logic
        self._epoch_end(epoch_idx)

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

    def __batch_start(self, batch_idx: int) -> None:
        """
        Private batch initialization. Calls _batch_start() for algorithm-specific setup.
        """
        # Framework batch initialization (currently minimal, but could expand)
        # Future: batch-level metrics reset, optimizer state tracking, etc.

        # Call algorithm-specific batch start logic
        self._batch_start(batch_idx)

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

    def __batch_end(self, batch_idx: int) -> None:
        """
        Private batch finalization. Handles ITER-level aggregation and calls _batch_end().
        """
        # Framework aggregation logic (always happens when appropriate)
        if self.agg_level == AggLevel.ITER:
            should_agg = self.__should_aggregate()
            if should_agg:
                self._aggregate()
            else:
                print(
                    f"[AGG-SKIP] B{batch_idx + 1} | agg_freq={self.agg_freq} | next aggregation at B{((self.batch_idx // self.agg_freq) + 1) * self.agg_freq + 1}",
                    flush=True,
                )

        # Call algorithm-specific batch end logic
        self._batch_end(batch_idx)

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
