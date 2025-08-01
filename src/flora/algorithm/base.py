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
import math
import time
import warnings
from abc import abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Optional

import rich.repr
import torch
from torch import nn
from typeguard import typechecked

from ..communicator import AggregationOp, BaseCommunicator
from ..data import DataModule
from ..utils import MetricAggType, MetricLogger, RequiredSetup
from . import utils
from ._lifecycle_hooks import LifecycleHooks
from ._schedules import ExecutionSchedules

# ======================================================================================


@rich.repr.auto
class BaseAlgorithm(RequiredSetup, LifecycleHooks, MetricLogger):
    """
    Base class for implementing federated learning algorithms.

    Inherit from this to create FL algorithms like FedAvg, FedProx, or SCAFFOLD.
    Handles FL infrastructure (distributed computing, communication, lifecycle management,
    metrics) so you can focus on the algorithm logic.

    **Required Methods:**
    You only need to implement two methods:
    - `_configure_local_optimizer()`: Return your optimizer (SGD, Adam, etc.)
    - `_compute_loss()`: Forward pass and loss calculation

    The default aggregation uses sample-weighted averaging (works for FedAvg).
    Override `_aggregate_within_group()` for custom algorithms like FedProx or SCAFFOLD.

    **Examples:**
        # Simple algorithm (FedAvg) - uses default weighted aggregation
        class FedAvg(BaseAlgorithm):
            def _configure_local_optimizer(self, local_lr):
                return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

            def _compute_loss(self, batch):
                x, y = batch
                logits = self.local_model(x)
                return F.cross_entropy(logits, y)

        # Custom algorithm - overrides aggregation
        class CustomAlgorithm(BaseAlgorithm):
            # ... same required methods ...
            def _aggregate_within_group(self, comm, weight):
                utils.scale_params(self.local_model, weight)
                return comm.aggregate(self.local_model, AggregationOp.SUM)

    **Advanced - MultiGroupTopology (Cross-Institutional FL):**
    When using MultiGroupTopology for cross-institutional federated learning,
    the framework uses two-level sample-weighted aggregation:
    - **Within-group**: Each client weighted by personal samples / group total samples
    - **Cross-group**: Each group weighted by group total samples / global total samples

    This ensures fair representation when institutions have different data sizes.

    **Optional Customization Hooks:**
    Override only what you need:

    *Aggregation Methods:*
    - `_aggregate_within_group()`: Custom FL aggregation (FedProx, SCAFFOLD, etc.)
    - `_aggregate_across_groups()`: Cross-institutional aggregation (MultiGroupTopology)

    *Lifecycle Hooks:*
    - `_round_start()`, `_round_end()`: Round-level setup/cleanup
    - `_train_epoch_start()`, `_train_epoch_end()`: Training epoch boundaries
    - `_eval_epoch_start()`, `_eval_epoch_end()`: Evaluation epoch boundaries
    - `_train_batch_start()`, `_train_batch_end()`: Training batch boundaries
    - `_eval_batch_start()`, `_eval_batch_end()`: Evaluation batch boundaries

    *Custom Processing:*
    - `_train_batch()`, `_eval_batch()`: Custom batch handling
    - `_backward_pass()`, `_optimizer_step()`: Custom training operations
    - `_transfer_batch_to_device()`, `_infer_batch_size()`: Custom data handling
    """

    @typechecked
    def __init__(
        self,
        local_lr: float,
        max_epochs_per_round: int,
        schedules: ExecutionSchedules,
        log_dir: str,
    ):
        """
        Set up a federated learning algorithm with training parameters.

        Args:
            local_lr: Learning rate for each client's local training
            max_epochs_per_round: How many epochs each client trains per FL round
            schedules: When to aggregate models and run evaluations
            log_dir: Where to save TensorBoard logs and metrics CSV files
        """
        # Validate training parameters
        if local_lr <= 0:
            raise ValueError(f"local_lr must be positive, got {local_lr}")
        if max_epochs_per_round <= 0:
            raise ValueError(
                f"max_epochs_per_round must be positive, got {max_epochs_per_round}"
            )

        RequiredSetup.__init__(self)
        LifecycleHooks.__init__(self)
        MetricLogger.__init__(
            self,
            log_dir=log_dir,
            global_step_fn=lambda: self.global_step,
            metadata_fields={
                "round_idx": lambda: self.round_idx,
                "epoch_idx": lambda: self.epoch_idx,
                "batch_idx": lambda: self.batch_idx,
            },
        )

        # Store training parameters
        self.local_lr: float = local_lr
        self.max_epochs_per_round: int = max_epochs_per_round

        # Store execution schedules
        self.schedules: ExecutionSchedules = schedules

        # Directory for metrics logging and TensorBoard output
        self.log_dir: str = log_dir

        # Node context dependencies (injected via _setup())
        self.__local_comm: Optional[BaseCommunicator] = None
        self.__global_comm: Optional[BaseCommunicator] = None
        self.__local_model: Optional[nn.Module] = None
        self.__datamodule: Optional[DataModule] = None

        # Training state indices
        self.__round_idx: int = 0
        self.__epoch_idx: int = 0
        self.__batch_idx: int = 0
        self.__num_samples_trained: int = 0  # For aggregation weights

        # Training components
        self.__local_optimizer: Optional[torch.optim.Optimizer] = None

        # Distributed training parameters (discovered during setup)
        self.__group_max_iters_per_epoch: Optional[int] = None
        self.__group_max_epochs_per_round: Optional[int] = None

    # =============================================================================
    # PROPERTIES
    # =============================================================================

    @property
    def local_comm(self) -> BaseCommunicator:
        """
        Talk to other clients in your group (cluster, organization, etc.).

        Use this for the main FL aggregation, averaging models with other
        clients that have similar network conditions or are in the same datacenter.
        """
        if self.__local_comm is None:
            raise RuntimeError(
                "local_comm accessed before setup() - call setup() first"
            )
        return self.__local_comm

    @property
    def global_comm(self) -> Optional[BaseCommunicator]:
        """
        Talk to other groups in cross-institutional/hierarchical FL (None for simple centralized FL).

        Only some nodes have this, typically the "group leaders" that aggregate
        results from multiple clusters/organizations. Most algorithms won't touch this.
        """
        return self.__global_comm

    @property
    def local_model(self) -> nn.Module:
        """
        The neural network model this client is training.

        Gets updated during FL rounds as you aggregate with other clients.
        Use this for training, evaluation, and in your aggregation logic.
        """
        if self.__local_model is None:
            raise RuntimeError("model accessed before setup() - call setup() first")
        return self.__local_model

    @local_model.setter
    def local_model(self, value: nn.Module) -> None:
        """Update the model (usually happens during aggregation)."""
        self.__local_model = value

    @property
    def datamodule(self) -> DataModule:
        """
        This client's local data for training and evaluation.

        Use datamodule.train for training batches and datamodule.eval for testing.
        Each client has different data, which is what makes federated learning work.
        """
        if self.__datamodule is None:
            raise RuntimeError(
                "datamodule accessed before setup() - call setup() first"
            )
        return self.__datamodule

    @property
    def local_optimizer(self) -> torch.optim.Optimizer:
        """Current optimizer for local training.

        Created during round initialization in __reset_round_state().
        """
        if self.__local_optimizer is None:
            raise RuntimeError("local_optimizer accessed before round initialization")
        return self.__local_optimizer

    @local_optimizer.setter
    def local_optimizer(self, value: torch.optim.Optimizer) -> None:
        self.__local_optimizer = value

    @property
    def round_idx(self) -> int:
        """Current federated learning round index."""
        return self.__round_idx

    @round_idx.setter
    def round_idx(self, value: int) -> None:
        if value not in (0, self.__round_idx, self.__round_idx + 1):
            raise ValueError(
                f"round_idx can only be reset (0) or incremented ({self.__round_idx} → {self.__round_idx + 1}), got {value}"
            )
        self.__round_idx = value

    @property
    def epoch_idx(self) -> int:
        """Current local training epoch index within the current round."""
        return self.__epoch_idx

    @epoch_idx.setter
    def epoch_idx(self, value: int) -> None:
        if value not in (0, self.__epoch_idx, self.__epoch_idx + 1):
            raise ValueError(
                f"epoch_idx can only be reset (0) or incremented ({self.__epoch_idx} → {self.__epoch_idx + 1}), got {value}"
            )
        self.__epoch_idx = value

    @property
    def batch_idx(self) -> int:
        """Current batch index within the current epoch."""
        return self.__batch_idx

    @batch_idx.setter
    def batch_idx(self, value: int) -> None:
        if value not in (0, self.__batch_idx, self.__batch_idx + 1):
            raise ValueError(
                f"batch_idx can only be reset (0) or incremented ({self.__batch_idx} → {self.__batch_idx + 1}), got {value}"
            )
        self.__batch_idx = value

    @property
    def global_max_iters_per_epoch(self) -> int:
        """Global maximum iterations per epoch across all nodes.

        Used by training loops to synchronize all nodes for the same number of
        iterations, preventing nodes with less data from finishing early.
        """
        if self.__group_max_iters_per_epoch is None:
            raise RuntimeError(
                "global_max_iters_per_epoch accessed before setup() - call setup() first"
            )
        return self.__group_max_iters_per_epoch

    @property
    def global_max_epochs_per_round(self) -> int:
        """Global maximum epochs per round across all nodes.

        Used by training loops to synchronize all nodes for the same number of
        epochs, maintaining consistency even if nodes have different max_epochs settings.
        """
        if self.__group_max_epochs_per_round is None:
            raise RuntimeError(
                "global_max_epochs_per_round accessed before setup() - call setup() first"
            )
        return self.__group_max_epochs_per_round

    # =============================================================================
    # SETUP
    # =============================================================================

    def _setup(
        self,
        local_comm: BaseCommunicator,
        global_comm: Optional[BaseCommunicator],
        model: nn.Module,
        datamodule: DataModule,
        group_max_iters_per_epoch: int,
        group_max_epochs_per_round: int,
    ) -> None:
        """
        Setup algorithm with injected dependencies.

        **Override for algorithm-specific initialization logic.**
        ALWAYS call super()._setup(...) first when overriding.

        Args:
            local_comm: Local communication interface for intra-group operations
            global_comm: Optional global communication interface for cross-institutional/hierarchical FL
            model: ML model being trained
            datamodule: Data loading interface providing train/eval dataloaders
            global_max_iters_per_epoch: Global maximum iterations per epoch across all nodes
            global_max_epochs_per_round: Global maximum epochs per round across all nodes
        """
        # Store injected dependencies
        self.__local_comm = local_comm
        self.__global_comm = global_comm
        self.__local_model = model
        self.__datamodule = datamodule

        # Store distributed training parameters
        self.__group_max_iters_per_epoch = group_max_iters_per_epoch
        self.__group_max_epochs_per_round = group_max_epochs_per_round

    # =============================================================================
    # MINIMAL OVERRIDES
    # =============================================================================

    @abstractmethod
    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        Create the optimizer for this client's local training.

        **REQUIRED OVERRIDE**: Subclasses must implement this method.

        Called once per FL round to get a fresh optimizer.
        Most algorithms just use SGD, but you can use Adam, AdamW, or whatever works for your problem.

        Args:
            local_lr: Learning rate for local training

        Returns:
            Optimizer that will train the local model

        Example:
            return torch.optim.SGD(self.local_model.parameters(), lr=local_lr, momentum=0.9)
        """
        pass

    @abstractmethod
    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Run the forward pass and compute loss for one batch.

        **REQUIRED OVERRIDE**: Subclasses must implement this method.

        This is where your model's forward pass happens.
        The framework handles everything else (backward pass, optimizer steps, metrics tracking).
        Just focus on getting your predictions and computing the loss.

        Args:
            batch: Single batch from your DataLoader (already moved to device)

        Returns:
            loss: Scalar tensor that PyTorch can backprop through

        Example:
            x, y = batch  # or however your data is structured
            logits = self.local_model(x)
            loss = F.cross_entropy(logits, y)
            return loss
        """
        pass

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        Combine your model with other clients' models within the same group.

        **Override for custom FL algorithms** like FedProx, SCAFFOLD, etc.
        Default implementation provides sample-weighted averaging (FedAvg).

        Called after local training when it's time to sync up with other clients.
        This is where different FL algorithms differ - FedAvg just averages,
        FedProx adds regularization, SCAFFOLD tracks control variates, etc.

        Args:
            comm: Communication interface to talk to other clients in your group
            weight: This client's contribution weight pre-calculated by framework: client_samples / group_total_samples.

        Returns:
            The aggregated model that combines knowledge from multiple clients

        Examples:
            # Simple unweighted FedAvg (ignores data distribution)
            return comm.aggregate(self.local_model, AggregationOp.MEAN)

            # Sample-weighted aggregation (default behavior, better for unbalanced data)
            utils.scale_params(self.local_model, weight)
            return comm.aggregate(self.local_model, AggregationOp.SUM)
        """
        # Scale this client's model by its data proportion within the group
        utils.scale_params(self.local_model, weight)

        # Aggregate weighted models across all clients in the group
        aggregated_model = comm.aggregate(
            self.local_model,
            reduction=AggregationOp.SUM,
        )

        return aggregated_model

    def _aggregate_across_groups(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        Perform global aggregation across groups.

        **Override for custom cross-institutional aggregation** in MultiGroupTopology setups.
        Default implementation provides sample-weighted averaging across groups.

        Called when global_comm is available.
        Aggregates locally-aggregated models across different groups/clusters/organizations
        in cross-institutional/hierarchical FL topologies.

        Args:
            comm: Communication interface for inter-group coordination
            weight: This group's contribution weight pre-calculated by framework: group_total_samples / global_total_samples.

        Returns:
            Globally aggregated model after inter-group coordination
        """
        # Weight this group's model by its data proportion relative to all groups
        utils.scale_params(self.local_model, weight)

        # Aggregate weighted group models across all groups
        aggregated_model = comm.aggregate(
            self.local_model,
            reduction=AggregationOp.SUM,
        )

        return aggregated_model

    # =============================================================================
    # =============================================================================

    @MetricLogger.context("sync", duration_key="total_time")
    def __synchronize(self) -> None:
        """
        Coordinate federated model aggregation and evaluation.

        Orchestrates the complete synchronization process across different FL topologies:
        - Centralized: Only intra-group aggregation (global_comm=None)
        - Hierarchical: Intra-group → inter-group → broadcast (global_comm present)

        Called at different granularities based on schedules.aggregation configuration:
        - round_end: After complete training rounds (most common FL pattern)
        - epoch_end: After local training epochs (for frequent sync algorithms)
        - batch_end: After individual training batches (for high-frequency sync)

        Five-phase execution:
        0. Pre-aggregation evaluation (current local model state)
        1. Intra-group aggregation: All nodes aggregate within their group
        2. Inter-group coordination: Group representatives aggregate globally
        3. Conditional broadcast: Distribute global results if inter-group occurred
        4. Post-aggregation evaluation (final aggregated model state)
        """
        # Phase 0: Pre-aggregation evaluation (before any aggregation)
        if self.schedules.evaluation.pre_aggregation():
            self.__eval_epoch(self.local_model)

        # Phase 1: Intra-group aggregation via all-reduce
        with self.log_duration("local_agg_time"):
            # Calculate within-group sample totals and weights
            group_total_samples = self.local_comm.aggregate(
                torch.tensor([self.__num_samples_trained], dtype=torch.float32),
                reduction=AggregationOp.SUM,
            ).item()

            # Validation: warn if no samples trained in group
            if group_total_samples == 0:
                warnings.warn(
                    f"Zero samples trained across all nodes in group ({self.progress_info_str}). "
                    "Check data availability or epoch scheduling. Using uniform weights for aggregation.",
                    UserWarning,
                )

            within_group_weight = self.__num_samples_trained / max(
                group_total_samples, 1
            )

            self.local_model = self._aggregate_within_group(
                self.local_comm, within_group_weight
            )

        # Phase 2: Inter-group coordination (group servers only)
        if self.global_comm is not None:
            with self.log_duration("global_agg_time"):
                # Calculate across-group sample totals and weights using group totals
                global_total_samples = self.global_comm.aggregate(
                    torch.tensor([group_total_samples], dtype=torch.float32),
                    reduction=AggregationOp.SUM,
                ).item()

                # Validation: warn if no samples trained globally
                if global_total_samples == 0:
                    warnings.warn(
                        f"Zero samples trained across all groups globally ({self.progress_info_str}). "
                        "Check data availability or cross-group coordination. Using uniform weights for cross-group aggregation.",
                        UserWarning,
                    )

                across_group_weight = group_total_samples / max(global_total_samples, 1)

                self.local_model = self._aggregate_across_groups(
                    self.global_comm, across_group_weight
                )

        # Phase 3: Conditional broadcast to distribute global results
        # In cross-institutional/hierarchical FL: only group representatives participate in global_comm,
        # but all nodes need the final global model.
        # Check if any node in this local group participated in global aggregation.
        with self.log_duration("local_bcast_time"):
            needs_final_bcast = (
                self.local_comm.aggregate(
                    torch.tensor(1.0 if self.global_comm is not None else 0.0),
                    AggregationOp.MAX,
                )
                > 0
            )

            if needs_final_bcast:
                self.local_model = self.local_comm.broadcast(self.local_model)

        # Post-aggregation evaluation (global model)
        if self.schedules.evaluation.post_aggregation():
            self.__eval_epoch(self.local_model)

        # Reset optimizer and sample counter after aggregation
        # After any aggregation (including broadcast), the model parameters have changed,
        # so the optimizer's internal state (momentum, Adam statistics, etc.) is no longer valid.
        # All nodes must create fresh optimizers for the new parameters.
        # Similarly, num_samples_trained resets to track samples for the next aggregation.
        self.__local_optimizer = self._configure_local_optimizer(self.local_lr)
        self.__num_samples_trained = 0

    def round_exec(self, round_idx: int, max_rounds: int) -> None:
        """
        Execute one complete federated learning round.

        **Override for custom round logic** or specialized FL algorithms that need
        non-standard round execution flow.

        Runs local training epochs, handles round-level aggregation,
        and coordinates evaluation based on the configured schedules.

        Args:
            round_idx: Current round number (0-indexed)
            max_rounds: Total rounds in this experiment
        """

        # Initialize optimizer for first round
        # For subsequent rounds, optimizer is reset after aggregation in __synchronize()
        # If no aggregation occurred in previous round, we keep the existing optimizer
        if self.__local_optimizer is None:
            self.__local_optimizer = self._configure_local_optimizer(self.local_lr)
            self.__num_samples_trained = 0

        # Reset state indices
        self.round_idx = round_idx
        self.epoch_idx = 0
        self.batch_idx = 0

        # Experiment start evaluation (only on first round) - before any training work
        if round_idx == 0 and self.schedules.evaluation.experiment_start():
            self.__eval_epoch(self.local_model)

        print(
            f"[ROUND-START] {self.progress_info_str} | "
            f"global_max_epochs_per_round={self.global_max_epochs_per_round} | "
            f"global_max_iters_per_epoch={self.global_max_iters_per_epoch}",
            flush=True,
        )

        # Overridable hook for algorithm-specific logic
        self._round_start()

        # ---
        # All nodes enter synchronized epoch loop structure
        self.local_model.train()  # Future: .eval() for evaluation phases

        for epoch_idx in range(self.global_max_epochs_per_round):
            # Run epoch training (timing handled by decorator)
            self.__train_epoch(epoch_idx)

        # Round-level aggregation
        if self.schedules.aggregation.round_end():
            self.__synchronize()

        # Overridable hook for algorithm-specific logic
        self._round_end()

        print(f"[ROUND-END] {self.progress_info_str}", flush=True)

        # Experiment end evaluation (only on last round)
        if round_idx == max_rounds - 1 and self.schedules.evaluation.experiment_end():
            self.__eval_epoch(self.local_model)

    @MetricLogger.context("train", duration_key="epoch_time", print_progress=True)
    def __train_epoch(
        self,
        epoch_idx: int,
    ) -> None:
        """
        Run one complete training epoch across all batches.

        Handles the full training loop for one epoch - loads batches, runs training,
        tracks timing, and can trigger aggregation if configured for epoch-level sync.
        All clients stay synchronized even if they have different amounts of data.
        """
        self.epoch_idx = epoch_idx

        # Train epoch start hook
        self._train_epoch_start()

        # Initialize dataloader iterator for sequential batch processing
        dataloader_iter = iter(self.datamodule.train or [])

        device = next(self.local_model.parameters()).device

        # All nodes participate in synchronized batch loop
        for batch_idx in range(self.global_max_iters_per_epoch):
            # Set batch index and start batch processing
            self.batch_idx = batch_idx
            _t_batch_start = time.time()
            # Overridable hook for algorithm-specific logic
            self._train_batch_start()

            # Data preparation: fetch and transfer batch
            batch = None
            _t_batch_data_start = time.time()
            if self.epoch_idx < self.max_epochs_per_round:
                try:
                    batch = next(dataloader_iter)
                    batch = self._transfer_batch_to_device(batch, device=device)
                except StopIteration:
                    # Node has exhausted its data - continue with None batch for synchronization
                    pass
            _t_batch_data_end = time.time()

            # Execute batch computation
            _t_batch_compute_start = time.time()
            if batch is not None:
                # Framework handles batch size inference first
                batch_size = self._infer_batch_size(batch)

                # Execute user training logic and get metrics
                user_metrics = self._train_batch(batch)

                # Only count samples after successful training
                self.__num_samples_trained += batch_size

                # Framework handles metric logging
                for metric_name, metric_value in user_metrics.items():
                    self.log_metric(metric_name, metric_value)

                # Framework adds automatic metrics
                self.log_metric("num_samples", batch_size, MetricAggType.SUM)
                self.log_metric("num_batches", 1, MetricAggType.SUM)

            _t_batch_compute_end = time.time()

            # Batch-level aggregation
            if self.schedules.aggregation.batch_end():
                self.__synchronize()

            # Overridable hook for algorithm-specific logic
            self._train_batch_end()

            # Accumulate timing metrics for batch-level processing
            _t_batch_end = time.time()

            # Add batch timing metrics to accumulator
            self.log_metric(
                "batch_data_time",
                _t_batch_data_end - _t_batch_data_start,
            )
            self.log_metric(
                "batch_compute_time",
                _t_batch_compute_end - _t_batch_compute_start,
            )
            self.log_metric(
                "batch_time",
                _t_batch_end - _t_batch_start,
            )

        # ---
        # Epoch boundary synchronization
        with self.log_duration("epoch_heartbeat_time"):
            sync_signal = torch.tensor([1.0])
            total_signals = self.local_comm.aggregate(sync_signal, AggregationOp.SUM)

        # Epoch-level aggregation
        if self.schedules.aggregation.epoch_end():
            self.__synchronize()

        # Train epoch end hook
        self._train_epoch_end()

    @MetricLogger.context("eval", duration_key="epoch_time", print_progress=True)
    def __eval_epoch(self, model: nn.Module) -> None:
        """
        Evaluate the model on this client's test data.

        Runs through the entire evaluation dataset without computing gradients
        (saves GPU memory). Called at different points depending on your evaluation
        schedule - before aggregation, after aggregation, or both.

        Args:
            model: The model to evaluate (usually self.local_model)
        """
        if self.datamodule.eval is None:
            raise RuntimeError(
                f"Evaluation data not available for {self.progress_info_str}. "
                "Ensure datamodule.eval is properly configured or disable evaluation in the schedule."
            )

        # Temporarily switch to eval mode
        was_training = model.training
        model.eval()

        # Overridable hook for algorithm-specific logic
        self._eval_epoch_start()

        # Initialize dataloader iterator for sequential batch processing
        dataloader_iter = iter(self.datamodule.eval or [])

        with torch.no_grad():
            # Simple loop through eval data - no synchronization needed during eval
            for idx, batch in enumerate(dataloader_iter):
                # Start batch processing with detailed timing
                _t_batch_start = time.time()
                # Overridable hook for algorithm-specific logic
                self._eval_batch_start()

                # Data preparation: fetch and transfer batch
                _t_batch_data_start = time.time()
                batch = self._transfer_batch_to_device(
                    batch, next(model.parameters()).device
                )
                _t_batch_data_end = time.time()

                # Execute evaluation batch computation
                _t_batch_compute_start = time.time()

                # Framework handles batch size inference first
                batch_size = self._infer_batch_size(batch)

                # Execute user evaluation logic and get metrics
                user_metrics = self._eval_batch(batch)

                # Framework handles metric logging
                for metric_name, metric_value in user_metrics.items():
                    self.log_metric(metric_name, metric_value)

                # Framework adds automatic metrics
                self.log_metric("num_samples", batch_size, MetricAggType.SUM)
                self.log_metric("num_batches", 1, MetricAggType.SUM)

                _t_batch_compute_end = time.time()

                # Overridable hook for algorithm-specific logic
                self._eval_batch_end()

                # Accumulate timing metrics for batch-level processing
                _t_batch_end = time.time()

                # Add batch timing metrics to accumulator
                self.log_metric(
                    "batch_data_time",
                    _t_batch_data_end - _t_batch_data_start,
                )
                self.log_metric(
                    "batch_compute_time",
                    _t_batch_compute_end - _t_batch_compute_start,
                )
                self.log_metric(
                    "batch_time",
                    _t_batch_end - _t_batch_start,
                )

        # Overridable hook for algorithm-specific logic
        self._eval_epoch_end()

        # Restore original mode
        if was_training:
            model.train()

    def _train_batch(self, batch: Any) -> Dict[str, float]:
        """
        Execute training computation for one batch.

        **Override for custom training procedures** like gradient accumulation,
        mixed precision, or specialized batch processing.
        Default: Forward pass, backward pass, optimizer step, return loss metric.

        Args:
            batch: Training batch from DataLoader (already moved to device)

        Returns:
            Dictionary of metrics to log (e.g., {"loss": 0.5, "accuracy": 0.9})
            Framework automatically adds samples and batches metrics
        """
        # Forward pass
        loss = self._compute_loss(batch)

        # Training operations
        self.local_optimizer.zero_grad()
        self._backward_pass(loss)

        # Capture gradient norm before optimizer step
        grad_norm = utils.get_grad_norm(self.local_model)

        self._optimizer_step()

        # Return metrics to log
        return {
            "loss": loss.detach().item(),
            "grad_norm": grad_norm,
        }

    def _eval_batch(self, batch: Any) -> Dict[str, float]:
        """
        Execute evaluation computation for one batch.

        **Override for custom evaluation metrics** or specialized evaluation procedures.
        Default: Forward pass (no gradients), return loss metric.

        Args:
            batch: Evaluation batch from DataLoader (already moved to device)

        Returns:
            Dictionary of metrics to log (e.g., {"loss": 0.3, "accuracy": 0.85})
            Framework automatically adds samples and batches metrics
        """
        # Forward pass
        loss = self._compute_loss(batch)

        # Return metrics to log
        return {"loss": loss.detach().item()}

    # =============================================================================

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """
        Compute gradients from the loss.

        **Override for custom gradient computation** like gradient clipping,
        gradient accumulation, or specialized differentiation techniques.
        Default implementation uses standard PyTorch backpropagation.
        """
        loss.backward()

    def _optimizer_step(self) -> None:
        """
        Update model parameters using computed gradients.

        **Override for custom parameter updates** like gradient clipping,
        learning rate scheduling, or specialized optimizer behavior.
        Default implementation calls the optimizer's step() method.
        """
        self.local_optimizer.step()

    @property
    def global_step(self) -> int:
        """
        Convert FL coordinates to linear step number for TensorBoard.

        Maps (round, epoch, batch) position to a single increasing counter.
        TensorBoard uses this for the x-axis when plotting metrics over time.

        Returns:
            Step number for current FL position
        """
        # Steps from completed rounds
        completed_rounds_steps = (
            self.round_idx
            * self.global_max_epochs_per_round
            * self.global_max_iters_per_epoch
        )

        # Steps from completed epochs in current round
        completed_epochs_steps = self.epoch_idx * self.global_max_iters_per_epoch

        # Steps from current batch position
        current_batch_step = self.batch_idx

        return completed_rounds_steps + completed_epochs_steps + current_batch_step

    # =============================================================================
    # MISC UTILITY METHODS
    # =============================================================================

    def _transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        Move batch data to the compute device (CPU/GPU).

        **Override for custom batch formats** or specialized device transfer logic.
        Handles common batch formats automatically: tensors, tuples, lists, dicts.

        Examples of when to override:
        - Nested data structures that need recursive transfer
        - Mixed CPU/GPU processing where only some tensors go to GPU
        - Memory optimization by transferring tensors individually
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

        # Unsupported batch format
        logging.warning(
            f"Unknown batch type '{type(batch).__name__}' in _transfer_batch_to_device(). "
            "Override this method to handle custom batch formats."
        )
        return batch

    def _infer_batch_size(self, batch: Any) -> int:
        """
        Infer the batch size from a data batch.

        **Override for custom batch formats** not supported by the default logic.
        Attempts to determine batch size from common batch formats with absolute certainty.

        Args:
            batch: Data batch in any format

        Returns:
            Batch size as integer

        Raises:
            RuntimeError: If batch size cannot be determined with certainty

        Supported formats:
            - Single tensor: batch.shape[0]
            - Tuple/list: first tensor's shape[0]
            - Dict with 'input'/'inputs': tensor's shape[0]
        """
        # Single tensor
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]

        # Tuple or list - use first tensor
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            first_item = batch[0]
            if isinstance(first_item, torch.Tensor):
                return first_item.shape[0]

        # Dictionary with common input keys
        if isinstance(batch, dict):
            for key in ["input", "inputs", "x", "data"]:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    return batch[key].shape[0]

        # Cannot determine batch size with certainty
        raise RuntimeError(
            f"Cannot infer batch size from batch type '{type(batch).__name__}'. "
            f"Override _infer_batch_size() to handle your custom batch format."
        )

    @staticmethod
    def track_param_changes(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator that logs model parameter changes and detects numerical issues.

        # TODO: re-integrate this

        Useful for debugging aggregation problems, gradient explosion, or parameter
        corruption. Prints before/after parameter norms and detects common issues.
        """

        @wraps(func)
        def wrapper(algo: Any, *args: Any, **kwargs: Any) -> Any:
            phase = func.__name__  # Automatically get the function name
            before_norm = utils.get_param_norm(algo.local_model)
            before_hash = utils.hash_model_params(algo.local_model)

            # Fatal check: model must have valid parameters
            if before_norm == 0.0:
                raise RuntimeError(
                    f"Model has zero parameters before {phase}(). All weights are zero."
                )
            if math.isnan(before_norm) or math.isinf(before_norm):
                raise RuntimeError(
                    f"Model has invalid parameters before {phase}(). Contains NaN or Inf."
                )

            # Execute the original function
            result = func(algo, *args, **kwargs)

            after_norm = utils.get_param_norm(algo.local_model)
            after_hash = utils.hash_model_params(algo.local_model)

            delta = after_norm - before_norm
            changed = before_hash != after_hash

            print(
                f"[{phase.upper()}] local_model hash: {before_hash[:8]} → {after_hash[:8]} | "
                f"norm: {before_norm:.4f} → {after_norm:.4f} (Δ={delta:.6f}) | "
                f"{'CHANGED' if changed else 'UNCHANGED'}"
            )

            # Fatal check: operation must not corrupt the model
            if after_norm == 0.0:
                raise RuntimeError(
                    f"Operation {phase}() zeroed all parameters. Model is broken."
                )
            if math.isnan(after_norm) or math.isinf(after_norm):
                raise RuntimeError(
                    f"Operation {phase}() caused numerical instability. Parameters are NaN/Inf."
                )

            # Warnings for suspicious patterns
            if phase == "_aggregate" and not changed:
                warnings.warn(
                    f"Aggregation {phase}() completed but parameters unchanged. "
                    f"Check if nodes have training data and non-zero aggregation weights.",
                    UserWarning,
                )

            if after_norm > before_norm * 10:
                warnings.warn(
                    f"Parameter explosion in {phase}(). "
                    f"Norm increased {after_norm / before_norm:.1f}x from {before_norm:.4f} to {after_norm:.4f}. "
                    f"Consider reducing learning rate or gradient clipping.",
                    UserWarning,
                )

            if after_norm < before_norm * 0.1:
                warnings.warn(
                    f"Parameter norm vanishing in {phase}(). "
                    f"Norm decreased {before_norm / after_norm:.1f}x from {before_norm:.4f} to {after_norm:.4f}. "
                    f"Check for vanishing gradients, excessive regularization, or scaling issues.",
                    UserWarning,
                )

            return result

        return wrapper
