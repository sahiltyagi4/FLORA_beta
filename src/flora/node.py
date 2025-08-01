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

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import ray
import torch
from hydra.utils import instantiate
from omegaconf import MISSING
from torch import nn

from .algorithm import BaseAlgorithm, BaseAlgorithmConfig
from .communicator import AggregationOp, BaseCommunicator, BaseCommunicatorConfig
from .data import DataModule, DataModuleConfig
from .model import ModelConfig
from .utils import RequiredSetup


@dataclass
class RayActorConfig:
    """
    Ray actor options for Node resource allocation and scheduling.

    Contains the same options available in Ray's .options() method for controlling
    CPU/GPU assignment, memory limits, fault tolerance, and scheduling behavior.

    Most users can ignore this - defaults work for typical FL experiments.
    Useful for resource-constrained environments or when you need specific hardware.

    Example overrides in topology configs:
    ```yaml
    overrides:
      0: {ray_actor_options: {num_cpus: X.X, memory: NNNNNN}}  # Server
      1: {ray_actor_options: {num_gpus: X.X, accelerator_type: "TYPE"}}  # Client
    ```

    Reference: https://docs.ray.io/en/latest/ray-core/api/doc/ray.actor.ActorClass.options.html
    """

    # ---
    # The quantity of CPU cores to reserve for the lifetime of the actor
    num_cpus: Optional[float] = None

    # ---
    # The quantity of GPUs to reserve for the lifetime of the actor
    # None = automatic allocation, 0 = CPU-only, 0.5 = fractional sharing, 1+ = full GPUs
    num_gpus: Optional[float] = None

    # ---
    # The quantity of various custom resources to reserve for the lifetime of the actor.
    # Dictionary mapping strings (resource names) to floats
    resources: Optional[Dict[str, float]] = None

    # ---
    # Requires that the actor run on a node which meets the specified label conditions
    label_selector: Optional[Dict[str, str]] = None

    # ---
    # Requires that the actor run on a node with the specified type of accelerator
    accelerator_type: Optional[str] = None

    # ---
    # The heap memory request in bytes for this actor, rounded down to the nearest integer
    memory: Optional[int] = None

    # ---
    # The object store memory request for actors only
    # object_store_memory: Optional[int] = None

    # ---
    # Maximum number of times the actor should be restarted when it dies unexpectedly.
    # 0 = no restarts (default), -1 = infinite restarts
    max_restarts: int = 0

    # ---
    # How many times to retry an actor task if the task fails due to a runtime error.
    # 0 = no retries (default), -1 = retry until max_restarts limit, n > 0 = retry up to n times
    max_task_retries: int = 0

    # ---
    # Max number of pending calls allowed on the actor handle. -1 = unlimited
    # max_pending_calls: int = -1

    # ---
    # Max number of concurrent calls to allow for this actor (direct calls only).
    # Defaults to 1 for threaded execution, 1000 for asyncio execution
    # max_concurrency: Optional[int] = None

    # ---
    # The globally unique name for the actor, retrievable via ray.get_actor(name)
    # name: Optional[str] = None

    # ---
    # Override the namespace to use for the actor. Default is anonymous namespace
    namespace: Optional[str] = None

    # ---
    # Actor lifetime: None (fate share with creator) or "detached" (global object)
    # lifetime: Optional[str] = None

    # ---
    # Runtime environment for this actor and its children
    runtime_env: Optional[Dict[str, Any]] = None

    # ---
    # Scheduling strategy: None, "DEFAULT", "SPREAD", or placement group strategies
    scheduling_strategy: Optional[str] = None

    # ---
    # Extended options for Ray libraries (e.g., workflows)
    # _metadata: Optional[Dict[str, Any]] = None

    # ---
    # True if task events from the actor should be reported (tracing)
    # enable_task_events: bool = True


@dataclass
class NodeConfig:
    """
    Configuration for individual federated learning nodes.

    Defines a node's identity, communication partners, and resource requirements.
    Topologies create these automatically - you typically override specific settings
    rather than creating from scratch.

    Common overrides in topology configs:
    ```yaml
    overrides:
      0: {device_hint: "cpu", ray_actor_options: {num_cpus: X.X}}  # Server settings
      1: {device_hint: "cuda:X"}  # Client gets GPU
    ```

    See conf/ directory for working topology examples.
    """

    # Unique identifier for this node within the federated learning topology.
    # Used for logging, debugging, and actor naming. Must be unique per experiment.
    name: str = MISSING

    # Local communication configuration for intra-group federated learning operations.
    # Handles model aggregation within the same communication group (e.g., local cluster).
    # Required - must specify either TorchDist (NCCL/Gloo) or GRPC communicator.
    local_comm: BaseCommunicatorConfig = MISSING

    # Global communication configuration for inter-group federated learning operations.
    # Used by hierarchical topologies where local groups communicate with global coordinators.
    # Optional - None for purely local/centralized topologies.
    global_comm: Optional[BaseCommunicatorConfig] = MISSING

    # Ray actor configuration options for distributed execution.
    # Controls resource allocation, fault tolerance, and scheduling behavior.
    # Default creates actor with Ray defaults (no special resource requirements).
    ray_actor_options: RayActorConfig = field(default_factory=RayActorConfig)

    # Device hint for this node's computation placement
    device_hint: str = "auto"

    # Experiment directory for this node's log files
    # None defaults to Hydra's output directory
    log_dir_base: Optional[str] = None


@ray.remote
class Node(RequiredSetup):
    """
    Distributed federated learning participant (server or client).

    Ray actor that executes FL algorithms with local data and model state.
    Each node manages its own training loop, model updates, and communication with other nodes.

    Nodes execute FL rounds autonomously once Engine calls run_experiment().

    See conf/ directory for topology examples.
    """

    def __init__(
        self,
        name: str,
        local_comm: BaseCommunicatorConfig,
        global_comm: Optional[BaseCommunicatorConfig],
        ray_actor_options: RayActorConfig,
        log_dir_base: str,
        device_hint: str,
        *,  # Force keyword-only args after this point (passed from Engine)
        algorithm: BaseAlgorithmConfig,
        model: ModelConfig,
        datamodule: DataModuleConfig,
    ):
        """
        Initialize federated learning node with configs.

        Args:
            name: Unique node identifier (e.g., "0.1" for group 0, rank 1)
            local_comm: Communication config for intra-group coordination
            global_comm: Communication config for inter-group coordination (hierarchical only)
            algorithm: FL algorithm config
            model: Neural network model config
            datamodule: Data loading and preprocessing config
            device_hint: Device placement ("auto", "cpu", "cuda:X", etc.)
            exp_dir: Base experiment directory for logs and outputs
        """
        super().__init__()
        self.name: str = name
        self.device_hint: str = device_hint
        self.log_dir: str = os.path.join(log_dir_base, name)

        # Store config for model (instantiated during setup)
        self.model_cfg: ModelConfig = model

        # Instantiate components with setup phases
        self.local_comm: BaseCommunicator = instantiate(local_comm)
        self.global_comm: Optional[BaseCommunicator] = (
            instantiate(global_comm) if global_comm else None
        )

        self.algorithm: BaseAlgorithm = instantiate(algorithm, log_dir=self.log_dir)

        self.datamodule: DataModule = instantiate(datamodule)
        # Deferred instantiation
        self.__device: Optional[torch.device] = None

    def _setup(self) -> None:
        """
        Instantiate remaining components and establish connections.

        Called by Engine after all nodes are created but before experiment starts.
        Instantiates model, establishes communicator connections,
        and passes dependencies to algorithm.
        """
        model: nn.Module = instantiate(self.model_cfg)

        print(f"[NODE-SETUP] Device: {self.device}", flush=True)

        # Establish communicator connections
        self.local_comm.setup()
        if self.global_comm:
            self.global_comm.setup()

        # Standard federated learning setup: broadcast initial model from server
        # In hierarchical topologies: global comm first, then local comm
        _t_init_total_start = time.time()

        _t_bcast_total_start = time.time()
        _t_bcast_global_start = time.time()
        if self.global_comm:
            model = self.global_comm.broadcast(model)
        _t_bcast_global_end = time.time()

        _t_bcast_local_start = time.time()
        model = self.local_comm.broadcast(model)
        _t_bcast_local_end = time.time()
        _t_bcast_total_end = time.time()

        # Discover distributed training parameters for synchronized execution
        _t_agg_start = time.time()
        local_iters_per_epoch = (
            len(self.datamodule.train) if self.datamodule.train is not None else 0
        )

        # Find global maximum iterations and epochs (batched for efficiency)
        group_max_epochs_and_iters = self.local_comm.aggregate(
            dict(
                iters_per_epoch=torch.tensor(
                    local_iters_per_epoch,
                    dtype=torch.int,
                ),
                epochs_per_round=torch.tensor(
                    self.algorithm.max_epochs_per_round,
                    dtype=torch.int,
                ),
            ),
            AggregationOp.MAX,
        )
        _t_agg_end = time.time()

        self.algorithm.setup(
            self.local_comm,
            self.global_comm,
            model,
            self.datamodule,
            int(group_max_epochs_and_iters["iters_per_epoch"].item()),
            int(group_max_epochs_and_iters["epochs_per_round"].item()),
        )

        _t_init_total_end = time.time()

        # Log initialization timing metrics individually (uses current context)
        self.algorithm.log_metric(
            "comm_time/bcast_global", _t_bcast_global_end - _t_bcast_global_start
        )
        self.algorithm.log_metric(
            "comm_time/bcast_local", _t_bcast_local_end - _t_bcast_local_start
        )
        self.algorithm.log_metric(
            "comm_time/bcast_total", _t_bcast_total_end - _t_bcast_total_start
        )
        self.algorithm.log_metric("comm_time/agg", _t_agg_end - _t_agg_start)
        self.algorithm.log_metric(
            "comm_time/total", _t_init_total_end - _t_init_total_start
        )

    def run_experiment(self, total_rounds: int) -> Dict[str, Any]:
        """
        Execute federated learning experiment autonomously.

        Runs the complete experiment lifecycle.
        Moves model to compute device, executes all FL rounds via the algorithm.
        Collects timeline data and restores model to original device afterward.

        Args:
            total_rounds: Number of federated learning rounds to execute

        Returns:
            Timeline data containing metrics with FL coordinates
        """
        if not self.is_ready:
            raise RuntimeError("Node not ready - call setup() first")

        print(
            f"[EXPERIMENT-START] Node starting {total_rounds} round experiment",
            flush=True,
        )

        # Device management for experiment execution
        original_device = next(self.algorithm.local_model.parameters()).device
        self.algorithm.local_model = self.algorithm.local_model.to(self.device)

        try:
            for round_idx in range(total_rounds):
                self.algorithm.round_exec(round_idx, total_rounds)

            print(
                f"[EXPERIMENT-END] Node completed {total_rounds} round experiment",
                flush=True,
            )

        finally:
            # Restore original device placement
            self.algorithm.local_model = self.algorithm.local_model.to(original_device)
            print(
                f"[EXPERIMENT-CLEANUP] Model restored to original device: {original_device}",
                flush=True,
            )

        # Return experiment timeline data for display purposes
        return self.algorithm.get_experiment_data()

    def __repr__(self) -> str:
        """Node string representation with name and timestamp."""
        _time = time.strftime("%H:%M:%S", time.gmtime())
        return f"{self.name} {_time}"

    @property
    def device(self) -> torch.device:
        """Compute device with automatic GPU assignment based on rank."""
        if self.__device is None:
            self.__device = self.__resolve_device(
                self.device_hint, rank=self.local_comm.rank
            )
        return self.__device

    @staticmethod
    def __resolve_device(device_hint: str, rank: Optional[int] = None) -> torch.device:
        """
        Resolve device placement for this node.

        Args:
            device_hint: Device specification ("auto", "cpu", "cuda", "cuda:X", etc.)
            rank: Process rank for round-robin GPU assignment (optional)

        Returns:
            PyTorch device for computation
        """
        if device_hint != "auto":
            print(f"[NODE-DEVICE] Explicit: {device_hint}")
            return torch.device(device_hint)

        # Auto-assignment with GPU detection
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("[NODE-DEVICE] Auto: CPU (no GPUs available)")
            return torch.device("cpu")

        # Round-robin GPU assignment
        effective_rank = rank if rank is not None else 0
        if rank is None:
            warnings.warn("No rank provided, defaulting to GPU 0")

        gpu_id = effective_rank % gpu_count
        device_str = f"cuda:{gpu_id}"
        print(
            f"[NODE-DEVICE] Auto: {device_str} (rank {effective_rank}, {gpu_count} GPUs)"
        )
        return torch.device(device_str)
