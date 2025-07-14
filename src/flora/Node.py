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

import time
from enum import Enum
from typing import Any, Optional, Set

import ray
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
# from torch.utils.tensorboard import SummaryWriter

from .algorithms.BaseAlgorithm import Algorithm
from .communicator.BaseCommunicator import Communicator
from .data.DataModule import DataModule
from .mixins import SetupMixin

# class NodeRole(Enum):
#     """
#     Defines the fundamental capabilities of a Node participant in the federation.

#     Each role represents a specific capability.
#     Nodes can have multiple roles to express their full set of capabilities.
#     """

#     AGGREGATOR = "Aggregator"  # Aggregates updates from other nodes
#     TRAINER = "Trainer"  # Performs local training

#     # Future additions??
#     # COORDINATOR = "coordinator"  # Coordinates communication/scheduling
#     # RELAY = "relay"             # Forwards messages between nodes
#     # VALIDATOR = "validator"      # Validates updates or models


@ray.remote
class Node(SetupMixin):
    """
    Distributed compute node for federated learning participants.

    Responsibilities:
    - Execute local federated learning algorithm implementations
    - Manage local model state (and training data access? # TODO: local data may be unnecessary & redundant for some roles e.g. aggregator)
    - Own and manage local copy of global model model and communicator instances
    - Handle device management for hardware resources

    Integration:
    - Instantiated as Ray actors by the Engine for distributed execution
    - Configured through Hydra with algorithm and communication dependencies
    - Should enable any topology pattern through consistent and generalizable interfaces
    """

    def __init__(
        self,
        id: str,
        # roles: Set[NodeRole],
        comm_cfg: DictConfig,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        local_rank: int,
        world_size: int,
        log_dir: str,
        device: str = "auto",
        **kwargs: Any,
    ):
        super().__init__()
        print(f"[NODE-INIT] {id}")
        self.id: str = id
        # self.roles: Set[NodeRole] = roles

        # Distributed computing context
        self.local_rank: Optional[int] = local_rank
        self.world_size: Optional[int] = world_size
        self.device: torch.device = self.select_device(device, rank=local_rank)

        # Communication backend instantiation
        self.comm: Communicator = instantiate(
            comm_cfg,
            local_rank=local_rank,
            world_size=world_size,
        )

        # PyTorch model
        self.local_model: nn.Module = instantiate(model_cfg)

        # Data module
        self.datamodule: DataModule = instantiate(data_cfg)

        # TensorBoard setup
        # self.tb_writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

        # Federated learning algorithm
        self.algo: Algorithm = instantiate(
            algo_cfg,
            comm=self.comm,
            local_model=self.local_model,
            tb_writer=None,
        )

    def __repr__(self) -> str:
        """
        String representation showing node ID and capabilities.

        Returns:
            Formatted string with node ID and role list
        """
        # role_names = [role.value for role in self.roles]
        # return f"Node {self.id}: {role_names}"
        return f"{self.id}"

    def should_train(self) -> bool:
        """
        Determine if this node should perform training based on its role.

        In centralized topology:
        - Rank 0 (Server/Aggregator): No training, only aggregation
        - Rank 1+ (Client/Trainer): Performs local training

        Returns:
            True if node should train, False if it should skip training
        """
        return self.local_rank != 0
        # return True

    def _setup(self) -> None:
        """
        Initialize node dependencies and prepare for federated learning execution.

        Sets up communication backend and prepares all components for distributed training.
        Called once before federated learning begins.
        """
        print("[NODE-SETUP]", flush=True)

        # Initialize communication backend
        self.comm.setup()

        self.algo.setup()
        # summary(self.model, verbose=1)

    def round_exec(self, round_idx: int) -> dict[str, float]:
        """
        Execute federated learning round with algorithm-controlled communication.
        Node provides communication infrastructure, Algorithm controls federated lifecycle.

        Args:
            round_idx: Current training round number

        Returns:
            Dictionary with training metrics and results
        """
        if not self.is_ready:
            raise RuntimeError(f"Node {self.id} not ready - call setup() first")

        print(f"[ROUND-START] R{round_idx + 1}", flush=True)
        round_start_time = time.time()

        # Model state before any synchronization
        metrics: dict[str, float] = dict(round_idx=round_idx)

        # Round Start Logic with automatic tracking
        self.algo.round_start(round_idx)

        # 4. Local training (only for trainer nodes)
        if self.should_train():
            if self.datamodule.train is None:
                raise ValueError(
                    "ERROR: Node is configured to train but no training DataLoader is provided."
                )
            # Centralized device management: ensure model is on compute device
            self.algo.local_model.to(self.device)

            # Execute local training round
            self.algo.train_round(
                self.datamodule.train,
                round_idx,
            )
        else:
            print("[NODE-SKIP-TRAIN] Server node - no local training")

        # Round End Logic with automatic tracking
        self.algo.round_end(round_idx)

        # Collect all metrics from algorithm execution
        metrics["time/round"] = time.time() - round_start_time
        metrics.update(self.algo.metrics.to_dict())

        print(
            f"[ROUND-END] R{round_idx + 1} |",
            {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()},
            flush=True,
        )
        return metrics

    @staticmethod
    def select_device(device_hint: str, rank: Optional[int] = None) -> torch.device:
        """
        Select and configure compute device for this node.

        Supports automatic GPU detection with round-robin assignment based on rank.
        Falls back to CPU if no GPUs are available.

        # TODO: Round-robin GPU assignment assumes all nodes are on the same machine, which may not be true for multi-node setups.
        # TODO: If some nodes lack GPUs, we may need smarter logic for heterogeneous environments.
        # TODO: Potentially move into a NodeResources mixin in the future.

        Args:
            device_hint: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)
            rank: Process rank for round-robin GPU assignment (optional)

        Returns:
            Configured PyTorch device for computation
        """

        if device_hint == "auto":
            local_gpu_count = torch.cuda.device_count()
            if local_gpu_count > 0:
                # Use provided rank for round-robin GPU assignment
                if rank is None:
                    print(
                        "WARN: No rank provided for device assignment, defaulting to GPU 0"
                    )
                    rank = 0

                assigned_gpu_id = rank % local_gpu_count
                device_str = f"cuda:{assigned_gpu_id}"
                print(
                    f"[DEVICE-AUTO] {local_gpu_count} GPUs detected | assigned {device_str} | rank {rank}"
                )
                return torch.device(device_str)

            print("Device auto-selection: No GPUs detected, using CPU")
            return torch.device("cpu")

        print(f"Device explicit: Using {device_hint}")
        return torch.device(device_hint)
