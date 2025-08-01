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

import datetime
from enum import Enum

import rich.repr
import torch
import torch.distributed as dist
from torch import nn

from .base import BaseCommunicator, AggregationOp
from .utils import get_msg_info


class InitMethod(str, Enum):
    """Initialization methods for PyTorch distributed process groups."""

    TCP = "tcp"  # TCP-based initialization for network communication
    FILE = "file"  # File-based initialization for shared filesystem


# ======================================================================================


@rich.repr.auto
class TorchDistCommunicator(BaseCommunicator):
    """
    Communication backend using PyTorch distributed collective operations.

    Implements broadcast and aggregation via PyTorch's efficient collective
    primitives (broadcast, all-reduce).
    Uses process groups for coordination with support for multiple backends
    (gloo, nccl) and initialization methods.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        init_method: InitMethod = InitMethod.TCP,
        backend: str = "gloo",
        sharedfile: str = "sharedfile",
        timeout: int = 60,
        max_retries: int = 5,
    ) -> None:
        """
        Initialize PyTorch distributed communicator.

        Args:
            rank: Process rank in distributed group
            world_size: Total number of processes in distributed group
            master_addr: Master node address for coordination
            master_port: Master node port for coordination
            init_method: TCP or file-based process group initialization
            backend: Communication backend ('gloo' for CPU, 'nccl' for GPU)
            sharedfile: Shared file path for file-based initialization
            timeout: Process group initialization timeout (seconds)
            max_retries: Maximum initialization retry attempts
        """
        super().__init__()
        print(
            f"[COMM-INIT] rank={rank}/{world_size} | backend={backend} | addr={master_addr}:{master_port}"
        )

        # Core distributed parameters
        self.rank: int = rank
        self.world_size: int = world_size
        self.init_method: str = init_method
        self.master_addr: str = master_addr
        self.master_port: int = master_port
        self.backend: str = backend
        self.sharedfile: str = sharedfile
        self.timeout: datetime.timedelta = datetime.timedelta(seconds=timeout)
        self.max_retries: int = max_retries

        # Backend validation with automatic fallback
        if self.backend == "nccl" and not torch.cuda.is_available():
            print("[COMM-INIT] NCCL→gloo fallback (no CUDA available)")
            self.backend = "gloo"

    def _setup(self):
        """
        Initialize PyTorch distributed process group.

        Creates the distributed process group using the specified initialization
        method and backend.

        All ranks must call this before any collective operations.
        """

        print(
            f"[COMM-SETUP] rank={self.rank}/{self.world_size} | {self.init_method} | backend={self.backend}"
        )

        match self.init_method:
            case InitMethod.TCP:
                addr = f"tcp://{self.master_addr}:{self.master_port}"
                dist.init_process_group(
                    backend=self.backend,
                    init_method=addr,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=self.timeout,
                )
            case InitMethod.FILE:
                addr = f"file://{self.sharedfile}"
                dist.init_process_group(
                    backend=self.backend,
                    init_method=addr,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=self.timeout,
                )
            case _:
                raise ValueError(
                    f"Unknown init_method: {self.init_method}. Supported: {[m.value for m in InitMethod]}"
                )

    def broadcast(
        self,
        msg: BaseCommunicator.MsgT,
        src: int = 0,
    ) -> BaseCommunicator.MsgT:
        """
        Broadcast message from source rank to all other ranks.

        Uses PyTorch's distributed broadcast collective for efficient
        one-to-many communication within the process group.

        Args:
            msg: Model, tensor dict, or tensor to broadcast
            src: Source rank ID (default: 0)

        Returns:
            Message with broadcasted values
        """
        print(f"[COMM-BCAST] {get_msg_info(msg)} | src={src}")

        if isinstance(msg, nn.Module):
            # Broadcast all trainable parameters
            for _, p in msg.named_parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)
        elif isinstance(msg, dict):
            # Broadcast each tensor in dictionary
            for tensor in msg.values():
                dist.broadcast(tensor, src=src)
        else:
            # Broadcast single tensor
            dist.broadcast(msg, src=src)
        return msg

    def aggregate(
        self,
        msg: BaseCommunicator.MsgT,
        reduction: AggregationOp,
    ) -> BaseCommunicator.MsgT:
        """
        Aggregate message across all ranks using PyTorch all-reduce collective.

        Performs efficient element-wise reduction across all process ranks.

        Args:
            msg: Model, tensor dict, or tensor to aggregate
            reduction: SUM, MEAN, or MAX reduction operation

        Returns:
            Message with aggregated values distributed to all ranks
        """

        print(f"[COMM-AGG] {get_msg_info(msg)} | reduction={reduction}")

        # Map reduction type to PyTorch operation
        reduction_ops = {
            AggregationOp.SUM: dist.ReduceOp.SUM,
            AggregationOp.MEAN: dist.ReduceOp.AVG,
            AggregationOp.MAX: dist.ReduceOp.MAX,
        }

        if reduction not in reduction_ops:
            raise ValueError(f"Unsupported reduction type: {reduction}")

        op = reduction_ops[reduction]

        if isinstance(msg, nn.Module):
            # Aggregate all trainable parameters
            for _, p in msg.named_parameters():
                if p.requires_grad:
                    dist.all_reduce(p.data, op=op)
        elif isinstance(msg, dict):
            # Aggregate each tensor in dictionary
            for tensor in msg.values():
                dist.all_reduce(tensor, op=op)
        else:
            # Aggregate single tensor
            dist.all_reduce(msg, op=op)

        return msg

    def close(self):
        """
        Destroy PyTorch distributed process group and clean up resources.

        Should be called when distributed communication is no longer needed.
        All ranks must call this to properly clean up the process group.
        """
        print("[COMM-CLOSE]")
        dist.destroy_process_group()
