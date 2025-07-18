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
from typing import Union

import rich.repr
import torch
import torch.distributed as dist
from torch import nn

from .BaseCommunicator import Communicator, ReductionType
from ..algorithms import utils

# ======================================================================================


@rich.repr.auto
class TorchDistCommunicator(Communicator):
    """
    Communicator implementation using PyTorch distributed primitives.

    Provides broadcast and aggregation operations via torch.distributed
    collective communication functions. Supports TCP and file-based
    initialization methods with configurable backends (gloo, nccl).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        init_method: str = "tcp",
        # group_name: str = "default",
        master_addr: str = "127.0.0.1",
        master_port: str = "29500",
        backend: str = "gloo",
        sharedfile: str = "sharedfile",
        timeout: int = 30,
        max_retries: int = 5,
    ):
        super().__init__()
        print(
            f"[COMM-INIT] rank={rank}/{world_size} | backend={backend} | addr={master_addr}:{master_port}"
        )

        self.rank: int = rank
        self.world_size: int = world_size
        self.init_method: str = init_method
        # self.group_name = group_name
        self.master_addr: str = master_addr
        self.master_port: str = master_port
        self.backend: str = backend
        self.sharedfile: str = sharedfile
        self.timeout: datetime.timedelta = datetime.timedelta(seconds=timeout)
        self.max_retries: int = max_retries

        # Backend validation and fallback
        if self.backend == "nccl" and not torch.cuda.is_available():
            print("[COMM-INIT] NCCL→gloo fallback (no CUDA)")
            self.backend = "gloo"

    # def setup(self):
    #     """
    #     Initialize PyTorch distributed process group using TCP.
    #     """
    #     tcp_addr = f"tcp://{self.master_addr}:{self.master_port}"

    #     # Retry loop
    #     for attempt in range(self.max_retries + 1):
    #         try:
    #             print(
    #                 f"Initializing process group (attempt {attempt + 1})"
    #             )

    #             # NOTE: May no longer be necessary (need to re-think back through this)
    #             # small delay based on rank to avoid race conditions
    #             time.sleep(0.1 * self.rank)

    #             dist.init_process_group(
    #                 backend=self.backend,
    #                 init_method=tcp_addr,
    #                 rank=self.rank,
    #                 world_size=self.world_size,
    #                 timeout=self.timeout,
    #             )

    #             self._process_group = dist.group.WORLD
    #             break
    #         except Exception as e:
    #             print(
    #                 f"Initialization attempt {attempt + 1} failed: {str(e)}"
    #             )
    #             if attempt == self.max_retries:
    #                 raise RuntimeError(
    #                     f"Failed to initialize process group after {self.max_retries + 1} attempts"
    #                 ) from e
    #             time.sleep(1.0)

    #     print(f"Process group initialized successfully")

    def _setup(self):
        """
        Initialize PyTorch distributed process group.

        Creates the distributed process group using either TCP or file-based
        initialization method. Required before any collective operations.
        """

        print(
            f"[COMM-SETUP] rank={self.rank}/{self.world_size} | {self.init_method} | backend={self.backend}"
        )

        if self.init_method == "tcp":
            addr = f"tcp://{self.master_addr}:{self.master_port}"
            dist.init_process_group(
                backend=self.backend,
                init_method=addr,
                rank=self.rank,
                world_size=self.world_size,
                timeout=self.timeout,
            )
        else:
            addr = f"file://{self.sharedfile}"
            dist.init_process_group(
                backend=self.backend,
                init_method=addr,
                rank=self.rank,
                world_size=self.world_size,
            )

    def broadcast(
        self,
        msg: Communicator.MsgT,
        src: int = 0,
    ) -> Communicator.MsgT:
        """
        Broadcast message from source rank to all ranks.

        Args:
            msg: Model, tensor dict, or tensor to broadcast
            src: Source rank (default: 0)
        Returns:
            Broadcasted message
        """
        print(f"[COMM-BCAST] {type(msg).__name__} | src: {src}")

        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)
        elif isinstance(msg, dict):
            # Handle tensor dictionaries
            for tensor in msg.values():
                dist.broadcast(tensor, src=src)
        else:
            dist.broadcast(msg, src=src)
        return msg

    def aggregate(
        self,
        msg: Communicator.MsgT,
        reduction: ReductionType,
    ) -> Communicator.MsgT:
        """
        Aggregate message across all ranks using all-reduce operations.

        Performs distributed summation, averaging, or max operations on tensors/models.
        Algorithms are responsible for any pre-scaling or weighting.

        Args:
            msg: Model, tensor dict, or tensor to aggregate
            reduction: SUM, MEAN, or MAX reduction operation
        Returns:
            Aggregated message
        """

        print(
            f"[COMM-AGG] {type(msg).__name__} | reduction={reduction} | info={self.get_msg_info(msg)}"
        )

        # Map reduction type to PyTorch operation
        reduction_ops = {
            ReductionType.SUM: dist.ReduceOp.SUM,
            ReductionType.MEAN: dist.ReduceOp.AVG,
            ReductionType.MAX: dist.ReduceOp.MAX,
        }

        if reduction not in reduction_ops:
            raise ValueError(f"Unsupported reduction type: {reduction}")

        op = reduction_ops[reduction]

        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if not p.requires_grad:
                    continue
                tensor = p.data
                dist.all_reduce(tensor, op=op)

        elif isinstance(msg, dict):
            # Handle tensor dictionaries
            for tensor in msg.values():
                dist.all_reduce(tensor, op=op)

        else:
            # Handle single tensors
            dist.all_reduce(msg, op=op)

        return msg

    # def send(
    #     self,
    #     msg: Communicator.MsgT,
    #     dst: int,
    # ) -> Communicator.MsgT:
    #     """
    #     :param msg: message to send
    #     :param id: client or server id ranging from 0 to (total_clients - 1)
    #     :return: the sending message
    #     """
    #     print(
    #         f"Send to rank {dst} | {type(msg)}",
    #         flush=True,
    #     )

    #     if isinstance(msg, nn.Module):
    #         for _, p in msg.named_parameters():
    #             if not p.requires_grad:
    #                 continue

    #             tensor = p.data
    #             dist.send(tensor, dst=dst)
    #     else:
    #         dist.send(msg, dst=dst)
    #     return msg

    # def receive(
    #     self,
    #     msg: Communicator.MsgT,
    #     src: int,
    # ) -> Communicator.MsgT:
    #     """
    #     :param msg: message to receive
    #     :param id: client or server id ranging from 0 to (total_clients - 1)
    #     :return: the receiving message
    #     """
    #     print(
    #         f"Receive from rank {src} | {type(msg)}",
    #         flush=True,
    #     )

    #     if isinstance(msg, nn.Module):
    #         for _, p in msg.named_parameters():
    #             if not p.requires_grad:
    #                 continue

    #             tensor = p.data
    #             dist.recv(tensor, src=src)
    #     else:
    #         dist.recv(msg, src=src)
    #     return msg

    # def collect(
    #     self,
    #     msg: Union[nn.Module, torch.Tensor, float, int],
    # ) -> list[tuple[int, Communicator.MsgT]]:
    #     """
    #     all-gather in decentralized MPI collectives
    #     :param msg: message to receive
    #     :param id: client_id specifying the client update comes from. redundant in MPI communication as all_gather
    #     collects by rank ids
    #     :return: either nested list of layerwise model data collected from clients or a simple list of gathered data
    #     """
    #     print(
    #         f"Collect from all ranks | {type(msg)}",
    #         flush=True,
    #     )

    #     collected = []
    #     if isinstance(msg, nn.Module):
    #         for _, p in msg.named_parameters():
    #             if not p.requires_grad:
    #                 continue
    #             buf = [torch.zeros_like(p.data) for _ in range(self.world_size)]
    #             tensor = p.data
    #             dist.all_gather(buf, tensor)
    #             collected.append([(r, buf[r]) for r in range(self.world_size)])
    #     else:
    #         base = torch.tensor(msg)
    #         buf = [torch.zeros_like(base) for _ in range(self.world_size)]
    #         dist.all_gather(buf, base)
    #         collected = [(r, buf[r]) for r in range(self.world_size)]
    #     return collected

    def close(self):
        """
        Destroy the distributed process group and clean up resources.
        """
        print("[COMM-CLOSE]")
        dist.destroy_process_group()
