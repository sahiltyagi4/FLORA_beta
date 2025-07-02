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

import torch
import torch.distributed as dist
import torch.nn as nn

from .BaseCommunicator import Communicator


# ======================================================================================


class TorchDist(Communicator):
    """
    PyTorch distributed communicator
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
        timeout: int = 10,
        max_retries: int = 3,
        **kwargs,
    ):
        print(f"{self.__class__.__name__} init...")
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

        # Fallback if necessary
        if self.backend == "nccl" and not torch.cuda.is_available():
            print("NCCL backend requested but CUDA not available, falling back to gloo")
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

    def setup(self):
        """
        Initialize PyTorch distributed process group using the selected init_method.
        """
        # TODO: Check if this is already done internatlly in dist.init_process_group
        # if dist.is_initialized():
        #     print(f"Process group already initialized")
        #     return

        print(
            f"setup: rank={self.rank}, world_size={self.world_size}, init_method={self.init_method}, backend={self.backend}, master_addr={self.master_addr}, master_port={self.master_port}, sharedfile={self.sharedfile}",
            flush=True,
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
        :param msg: message to broadcast
        :param id: node id which initiates the broadcast
        :return: returns the broadcasted message
        """
        print(f"Broadcast from rank {src} to all ranks | {type(msg)}", flush=True)

        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if p.requires_grad:
                    dist.broadcast(p.data, src=src)
        elif isinstance(msg, dict):
            # Handle Dict[str, torch.Tensor] for parameter deltas, control variates, etc.
            for tensor in msg.values():
                dist.broadcast(tensor, src=src)
        else:
            dist.broadcast(msg, src=src)
        return msg

    def aggregate(
        self,
        msg: Communicator.MsgT,
        communicate_params: bool = True,
        compute_mean: bool = True,
        **kwargs,
    ) -> Communicator.MsgT:
        """
        :param msg: message to aggregate
        :param communicate_params: collect model parameters if True, else aggregate model gradients
        :return: aggregated message
        """
        print(
            f"Aggregate from all ranks | {type(msg)} communicate_params={communicate_params} compute_mean={compute_mean}",
            flush=True,
        )

        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if not p.requires_grad:
                    continue
                tensor = p.data if communicate_params else p.grad
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                if compute_mean:
                    tensor.div_(self.world_size)
        elif isinstance(msg, dict):
            # Handle Dict[str, torch.Tensor] for parameter deltas, control variates, etc.
            for name, tensor in msg.items():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                if compute_mean:
                    tensor.div_(self.world_size)
        else:
            dist.all_reduce(msg, op=dist.ReduceOp.SUM)
            if compute_mean:
                msg.div_(self.world_size)
        return msg

    def send(
        self,
        msg: Communicator.MsgT,
        dst: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        """
        :param msg: message to send
        :param id: client or server id ranging from 0 to (total_clients - 1)
        :param communicate_params: collect model parameters if True, else aggregate model gradients
        :return: the sending message
        """
        print(
            f"Send to rank {dst} | {type(msg)} communicate_params={communicate_params}",
            flush=True,
        )

        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if not p.requires_grad:
                    continue

                tensor = p.data if communicate_params else p.grad
                dist.send(tensor, dst=dst)
        else:
            dist.send(msg, dst=dst)
        return msg

    def receive(
        self,
        msg: Communicator.MsgT,
        src: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        """
        :param msg: message to receive
        :param id: client or server id ranging from 0 to (total_clients - 1)
        :param communicate_params: collect model parameters if True, else aggregate model gradients
        :return: the receiving message
        """
        print(
            f"Receive from rank {src} | {type(msg)} communicate_params={communicate_params}",
            flush=True,
        )

        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if not p.requires_grad:
                    continue

                tensor = p.data if communicate_params else p.grad
                dist.recv(tensor, src=src)
        else:
            dist.recv(msg, src=src)
        return msg

    def collect(
        self,
        msg: Union[nn.Module, torch.Tensor, float, int],
        communicate_params: bool = True,
    ) -> list[tuple[int, Communicator.MsgT]]:
        """
         all-gather in decentralized MPI collectives
        :param msg: message to receive
        :param id: client_id specifying the client update comes from. redundant in MPI communication as all_gather
        collects by rank ids
        :param communicate_params: collect model parameters if True, else send model gradients
        :return: either nested list of layerwise model data collected from clients or a simple list of gathered data
        """
        print(
            f"Collect from all ranks | {type(msg)} communicate_params={communicate_params}",
            flush=True,
        )

        collected = []
        if isinstance(msg, nn.Module):
            for _, p in msg.named_parameters():
                if not p.requires_grad:
                    continue
                buf = [torch.zeros_like(p.data) for _ in range(self.world_size)]
                tensor = p.data if communicate_params else p.grad
                dist.all_gather(buf, tensor)
                collected.append([(r, buf[r]) for r in range(self.world_size)])
        else:
            base = torch.tensor(msg)
            buf = [torch.zeros_like(base) for _ in range(self.world_size)]
            dist.all_gather(buf, base)
            collected = [(r, buf[r]) for r in range(self.world_size)]
        return collected

    def close(self):
        """
        Clean up the process group.
        """
        # if not dist.is_initialized():
        #     print(f"Process group not initialized, nothing to close")
        #     return
        print("Destroying process group")
        dist.destroy_process_group()
        print("Process group destroyed successfully")
