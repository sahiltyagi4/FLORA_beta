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
from concurrent import futures
from typing import Optional, Union

import grpc
import torch
from torch import nn

from . import Communicator, grpc_communicator_pb2_grpc
from .grpc_client import GrpcClient
from .grpc_server import CentralServerServicer


class GrpcCommunicator(Communicator):
    def __init__(
        self,
        model: nn.Module,
        local_rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        accumulate_updates: bool = True,
        **kwargs,
    ):
        print(f"{self.__class__.__name__} init...")

        self.model: nn.Module = model

        self.local_rank: int = local_rank
        self.world_size: int = world_size

        self.master_addr: str = master_addr
        self.master_port: int = master_port

        self.accumulate_updates: bool = accumulate_updates

        self.server: Optional[grpc.Server] = None
        self.client: Optional[GrpcClient] = None

    def setup(self):
        """Initialize the gRPC communicator."""
        print(f"{self.__class__.__name__} setup...")
        if self.local_rank == 0:
            # grpc send and receive message length max 100MB
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ],
            )

            # Add the server servicer with the model
            grpc_communicator_pb2_grpc.add_CentralServerServicer_to_server(
                CentralServerServicer(
                    self.world_size,
                    self.model,
                    accumulate_updates=self.accumulate_updates,
                ),
                self.server,
            )

            self.server.add_insecure_port(f"[::]:{self.master_port}")
            self.server.start()

            try:
                while True:
                    time.sleep(86400)
            except KeyboardInterrupt:
                print("Shutting down parameter server...")
                self.server.stop(0)

            return

        self.client = GrpcClient(
            client_id="client_" + str(self.local_rank),
            master_addr=self.master_addr,
            master_port=self.master_port,
        )

    def broadcast(
        self,
        msg: Communicator.MsgT,
        src: int = 0,
    ) -> Communicator.MsgT:
        """
        In gRPC centralized topology, this is handled by the aggregation round.
        Clients receive models via get_averaged_model in aggregate().
        """
        print(f"gRPC broadcast from src={src} | {type(msg)}")
        raise NotImplementedError()

    def aggregate(
        self,
        msg: Union[torch.nn.Module, torch.Tensor],
        batch_samples: int,
        communicate_params: bool = True,
        compute_mean: bool = True,
    ):
        # TODO: fix linting errors caused by function signature mismatch with base class

        if isinstance(msg, torch.nn.Module):
            # communicate either model parameters or gradients
            if communicate_params:
                # for weighted aggregation, summed updates are divided by total_samples on the server
                updates = {
                    name: torch.mul(param.data.detach(), batch_samples)
                    for (name, param) in msg.named_parameters()
                }
            else:
                # for weighted aggregation, summed updates are divided by total_samples on the server
                updates = {
                    name: torch.mul(param.grad.detach(), batch_samples)
                    for (name, param) in msg.named_parameters()
                }

            # if self.local_rank != 0:
            if self.client is None:
                raise RuntimeError(
                    "gRPC client is not initialized. Call setup() first."
                )

            self.client.send_update_to_server(
                updates=updates, batch_samples=batch_samples
            )
            msg = self.client.get_averaged_model(
                msg=msg, communicate_params=communicate_params
            )
            self.client.round_number += 1

        else:
            raise NotImplementedError(
                "TODO: Handle other types than torch.nn.Module for aggregation in gRPC!!!"
            )

        return msg

    def send(
        self,
        msg: Communicator.MsgT,
        dst: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        print(f"gRPC send to dst={dst} | {type(msg)}")
        raise NotImplementedError()

    def receive(
        self,
        msg: Communicator.MsgT,
        src: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        print(f"gRPC receive from src={src} | {type(msg)}")
        raise NotImplementedError()

    def collect(
        self,
        msg: Union[nn.Module, torch.Tensor, float, int],
        communicate_params: bool = True,
    ) -> list:
        print(f"gRPC collect | {type(msg)}")
        raise NotImplementedError()

    def close(self):
        """
        Clean up gRPC resources.
        """
        print("Closing gRPC communicator")

        # TODO: implement this

        print("gRPC communicator closed")
