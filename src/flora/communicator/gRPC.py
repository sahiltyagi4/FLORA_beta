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
import grpc
from concurrent import futures
from typing import Union

from torch import nn
import torch
from . import Communicator
from . import gRPC_pb2_grpc
from .gRPC_Server import CentralServerServicer
from .gRPC_Client import GrpcClient


class gRPC(Communicator):
    """
    gRPC-based communicator for federated learning.

    Implements the unified Communicator interface for gRPC client-server communication.
    Supports centralized topology with server acting as aggregator.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        accumulate_updates: bool = True,
        model: nn.Module = None,  # Optional model, can be set later
        **kwargs,
    ):
        """
        Initialize gRPC communicator.

        Args:
            rank: Node rank (0 for server, >0 for clients)
            world_size: Total number of participants
            master_addr: Server address
            master_port: Server port
            accumulate_updates: Whether to accumulate updates for weighted aggregation
            model: Model for server (required if rank == 0)
        """
        print(f"{self.__class__.__name__} init...")
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.accumulate_updates = accumulate_updates
        self.is_server = rank == 0
        self.total_clients = world_size - 1  # excluding server
        self.model = model

        # Will be initialized in setup()
        self.server = None
        self.client = None
        self._setup_complete = False

    def setup(self):
        """
        Initialize the gRPC communication layer.
        """
        if self._setup_complete:
            print("gRPC communicator already set up")
            return

        print(
            f"Setting up gRPC communicator - rank {self.rank}, is_server: {self.is_server}"
        )

        if self.is_server:
            # Server setup will be deferred until model is available
            print("gRPC server setup deferred until model is available")

        else:
            # Setup client
            client_id = f"client_{self.rank}"
            self.client = GrpcClient(
                client_id=client_id,
                master_addr=self.master_addr,
                master_port=self.master_port,
            )
            print(f"gRPC client {client_id} configured")

        self._setup_complete = True

    def _ensure_server_setup(self, model: nn.Module):
        """
        Ensure gRPC server is set up with the provided model.
        This is called lazily when the model becomes available.
        """
        if self.is_server and self.server is None:
            print("Setting up gRPC server with model")
            self.model = model
            
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ],
            )

            # Add the server servicer with the model
            gRPC_pb2_grpc.add_CentralServerServicer_to_server(
                CentralServerServicer(
                    self.total_clients,
                    self.model,
                    accumulate_updates=self.accumulate_updates,
                ),
                self.server,
            )

            listen_addr = f"[::]:{self.master_port}"
            self.server.add_insecure_port(listen_addr)
            print(f"gRPC server starting on {listen_addr}")
            self.server.start()
            print("gRPC server started successfully")

    def broadcast(
        self,
        msg: Communicator.MsgT,
        src: int = 0,
    ) -> Communicator.MsgT:
        """
        Broadcast a model to all nodes.

        In gRPC centralized topology, this is handled by the aggregation round.
        Clients receive models via get_averaged_model in aggregate().
        """
        print(f"gRPC broadcast from rank {src} | {type(msg)}")

        if not self.is_server and src == 0:
            # Client receiving broadcast - this happens during aggregate()
            # For now, just return the message as gRPC handles this differently
            return msg
        elif self.is_server:
            # Server has the global model to broadcast
            return msg
        else:
            raise NotImplementedError(
                "gRPC only supports server-initiated broadcasts (src=0)"
            )

    def aggregate(
        self,
        msg: Communicator.MsgT,
        communicate_params: bool = True,
        compute_mean: bool = True,
        batch_samples: int = None,
        **kwargs,
    ) -> Communicator.MsgT:
        """
        Aggregate model across nodes using gRPC.

        Args:
            msg: Model or tensor to aggregate
            communicate_params: Whether to communicate parameters (True) or gradients (False)
            compute_mean: Whether to compute weighted average (only used with batch_samples)
            batch_samples: Number of samples for weighted aggregation
        """
        print(f"gRPC aggregate | {type(msg)} communicate_params={communicate_params}")

        # Handle both nn.Module and torch.Tensor aggregation
        if isinstance(msg, torch.Tensor):
            # For tensor aggregation (like sample counts), simulate aggregation
            # In a real gRPC implementation, this would need to be implemented
            print(f"gRPC tensor aggregation not fully implemented, returning original tensor")
            return msg
        elif not isinstance(msg, nn.Module):
            raise NotImplementedError(
                "gRPC communicator currently only supports nn.Module and torch.Tensor aggregation"
            )

        # Ensure server is set up with the model (for lazy initialization)
        self._ensure_server_setup(msg)

        if self.is_server:
            # Server doesn't need to do anything during aggregation
            # The actual aggregation happens in the server servicer
            return msg
        else:
            # Client sends update and receives aggregated model
            if batch_samples is None:
                batch_samples = 1  # Default fallback

            # Prepare updates
            if communicate_params:
                updates = {
                    name: torch.mul(param.data.detach(), batch_samples)
                    for name, param in msg.named_parameters()
                }
            else:
                updates = {
                    name: torch.mul(param.grad.detach(), batch_samples)
                    for name, param in msg.named_parameters()
                }

            # Send update to server and get aggregated model back
            self.client.send_update_to_server(
                updates=updates, batch_samples=batch_samples
            )
            updated_msg = self.client.get_averaged_model(
                msg=msg, communicate_params=communicate_params
            )
            self.client.round_number += 1
            return updated_msg

    def send(
        self,
        msg: Communicator.MsgT,
        dst: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        """
        Send model to a specific destination.

        Note: gRPC is designed for centralized communication.
        Direct peer-to-peer send is not implemented.
        """
        print(f"gRPC send to rank {dst} | {type(msg)}")
        raise NotImplementedError()

    def receive(
        self,
        msg: Communicator.MsgT,
        src: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        """
        Receive model from a specific source.

        Note: gRPC is designed for centralized communication.
        Direct peer-to-peer receive is not implemented.
        """
        print(f"gRPC receive from rank {src} | {type(msg)}")
        raise NotImplementedError()

    def collect(
        self,
        msg: Union[nn.Module, torch.Tensor, float, int],
        communicate_params: bool = True,
    ) -> list:
        """
        Gather data from all ranks.

        Note: gRPC aggregation is handled through the aggregate() method.
        This collect operation is not directly supported.
        """
        print(f"gRPC collect | {type(msg)}")
        raise NotImplementedError()

    def close(self):
        """
        Clean up gRPC resources.
        """
        print("Closing gRPC communicator")

        if self.server is not None:
            print("Stopping gRPC server")
            self.server.stop(0)
            self.server = None

        if self.client is not None:
            print("Cleaning up gRPC client")
            # Client cleanup if needed
            self.client = None

        self._setup_complete = False
        print("gRPC communicator closed")
