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

from concurrent import futures

import grpc
import rich.repr
import torch
from torch import nn

from . import Communicator, grpc_communicator_pb2_grpc
from .BaseCommunicator import ReductionType
from .grpc_client import GrpcClient
from .grpc_server import CentralServerServicer


@rich.repr.auto
class GrpcCommunicator(Communicator):
    """
    Communicator implementation using gRPC client-server architecture.

    Rank 0 operates as server, higher ranks as clients. Provides broadcast
    and aggregation operations via centralized message passing.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        max_workers: int = 10,
        max_send_message_length: int = 100 * 1024 * 1024,  # 100 MB
        max_receive_message_length: int = 100 * 1024 * 1024,  # 100 MB
        aggregation_timeout: float = 600,  # Seconds for server to wait for all clients
        client_timeout: float = 60,  # Seconds for clients to wait for aggregation result
        retry_delay: float = 5.0,  # Seconds between retries
        max_retries: int = 5,  # Maximum retry attempts
        **kwargs,
    ):
        super().__init__()
        print(
            f"[COMM-INIT] rank={rank}/{world_size} | addr={master_addr}:{master_port}"
        )

        self.rank: int = rank
        self.world_size: int = world_size

        self.master_addr: str = master_addr
        self.master_port: int = master_port

        # Configuration
        self.max_workers = max_workers
        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length

        self.aggregation_timeout = aggregation_timeout
        self.client_timeout = client_timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        # Components
        self._server = None
        self._client = None
        self._servicer = None

    @property
    def client(self):
        """gRPC client instance."""
        if self._client is None:
            raise RuntimeError("gRPC client not initialized. Call setup() first.")
        return self._client

    @property
    def server(self):
        """gRPC server instance."""
        if self._server is None:
            raise RuntimeError("gRPC server not initialized. Call setup() first.")
        return self._server

    @property
    def servicer(self):
        """gRPC servicer instance."""
        if self._servicer is None:
            raise RuntimeError("gRPC servicer not initialized. Call setup() first.")
        return self._servicer

    @property
    def is_server(self) -> bool:
        """True if rank 0 (server)."""
        return self.rank == 0

    def _setup(self):
        """Initialize gRPC server or client based on rank."""
        # Setup - Ray already logs actor initialization
        if self.is_server:
            self._server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ("grpc.max_send_message_length", self.max_send_message_length),
                    (
                        "grpc.max_receive_message_length",
                        self.max_receive_message_length,
                    ),
                ],
            )

            self._servicer = CentralServerServicer(world_size=self.world_size)
            grpc_communicator_pb2_grpc.add_CentralServerServicer_to_server(
                self._servicer, self._server
            )

            self._server.add_insecure_port(f"[::]:{self.master_port}")
            print(f"[COMM-SETUP] Server listening | port {self.master_port}")
            self._server.start()
        else:
            self._client = GrpcClient(
                client_id=str(self.rank),
                master_addr=self.master_addr,
                master_port=self.master_port,
                max_send_message_length=self.max_send_message_length,
                max_receive_message_length=self.max_receive_message_length,
                retry_delay=self.retry_delay,
                max_retries=self.max_retries,
                client_timeout=self.client_timeout,
            )

    def broadcast(
        self,
        msg: Communicator.MsgT,
        src: int = 0,  # NOTE: unused here
    ) -> Communicator.MsgT:
        """Broadcast from source rank to all ranks."""
        if self.is_server:
            tensordict = self._extract_tensordict_from_msg(msg)
            print(f"[BCAST-SEND] {type(msg).__name__} | {len(tensordict)} tensors")
            self.servicer.set_broadcast_state(tensordict)
            return msg
        else:
            tensordict = self.client.get_broadcast_state()
            print(f"[BCAST-RECV] {type(msg).__name__} | {len(tensordict)} tensors")
            return self._apply_tensordict_to_msg(msg, tensordict)

    def aggregate(
        self,
        msg: Communicator.MsgT,
        reduction: ReductionType,
    ) -> Communicator.MsgT:
        """Aggregate across all ranks via central server."""
        # Extract tensors and perform aggregation
        tensordict = self._extract_tensordict_from_msg(msg)

        print(
            f"[COMM-AGG] {type(msg).__name__} | {len(tensordict)} tensors | reduction={reduction} | info={self.get_msg_info(msg)}"
        )

        aggregated_tensordict = self._grpc_aggregate(tensordict, reduction)

        # Apply result back to original message format
        return self._apply_tensordict_to_msg(msg, aggregated_tensordict)

    def _extract_tensordict_from_msg(self, msg: Communicator.MsgT) -> dict:
        """Convert message to tensor dictionary."""
        if isinstance(msg, nn.Module):
            return {
                name: param.data
                for name, param in msg.named_parameters()
                if param.requires_grad
            }
        elif isinstance(msg, dict):
            return msg
        else:
            return {"tensor": msg}

    def _apply_tensordict_to_msg(
        self, msg: Communicator.MsgT, tensordict: dict
    ) -> Communicator.MsgT:
        """Apply tensor dictionary to message."""
        if isinstance(msg, nn.Module):
            with torch.no_grad():
                for name, param in msg.named_parameters():
                    if param.requires_grad and name in tensordict:
                        param.data.copy_(tensordict[name])
            return msg
        elif isinstance(msg, dict):
            return tensordict
        else:
            return tensordict.get("tensor", msg)

    def _grpc_aggregate(self, tensordict: dict, reduction: ReductionType) -> dict:
        """Submit tensors for aggregation and retrieve result."""
        if self.is_server:
            current_session = self._submit_server_data(tensordict, reduction)
            return self._wait_for_aggregation_result(current_session)
        else:
            self.client.submit_for_aggregation(tensordict, reduction)
            return self.client.get_aggregation_result()

    def _submit_server_data(self, tensordict: dict, reduction: ReductionType) -> int:
        """Submit server data to aggregation session."""
        with self.servicer.lock:
            current_session = self.servicer.current_aggregation_session
            session_state = self.servicer.aggregation_state[current_session]

            # Set reduction type
            reduction_str = reduction.value
            if session_state["reduction_type"] is None:
                session_state["reduction_type"] = reduction_str
            elif session_state["reduction_type"] != reduction_str:
                raise ValueError(
                    f"Reduction type mismatch - expected {session_state['reduction_type']}, got {reduction_str}"
                )

            # Store server data
            session_state["data"]["server"] = tensordict
            data_count = len(session_state["data"])

            print(
                f"[COMM-AGG] Server submit | session={current_session} | {data_count}/{self.servicer.world_size}"
            )

            # Trigger aggregation if ready
            self.servicer.perform_aggregation_if_ready(session_state, current_session)

            return current_session

    def _wait_for_aggregation_result(self, session_id: int) -> dict:
        """Wait for aggregation completion and return result."""
        session_state = self.servicer.aggregation_state[session_id]

        print(f"[COMM-AGG] Server wait | session={session_id}")

        wait_result = session_state["event"].wait(timeout=self.aggregation_timeout)
        if not wait_result:
            raise RuntimeError(
                f"Aggregation timeout ({self.aggregation_timeout}s) for session {session_id}"
            )

        if session_state["result"] is None:
            raise RuntimeError(f"No result available for session {session_id}")

        return session_state["result"]

    def close(self):
        """Clean up gRPC resources."""
        print("[COMM-CLOSE]")
        if self._server is not None:
            self._server.stop(grace=15)
        if self._client is not None:
            self._client.channel.close()
