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

from . import BaseCommunicator, grpc_pb2_grpc
from .base import AggregationOp
from .grpc_client import GrpcClient
from .grpc_server import GrpcServer
from .utils import get_msg_info


@rich.repr.auto
class GrpcCommunicator(BaseCommunicator):
    """
    gRPC-based communication backend with centralized coordination.

    Uses client-server architecture where rank 0 acts as coordinator for
    broadcast and aggregation operations.
    Provides reliable communication across heterogeneous networks with
    built-in retry and timeout handling.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str = "127.0.0.1",
        master_port: int = 50051,
        max_workers: int = 10,
        max_send_message_length: int = 104857600,  # 100 MB
        max_receive_message_length: int = 104857600,  # 100 MB
        aggregation_timeout: float = 600.0,
        client_timeout: float = 60.0,
        retry_delay: float = 5.0,
        max_retries: int = 5,
        **kwargs,
    ) -> None:
        """
        Initialize gRPC-based federated learning communicator.

        Args:
            rank: Process rank (0 becomes server, others become clients)
            world_size: Total number of participants in FL experiment
            master_addr: gRPC server address (rank 0 listens here)
            master_port: gRPC server port
            max_workers: Thread pool size for gRPC server
            max_send_message_length: Maximum outbound message size in bytes
            max_receive_message_length: Maximum inbound message size in bytes
            aggregation_timeout: Server timeout waiting for all clients (seconds)
            client_timeout: Client timeout waiting for aggregation result (seconds)
            retry_delay: Seconds between connection retry attempts
            max_retries: Maximum connection retry attempts
        """
        super().__init__()
        print(
            f"[COMM-INIT] rank={rank}/{world_size} | addr={master_addr}:{master_port}"
        )

        # Core distributed parameters
        self.rank: int = rank
        self.world_size: int = world_size
        self.master_addr: str = master_addr
        self.master_port: int = master_port

        # gRPC configuration
        self.max_workers = max_workers
        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length

        # Timeout and retry settings
        self.aggregation_timeout = aggregation_timeout
        self.client_timeout = client_timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        # Runtime components (initialized in _setup)
        self._server = None
        self._client = None
        self._servicer = None

    @property
    def client(self):
        """
        gRPC client instance for non-server ranks.

        Returns:
            GrpcClient instance for communication with server

        Raises:
            RuntimeError: If accessed before setup() or on server rank
        """
        if self._client is None:
            raise RuntimeError("gRPC client not initialized. Call setup() first.")
        return self._client

    @property
    def server(self):
        """
        gRPC server instance for rank 0.

        Returns:
            grpc.Server instance handling client connections

        Raises:
            RuntimeError: If accessed before setup() or on client rank
        """
        if self._server is None:
            raise RuntimeError("gRPC server not initialized. Call setup() first.")
        return self._server

    @property
    def servicer(self):
        """
        gRPC servicer instance containing FL coordination logic.

        Returns:
            CentralServerServicer managing aggregation and broadcast state

        Raises:
            RuntimeError: If accessed before setup() or on client rank
        """
        if self._servicer is None:
            raise RuntimeError("gRPC servicer not initialized. Call setup() first.")
        return self._servicer

    @property
    def is_server(self) -> bool:
        """
        True if this rank acts as the central gRPC server.

        Returns:
            True for rank 0, False for all other ranks
        """
        return self.rank == 0

    def _setup(self):
        """
        Initialize gRPC server (rank 0) or client (other ranks).

        Server setup: Creates gRPC server, servicer, and starts listening
        Client setup: Creates gRPC client and connects to server
        """
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

            self._servicer = GrpcServer(world_size=self.world_size)
            grpc_pb2_grpc.add_GrpcServerServicer_to_server(self._servicer, self._server)

            self._server.add_insecure_port(f"[::]:{self.master_port}")
            print(f"[COMM-SETUP] Server listening on port {self.master_port}")
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
        msg: BaseCommunicator.MsgT,
        src: int = 0,
    ) -> BaseCommunicator.MsgT:
        """
        Broadcast message from server to all clients via gRPC.

        Server (rank 0): Stores message in broadcast state for client retrieval
        Clients: Retrieve broadcast state from server via polling

        Args:
            msg: Model, tensor dict, or tensor to broadcast
            src: Source rank (unused in centralized gRPC - always rank 0)

        Returns:
            Broadcasted message with updated values
        """
        if self.is_server:
            # Server: Store broadcast state for client retrieval
            tensordict = self._extract_tensordict_from_msg(msg)
            print(f"[BCAST-SEND] {get_msg_info(msg)} | src={src}")
            self.servicer.set_broadcast_state(tensordict)
            return msg
        else:
            # Client: Retrieve broadcast state from server
            tensordict = self.client.get_broadcast_state()
            print(f"[BCAST-RECV] {get_msg_info(msg)} | src={src}")
            return self._apply_tensordict_to_msg(msg, tensordict)

    def aggregate(
        self,
        msg: BaseCommunicator.MsgT,
        reduction: AggregationOp,
    ) -> BaseCommunicator.MsgT:
        """
        Aggregate message across all ranks via central gRPC server.

        All ranks submit their data to the server, which performs aggregation
        when all participants have contributed, then distributes results.

        Args:
            msg: Model, tensor dict, or tensor to aggregate
            reduction: SUM, MEAN, or MAX aggregation operation

        Returns:
            Aggregated message with combined values from all ranks
        """
        # Extract tensors and perform distributed aggregation
        tensordict = self._extract_tensordict_from_msg(msg)
        print(f"[COMM-AGG] {get_msg_info(msg)} | reduction={reduction}")

        # Perform aggregation via gRPC protocol
        aggregated_tensordict = self._grpc_aggregate(tensordict, reduction)

        # Apply aggregated results back to original message format
        return self._apply_tensordict_to_msg(msg, aggregated_tensordict)

    def _extract_tensordict_from_msg(self, msg: BaseCommunicator.MsgT) -> dict:
        """
        Extract tensors from message for gRPC serialization.

        Args:
            msg: Model, tensor dict, or single tensor

        Returns:
            Dictionary mapping parameter names to tensor values
        """
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
        self, msg: BaseCommunicator.MsgT, tensordict: dict
    ) -> BaseCommunicator.MsgT:
        """
        Apply deserialized tensors back to original message format.

        Args:
            msg: Original message providing structure and device info
            tensordict: Deserialized tensors from gRPC

        Returns:
            Message with updated tensor values and proper device placement
        """
        if isinstance(msg, nn.Module):
            with torch.no_grad():
                for name, param in msg.named_parameters():
                    if param.requires_grad and name in tensordict:
                        # Ensure tensor is on the same device as the parameter before copying
                        tensor = tensordict[name].to(param.device)
                        param.data.copy_(tensor)
            return msg
        elif isinstance(msg, dict):
            return tensordict
        else:
            return tensordict.get("tensor", msg)

    def _grpc_aggregate(self, tensordict: dict, reduction: AggregationOp) -> dict:
        """
        Perform distributed aggregation via gRPC protocol.

        Server: Submits data to local session and waits for aggregation
        Clients: Submit data to server and retrieve aggregated result

        Args:
            tensordict: Local tensors to contribute
            reduction: Aggregation operation type

        Returns:
            Aggregated tensor dictionary
        """
        if self.is_server:
            current_session = self._submit_server_data(tensordict, reduction)
            return self._wait_for_aggregation_result(current_session)
        else:
            self.client.submit_for_aggregation(tensordict, reduction)
            return self.client.get_aggregation_result()

    def _submit_server_data(self, tensordict: dict, reduction: AggregationOp) -> int:
        """
        Submit server's local data to current aggregation session.

        Args:
            tensordict: Server's local tensors
            reduction: Aggregation operation type

        Returns:
            Session ID for tracking aggregation progress
        """
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
        """
        Wait for aggregation to complete and return aggregated tensors.

        Args:
            session_id: Aggregation session identifier

        Returns:
            Aggregated tensor dictionary

        Raises:
            RuntimeError: If aggregation times out or fails
        """
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
        """
        Clean up gRPC resources and close connections.

        Gracefully shuts down server (rank 0) or closes client connections.
        Should be called when communication is no longer needed.
        """
        print("[COMM-CLOSE]")
        if self._server is not None:
            self._server.stop(grace=15)
        if self._client is not None:
            self._client.channel.close()
