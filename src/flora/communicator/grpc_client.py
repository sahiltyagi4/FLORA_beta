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
from typing import Dict

import grpc
import rich.repr
import torch

from . import AggregationOp, grpc_pb2
from . import grpc_pb2_grpc
from .utils import get_msg_info, proto_to_tensordict, tensordict_to_proto


@rich.repr.auto
class GrpcClient:
    """
    gRPC client for federated learning communication coordination.

    Connects to a central gRPC server for broadcast and aggregation operations.
    Provides automatic retry logic, timeout handling, and error recovery
    for robust distributed communication.

    Used by: GrpcCommunicator for client-side operations
    """

    def __init__(
        self,
        client_id: str,
        master_addr: str,
        master_port: int,
        max_send_message_length: int,
        max_receive_message_length: int,
        retry_delay: float = 5.0,
        max_retries: int = 3,
        client_timeout: float = 60,
    ):
        """
        Initialize gRPC client with connection and retry settings.

        Args:
            client_id: Unique identifier for this client (typically rank)
            master_addr: gRPC server address
            master_port: gRPC server port
            max_send_message_length: Maximum outbound message size in bytes
            max_receive_message_length: Maximum inbound message size in bytes
            retry_delay: Seconds between connection retry attempts
            max_retries: Maximum connection retry attempts
            client_timeout: Seconds to wait for server responses
        """
        print(f"[COMM-INIT] Client | addr={master_addr}:{master_port}")

        # Store configuration
        self.client_id = client_id
        self.master_addr = master_addr
        self.master_port = master_port
        self.max_send_message_length = max_send_message_length
        self.max_receive_message_length = max_receive_message_length
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.client_timeout = client_timeout

        # Initialize connection state
        self.channel = None
        self.stub = None

        # Establish connection with retry logic
        self._establish_connection()

    def _establish_connection(self):
        """Establish gRPC connection with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.channel = grpc.insecure_channel(
                    self.master_addr + ":" + str(self.master_port),
                    options=[
                        (
                            "grpc.max_receive_message_length",
                            self.max_receive_message_length,
                        ),
                        (
                            "grpc.max_send_message_length",
                            self.max_send_message_length,
                        ),
                    ],
                )
                self.stub = grpc_pb2_grpc.GrpcServerStub(self.channel)
                response = self.stub.RegisterClient(
                    grpc_pb2.ClientInfo(client_id=self.client_id),
                )
                print(f"[COMM-CLIENT] Register | success={response.success}")
                return

            except grpc.RpcError as e:
                if attempt >= self.max_retries:
                    print(
                        f"[COMM-ERROR] Connection failed after {self.max_retries} attempts"
                    )
                    raise e

                print(
                    f"[COMM-ERROR] Connection retry | attempt {attempt}/{self.max_retries} | {self.retry_delay}s delay"
                )
                time.sleep(self.retry_delay)

    def get_broadcast_state(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve broadcast state from server with polling and retry logic.

        Continuously polls server until broadcast state is available.
        Used during broadcast operations to receive global model.

        Returns:
            Dictionary mapping parameter names to tensor values
        """
        print(f"[BCAST-REQUEST] Waiting for server to broadcast model")

        poll_count = 0
        error_count = 0

        while True:
            try:
                request = grpc_pb2.ClientInfo(client_id=self.client_id)
                response = self.stub.GetBroadcastState(request)
                if response.is_ready:
                    tensordict = proto_to_tensordict(response.tensor_dict)
                    print(f"[BCAST-RECEIVED] {get_msg_info(tensordict)}")
                    return tensordict
                poll_count += 1
                print(
                    f"[COMM-BCAST] Waiting | poll {poll_count} | retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)

            except grpc.RpcError as e:
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Max retries ({self.max_retries}) exceeded for broadcast state"
                    )
                print(
                    f"[COMM-ERROR] Broadcast fetch | error {error_count}/{self.max_retries} | retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)

    def submit_for_aggregation(
        self, tensordict: Dict[str, torch.Tensor], reduction_type: AggregationOp
    ):
        """
        Submit local tensors to server for distributed aggregation.

        Args:
            tensordict: Local tensors to contribute to aggregation
            reduction_type: SUM, MEAN, or MAX aggregation operation
        """
        try:
            proto_tensordict = tensordict_to_proto(tensordict)
            request = grpc_pb2.AggregationRequest(
                client_id=self.client_id,
                tensor_dict=proto_tensordict,
                reduction_type=reduction_type.value,
            )
            response = self.stub.SubmitForAggregation(request)
            if response.success:
                print(f"[AGG-SUBMIT] Successfully sent local model to server")
            else:
                print(f"[COMM-ERROR] Submit failed")
        except grpc.RpcError as e:
            print(f"[COMM-ERROR] Submit exception | {e}")

    def get_aggregation_result(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve aggregated result from server with timeout and polling.

        Waits for server to complete aggregation across all clients,
        then returns the aggregated tensors.

        Returns:
            Dictionary mapping parameter names to aggregated tensor values

        Raises:
            RuntimeError: If aggregation times out or max retries exceeded
        """
        print(
            f"[AGG-WAIT] Waiting for server to aggregate models (timeout={self.client_timeout}s)"
        )

        start_time = time.time()
        poll_count = 0
        error_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.client_timeout:
                raise RuntimeError(f"Aggregation timeout ({self.client_timeout}s)")
            try:
                request = grpc_pb2.ClientInfo(client_id=self.client_id)
                response = self.stub.GetAggregationResult(request)
                if response.is_ready:
                    tensordict = proto_to_tensordict(response.tensor_dict)
                    print(
                        f"[AGG-RECEIVED] {get_msg_info(tensordict)} (waited {elapsed:.1f}s)"
                    )
                    return tensordict
                poll_count += 1
                remaining = self.client_timeout - elapsed
                print(
                    f"[COMM-AGG] Waiting | poll {poll_count} | {remaining:.1f}s remaining"
                )
                time.sleep(min(self.retry_delay, remaining))

            except grpc.RpcError as e:
                error_count += 1
                if error_count > self.max_retries:
                    raise RuntimeError(
                        f"Max retries ({self.max_retries}) exceeded for aggregation result"
                    )
                print(
                    f"[COMM-ERROR] Aggregation fetch | error {error_count}/{self.max_retries} | retry in {self.retry_delay}s"
                )
                time.sleep(self.retry_delay)
