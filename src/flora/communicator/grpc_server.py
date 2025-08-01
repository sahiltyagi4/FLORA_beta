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

import threading
from collections import defaultdict
from typing import Any, Dict

import rich.repr
import torch

from . import grpc_pb2, grpc_pb2_grpc
from .base import AggregationOp
from .utils import get_msg_info, proto_to_tensordict, tensordict_to_proto


@rich.repr.auto
class GrpcServer(grpc_pb2_grpc.GrpcServerServicer):
    """
    gRPC server implementation for federated learning communication coordination.

    Coordinates broadcast and aggregation operations across multiple clients.
    Handles session management, client synchronization, and tensor aggregation
    with support for multiple concurrent FL rounds.
    """

    def __init__(
        self,
        world_size: int,
    ):
        """
        Initialize gRPC server for federated learning coordination.

        Args:
            world_size: Total number of FL participants (including server)
        """
        print(f"[COMM-INIT] gRPC Server | world_size={world_size}")

        # Core configuration
        self.world_size = world_size
        self.registered_clients = set()
        self.lock = threading.Lock()

        # Aggregation session management
        self.current_aggregation_session = 0
        self.aggregation_state: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "data": {},
                "result": None,
                "event": threading.Event(),
                "reduction_type": None,
            }
        )

        # Broadcast state storage
        self._broadcast_state = {}

        print(f"[COMM-READY] Server listening for {world_size} clients")

    def get_broadcast_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current broadcast state with thread-safe tensor cloning.

        Returns:
            Deep copy of broadcast tensors to prevent race conditions
        """
        with self.lock:
            return {
                key: tensor.clone() for key, tensor in self._broadcast_state.items()
            }

    def set_broadcast_state(self, tensordict: Dict[str, torch.Tensor]):
        """
        Store broadcast state for distribution to clients.

        Args:
            tensordict: Tensors to broadcast (typically global model)
        """
        with self.lock:
            self._broadcast_state = tensordict
        print(f"[COMM-BCAST] Stored | {get_msg_info(tensordict)}")

    def perform_aggregation_if_ready(
        self, session_state: Dict, current_session: int
    ) -> bool:
        """
        Execute aggregation when all clients have submitted data.

        Args:
            session_state: Current aggregation session data
            current_session: Session identifier

        Returns:
            True if aggregation was performed, False if still waiting
        """
        submitted_count = len(session_state["data"])

        print(
            f"[AGG-STATUS] Waiting for clients ({submitted_count}/{self.world_size} ready)"
        )

        if submitted_count == self.world_size:
            print(
                f"[AGG-START] All {self.world_size} clients ready - beginning aggregation"
            )

            first_data = next(iter(session_state["data"].values()))
            aggregated_tensors = {}

            with torch.no_grad():
                reduction_type = session_state["reduction_type"]
                if reduction_type is None:
                    raise ValueError(
                        f"No reduction type set for session {current_session}"
                    )

                # Initialize aggregated tensors for each key
                aggregated_tensors = {}

                if reduction_type == AggregationOp.MAX.value:
                    # MAX reduction: element-wise maximum
                    for key, tensor in first_data.items():
                        all_tensors = [
                            client_data[key]
                            for client_data in session_state["data"].values()
                        ]
                        aggregated_tensors[key] = torch.max(
                            torch.stack(all_tensors), dim=0
                        )[0]

                elif reduction_type in (
                    AggregationOp.SUM.value,
                    AggregationOp.MEAN.value,
                ):
                    # SUM/MEAN reduction: sum all contributions
                    for key, tensor in first_data.items():
                        aggregated_tensors[key] = torch.zeros_like(tensor)

                    # Sum all client contributions
                    for client_data in session_state["data"].values():
                        for key, tensor in client_data.items():
                            if key in aggregated_tensors:
                                aggregated_tensors[key] += tensor

                    # Convert sum to mean if needed
                    if reduction_type == AggregationOp.MEAN.value:
                        for tensor in aggregated_tensors.values():
                            tensor /= self.world_size

                else:
                    raise ValueError(f"Unknown reduction type: {reduction_type}")

            session_state["result"] = aggregated_tensors
            session_state["event"].set()
            print(
                f"[AGG-COMPLETE] Aggregated {len(aggregated_tensors)} tensors using {reduction_type}"
            )
            self.current_aggregation_session += 1
            return True
        return False

    def GetBroadcastState(self, request, context):
        """
        gRPC endpoint: Send broadcast state to requesting client.

        Args:
            request: ClientInfo with client identifier
            context: gRPC context (unused)

        Returns:
            OperationResponse with tensor data or not-ready status
        """
        print(f"[BCAST-REQUEST] Client {request.client_id}")

        with self.lock:
            if self._broadcast_state:
                proto_tensordict = tensordict_to_proto(self._broadcast_state)
                return grpc_pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True
                )
            else:
                return grpc_pb2.OperationResponse(is_ready=False)

    def _create_aggregation_result_response(self, session_id: int):
        """
        Create gRPC response containing aggregated tensor results.

        Args:
            session_id: Aggregation session identifier

        Returns:
            OperationResponse with aggregated tensors or not-ready status
        """
        if session_id in self.aggregation_state:
            session_state = self.aggregation_state[session_id]
            if session_state["result"] is not None:
                aggregated_tensors = session_state["result"]
                proto_tensordict = tensordict_to_proto(aggregated_tensors)
                return grpc_pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True
                )
        return grpc_pb2.OperationResponse(is_ready=False)

    def SubmitForAggregation(self, request, context):
        """
        gRPC endpoint: Receive client tensors for distributed aggregation.

        Stores client contributions and triggers aggregation when all
        clients have submitted their data.

        Args:
            request: AggregationRequest with client data and reduction type
            context: gRPC context (unused)

        Returns:
            StatusResponse indicating success or failure
        """
        with self.lock:
            client_id = request.client_id
            current_session = self.current_aggregation_session
            print(
                f"[AGG-SUBMIT] Client {client_id} submitting {len(request.tensor_dict.entries)} tensors"
            )

            try:
                # Deserialize tensors (will be on CPU for consistent aggregation)
                data = proto_to_tensordict(request.tensor_dict)
                session_state = self.aggregation_state[current_session]

                if session_state["reduction_type"] is None:
                    session_state["reduction_type"] = request.reduction_type
                    print(
                        f"[COMM-AGG] Session | {current_session} | reduction={request.reduction_type}"
                    )
                elif session_state["reduction_type"] != request.reduction_type:
                    print(
                        f"[COMM-ERROR] Reduction mismatch | expected={session_state['reduction_type']} got={request.reduction_type}"
                    )

                session_state["data"][client_id] = data
                data_count = len(session_state["data"])
                print(
                    f"[AGG-COLLECT] Received from client {client_id} ({data_count}/{self.world_size} ready)"
                )

                self.perform_aggregation_if_ready(session_state, current_session)
                return grpc_pb2.StatusResponse(success=True)

            except Exception as e:
                print(f"[COMM-ERROR] Submit | {e}")
                return grpc_pb2.StatusResponse(success=False)

    def GetAggregationResult(self, request, context):
        """
        gRPC endpoint: Send aggregation result to requesting client.

        Waits for aggregation to complete if necessary, then returns
        the aggregated tensors to the requesting client.

        Args:
            request: ClientInfo with client identifier
            context: gRPC context (unused)

        Returns:
            OperationResponse with aggregated tensors or error status
        """
        client_id = request.client_id

        with self.lock:
            target_session = None
            for session_id in sorted(self.aggregation_state.keys(), reverse=True):
                if client_id in self.aggregation_state[session_id]["data"]:
                    target_session = session_id
                    break
            if target_session is None:
                print(
                    f"[COMM-ERROR] GetResult | client={client_id} | no submission found"
                )
                return grpc_pb2.OperationResponse(is_ready=False)

        print(f"[AGG-REQUEST] Client {client_id} requesting aggregation result")

        try:
            session_state = self.aggregation_state[target_session]
            with self.lock:
                if session_state["result"] is not None:
                    print(f"[AGG-SEND] Sending aggregated model to client {client_id}")
                    return self._create_aggregation_result_response(target_session)

            print(f"[AGG-WAIT] Client {client_id} waiting for aggregation to complete")
            session_state["event"].wait()
            print(f"[AGG-READY] Aggregation complete for client {client_id}")
            with self.lock:
                if session_state["result"] is not None:
                    print(f"[AGG-SEND] Sending aggregated model to client {client_id}")
                    return self._create_aggregation_result_response(target_session)
            return grpc_pb2.OperationResponse(is_ready=False)

        except Exception as e:
            print(f"[COMM-ERROR] GetResult | {e}")
            return grpc_pb2.OperationResponse(is_ready=False)

    def RegisterClient(self, request, context):
        """
        gRPC endpoint: Register client connection and track participant count.

        Args:
            request: ClientInfo with unique client identifier
            context: gRPC context (unused)

        Returns:
            StatusResponse confirming successful registration
        """
        with self.lock:
            self.registered_clients.add(request.client_id)
            total_clients = len(self.registered_clients)
            print(
                f"[COMM-REGISTER] Client registered | {total_clients}/{self.world_size} total"
            )
            return grpc_pb2.StatusResponse(success=True)
