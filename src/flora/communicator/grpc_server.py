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
from typing import Dict, Any

import rich.repr
import torch

from . import grpc_communicator_pb2
from . import grpc_communicator_pb2_grpc
from .protobuf_utils import tensordict_to_proto, proto_to_tensordict
from .BaseCommunicator import ReductionType


@rich.repr.auto
class CentralServerServicer(grpc_communicator_pb2_grpc.CentralServerServicer):
    """
    gRPC servicer for centralized federated learning aggregation.

    Manages broadcast state distribution and multi-session tensor aggregation
    with automatic session management and client synchronization.
    """

    def __init__(
        self,
        world_size: int,
    ):
        print(f"[COMM-INIT] gRPC Server | world_size={world_size}")
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
        """Get current broadcast state with tensor cloning."""
        with self.lock:
            return {
                key: tensor.clone() for key, tensor in self._broadcast_state.items()
            }

    def set_broadcast_state(self, tensordict: Dict[str, torch.Tensor]):
        """Set broadcast state for client retrieval."""
        with self.lock:
            self._broadcast_state = tensordict
        print(f"[COMM-BCAST] Stored | {len(tensordict)} tensors")

    def perform_aggregation_if_ready(
        self, session_state: Dict, current_session: int
    ) -> bool:
        """Execute aggregation when all participants ready."""
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

                if reduction_type == ReductionType.MAX.value:
                    # MAX reduction: element-wise maximum across all clients
                    for key, tensor in first_data.items():
                        all_tensors = [
                            client_data[key]
                            for client_data in session_state["data"].values()
                        ]
                        aggregated_tensors[key] = torch.max(
                            torch.stack(all_tensors), dim=0
                        )[0]

                elif reduction_type in (
                    ReductionType.SUM.value,
                    ReductionType.MEAN.value,
                ):
                    # SUM/MEAN reduction: sum all client contributions
                    for key, tensor in first_data.items():
                        aggregated_tensors[key] = torch.zeros_like(tensor)

                    # Sum all client contributions
                    for client_data in session_state["data"].values():
                        for key, tensor in client_data.items():
                            if key in aggregated_tensors:
                                aggregated_tensors[key] += tensor

                    # Convert sum to mean if needed
                    if reduction_type == ReductionType.MEAN.value:
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
        """Send broadcast state to requesting client."""
        print(f"[BCAST-REQUEST] Client {request.client_id}")

        with self.lock:
            if self._broadcast_state:
                proto_tensordict = tensordict_to_proto(self._broadcast_state)
                return grpc_communicator_pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True
                )
            else:
                return grpc_communicator_pb2.OperationResponse(is_ready=False)

    def _create_aggregation_result_response(self, session_id: int):
        """Create response with aggregated tensors."""
        if session_id in self.aggregation_state:
            session_state = self.aggregation_state[session_id]
            if session_state["result"] is not None:
                aggregated_tensors = session_state["result"]
                proto_tensordict = tensordict_to_proto(aggregated_tensors)
                return grpc_communicator_pb2.OperationResponse(
                    tensor_dict=proto_tensordict, is_ready=True
                )
        return grpc_communicator_pb2.OperationResponse(is_ready=False)

    def SubmitForAggregation(self, request, context):
        """Receive and store client tensors for aggregation."""
        with self.lock:
            client_id = request.client_id
            current_session = self.current_aggregation_session
            print(
                f"[AGG-SUBMIT] Client {client_id} submitting {len(request.tensor_dict.entries)} tensors"
            )

            try:
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
                return grpc_communicator_pb2.StatusResponse(success=True)

            except Exception as e:
                print(f"[COMM-ERROR] Submit | {e}")
                return grpc_communicator_pb2.StatusResponse(success=False)

    def GetAggregationResult(self, request, context):
        """Send aggregation result to requesting client."""
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
                return grpc_communicator_pb2.OperationResponse(is_ready=False)

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
            return grpc_communicator_pb2.OperationResponse(is_ready=False)

        except Exception as e:
            print(f"[COMM-ERROR] GetResult | {e}")
            return grpc_communicator_pb2.OperationResponse(is_ready=False)

    def RegisterClient(self, request, context):
        """Register client and track connection count."""
        with self.lock:
            self.registered_clients.add(request.client_id)
            total_clients = len(self.registered_clients)
            print(
                f"[COMM-REGISTER] Client registered | {total_clients}/{self.world_size} total"
            )
            return grpc_communicator_pb2.StatusResponse(success=True)
