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
import numpy as np
import torch

from src.flora.communicator import (
    Communicator,
    modelupdates_pb2,
    modelupdates_pb2_grpc,
)

# proto files generated by running: python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model_updates.proto
# TODO: node with id-0 should not train, only return aggregated updates (look at aggregate fn for id-0)


class ModelService(modelupdates_pb2_grpc.ModelServiceServicer):
    def __init__(self, total_clients=1, aggregate_sum=True):
        self.model_updates = []
        self.total_clients = total_clients
        self.aggregate_sum = aggregate_sum

    def SendModelUpdate(self, request, context):
        # Convert the received parameters to a numpy array
        parameters = np.array(request.parameters)

        # Store the update for averaging
        self.model_updates.append(parameters)

        # keep processing the next request until self.total_clients are received
        # This means that the worker will keep blocking, but we won't have an explicit "wait" message sent back.
        # Returning without clearing the list so it can carry on collecting updates.
        if len(self.model_updates) >= self.total_clients:
            if self.aggregate_sum:
                averaged_parameters = np.sum(self.model_updates, axis=0)
            else:
                averaged_parameters = np.mean(self.model_updates, axis=0)

            response = modelupdates_pb2.AggregatedUpdate()
            response.averaged_parameters.extend(averaged_parameters.tolist())
            # Clear updates for the next round
            self.model_updates.clear()
            return response


class GrpcCommunicator(Communicator):
    def __init__(self, id=0, total_clients=1, host="127.0.0.1", port=50051):
        super().__init__(protocol_type="grpc")
        self.id = id
        self.total_clients = total_clients
        self.host = host
        self.port = port
        # id 0 is for the central server
        if self.id == 0:
            # TODO: change max workers proportional to total threads available
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            self.model_service = ModelService(self.total_clients)
            modelupdates_pb2_grpc.add_ModelServiceServicer_to_server(
                self.model_service, self.server
            )
            # Bind to port
            self.server.add_insecure_port("[::]:" + str(self.port))
            self.server.start()
            print("gRPC server is running...")
            try:
                self.server.wait_for_termination()
            except KeyboardInterrupt:
                self.server.stop(0)

    def aggregate(self, msg, communicate_params=True):
        # TODO: implement chunking to break model into fixed-size buckets

        if isinstance(msg, torch.nn.Module):
            if communicate_params:
                parameters = torch.cat([p.data.view(-1) for p in msg.parameters()])
            else:
                parameters = torch.cat([p.grad.view(-1) for p in msg.parameters()])
        else:
            raise TypeError("aggregate fn only supports torch.nn.Module type")

        with grpc.insecure_channel(str(self.host) + ":" + str(self.port)) as channel:
            stub = modelupdates_pb2_grpc.ModelServiceStub(channel)
            update = modelupdates_pb2.ModelUpdate()
            update.parameters.extend(parameters.flatten().tolist())

            # Send update to the server, this call blocks until the server responds
            response = stub.SendModelUpdate(update)
            # Return the averaged parameters
            return response.averaged_parameters

    def close(self):
        self.server.stop(0)
