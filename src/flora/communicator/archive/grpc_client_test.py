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

import sys
import time
import grpc
import random

import torch
import numpy as np

import src.flora.communicator.grpc_communicator_pb2 as flora_grpc_pb2
import src.flora.communicator.grpc_communicator_pb2_grpc as flora_grpc_pb2_grpc


class SimpleModel(torch.nn.Module):
    """Simple neural network for demonstration"""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CentralServerClient:
    def __init__(
        self,
        client_id: str,
        batch_samples: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        communicate_params: bool = True,
        compute_mean: bool = True,
        server_address: str = "localhost:50051",
    ):
        self.client_id = client_id
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = flora_grpc_pb2_grpc.CentralServerStub(self.channel)
        self.batch_samples = batch_samples
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.communicate_params = communicate_params
        self.compute_mean = compute_mean
        self.round_number = 0

        print(f"Client {client_id} initialized, connecting to {server_address}")
        self._register_with_server()

    def _register_with_server(self):
        """Register this client with the parameter server"""
        try:
            request = flora_grpc_pb2.ClientInfo(client_id=self.client_id)
            response = self.stub.RegisterClient(request)

            if response.success:
                print(
                    f"Successfully registered with server. Total clients: {response.total_clients}"
                )
            else:
                print(f"Failed to register with server: {response.message}")

        except grpc.RpcError as e:
            print(f"Failed to connect to server: {e}")

    def generate_dummy_data(self, batch_size=32):
        """Generate dummy training data"""
        x = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))
        return x, y

    def train_local_epoch(self, num_batches=5):
        """Train model locally for one epoch"""
        self.model.train()
        total_loss = 0.0
        for _ in range(num_batches):
            x, y = self.generate_dummy_data()
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(
            f"Round {self.round_number}: Local training completed. Average loss: {avg_loss:.4f}"
        )
        return avg_loss

    def _model_params_to_protobuf(self):
        """Convert model parameters to protobuf format"""
        proto_layers = []
        for name, param in self.model.named_parameters():
            # param = param.cpu().detach().numpy()
            param = param.cpu()
            layer_proto = flora_grpc_pb2.LayerState(layer_name=name)
            if self.communicate_params:
                # for weighted aggregation, summed updates are divided by self.total_samples on the server
                if self.compute_mean:
                    param.data = torch.mul(param.data, self.batch_samples)

                layer_proto.param_update.extend(param.data.flatten().tolist())
                layer_proto.param_shape.extend(list(param.data.shape))

                # if self.client_id == 'client_1':
                #     layer_proto.param_update.extend(torch.ones_like(param.data).flatten().tolist())
                # elif self.client_id == 'client_2':
                #     layer_proto.param_update.extend(torch.zeros_like(param.data).flatten().tolist())
                #
                # layer_proto.param_shape.extend(list(param.data.shape))
            else:
                # for weighted aggregation, summed updates are divided by self.total_samples on the server
                param.grad = torch.mul(param.grad, self.batch_samples)
                layer_proto.param_update.extend(param.grad.flatten().tolist())
                layer_proto.param_shape.extend(list(param.grad.shape))

                # layer_proto.param_update.extend(torch.ones_like(param.grad).flatten().tolist())
                # layer_proto.param_shape.extend(list(param.grad.shape))

            # print(f"DEBUGGING CLIENT {self.client_id} layer_proto: {layer_proto.param_update} with shape: {layer_proto.param_shape}")
            proto_layers.append(layer_proto)

        return proto_layers

    def send_update_to_server(self):
        """Send model update to parameter server"""
        try:
            proto_layers = self._model_params_to_protobuf()

            request = flora_grpc_pb2.ModelUpdate(
                client_id=self.client_id,
                round_number=self.round_number,
                layers=proto_layers,
                number_samples=self.batch_samples,
            )

            response = self.stub.SendUpdate(request)

            if response.success:
                print(
                    f"Round {self.round_number}: Update sent successfully. "
                    f"Updates received: {response.updates_received}/{response.clients_registered}"
                )
                return True
            else:
                print(f"Failed to send update: {response.message}")
                return False

        except grpc.RpcError as e:
            print(f"Failed to send update to server: {e}")
            return False

    def _update_model_from_protobuf(self, proto_layers):
        """Update model parameters from protobuf format"""
        with torch.no_grad():
            for (name, param), layer in zip(self.model.named_parameters(), proto_layers):
                layer_name = layer.layer_name
                if name == layer_name:
                    if self.communicate_params:
                        param.data = torch.tensor(np.array(layer.param_update).reshape(tuple(layer.param_shape)),
                                                  dtype=torch.float32,)
                        print(f"on client {self.client_id} param.data {param.data} with shape {param.data.shape}")

                    else:
                        param.grad = torch.tensor(np.array(layer.param_update).reshape(tuple(layer.param_shape)),
                                                  dtype=torch.float32,)
                        # print(f"on client {self.client_id} param.grad {param.grad} with shape {param.grad.shape}")


    def get_averaged_model(self):
        """Get averaged model from parameter server - wait indefinitely until ready"""
        print(f"Round {self.round_number}: Waiting for averaged model from server...")
        while True:
            try:
                request = flora_grpc_pb2.GetModelRequest(client_id=self.client_id, round_number=self.round_number)

                response = self.stub.GetUpdatedModel(request)

                if response.is_ready:
                    self._update_model_from_protobuf(response.layers)
                    print(
                        f"Round {self.round_number}: Received averaged model from server"
                    )
                    return True
                else:
                    # Model not ready yet, wait and try again
                    print(
                        f"Round {self.round_number}: Averaged model not ready, waiting 2 seconds..."
                    )
                    time.sleep(2)

            except grpc.RpcError as e:
                print(f"Failed to get averaged model (will retry): {e}")
                time.sleep(2)
                continue

    def federated_training_round(self):
        """Execute one round of federated training"""
        print(f"\n=== Round {self.round_number} ===")

        # Step 1: Train locally
        self.train_local_epoch()

        # Step 2: Send update to server
        if not self.send_update_to_server():
            return False

        # Step 3: Get averaged model from server
        if not self.get_averaged_model():
            return False

        self.round_number += 1
        return True

    def run_federated_training(self, num_rounds: int):
        """Run complete federated training process"""
        print(f"Starting federated training for {num_rounds} rounds...")

        for round_num in range(num_rounds):
            if not self.federated_training_round():
                print(f"Training failed at round {round_num}")
                break

            # Add some random delay to simulate real-world conditions
            time.sleep(random.uniform(0.5, 2.0))

            print(f"Federated training completed for client {self.client_id}")

    def close(self):
        """Close the connection to the server"""
        self.channel.close()


def start_client():
    if len(sys.argv) != 2:
        print("Usage: python scalable_client.py <client_id>")
        sys.exit(1)

    client_id = sys.argv[1]
    # client_id = 0

    # Create and run client
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch_samples = 32
    criterion = torch.nn.CrossEntropyLoss()
    client = CentralServerClient(
        client_id=client_id,
        batch_samples=batch_samples,
        criterion=criterion,
        communicate_params=True,
        compute_mean=True,
        model=model,
        optimizer=optimizer,
        server_address="localhost:50051",
    )
    try:
        # Wait a bit for other clients to connect
        print("Waiting for other clients to connect...")
        time.sleep(5)

        # Run federated training
        client.run_federated_training(num_rounds=1)

    except KeyboardInterrupt:
        print(f"\nClient {client_id} interrupted by user")
    finally:
        client.close()


if __name__ == "__main__":
    start_client()
