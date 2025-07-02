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

import os

import torch
import torch.distributed.rpc as rpc

from src.flora.communicator import Communicator


class RpcServer(object):
    def __init__(self, model, total_clients=1):
        """
        :param model: model to communicate
        :param total_clients: total number of clients/ world-size (including the server)
        """
        self.model_update = [torch.zeros_like(param) for param in model.parameters()]
        self.aggregated_update = None
        self.total_clients = total_clients
        self.client_count = 0
        self.samples = 0
        self.collected_samples = 0
        self.sample_count = 0
        self.collect_updates = {}
        self.collect_count = 0

    def aggregate_updates(self, updates, compute_mean=True):
        for server_update, model_update in zip(self.model_update, updates):
            server_update.add_(model_update)

        self.client_count += 1
        if self.client_count == self.total_clients:
            self.client_count = 0
            self.model_update /= self.total_clients if compute_mean else 1

            self.aggregated_update = self.model_update
            self.model_update = [torch.zeros_like(param) for param in self.model_update]
            return self.aggregated_update

    def collect_updates(self, msg, client_id, communicate_params=True):
        if isinstance(msg, torch.nn.Module):
            for name, param in msg.named_parameters():
                if not param.requires_grad:
                    continue

                if name in self.collect_updates.keys():
                    update_collection = self.collect_updates[name]
                else:
                    update_collection = []

                # unlike torch_mpi, RPC keeps a tuple of (client_id, update) to know which client an update came from
                if communicate_params:
                    update_collection.append((client_id, param.data))
                else:
                    update_collection.append((client_id, param.grad))

                self.collect_updates[name] = update_collection

            self.collect_count += 1
            if self.collect_count == self.total_clients:
                self.collect_count = 0
                collected_data = list(self.collect_updates.values())
                self.collect_updates = {}
                return collected_data

    def server_model(self, id):
        print(f"fetching model update to server-id {id}")
        if self.aggregated_update is None:
            return self.model_update
        else:
            return self.aggregated_update

    def aggregate_metric(self, samples, compute_mean=True):
        self.samples += samples
        self.sample_count += 1
        if self.sample_count == self.total_clients:
            self.sample_count = 0
            self.samples /= self.total_clients if compute_mean else 1
            self.collected_samples = self.samples
            self.samples = 0
            return self.collected_samples


class TorchRpcCommunicator(Communicator):
    def __init__(
        self, id=0, total_clients=1, master_addr="127.0.0.1", master_port=27890
    ):
        super().__init__(protocol_type="torch_rpc")
        self.id = id
        self.total_clients = total_clients
        self.master_addr = master_addr
        self.master_port = master_port
        self.central_server = "central_server"
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

        # TODO: adjust based on total available threads
        opts = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4, rpc_timeout=0)
        if self.id == 0:
            rpc.init_rpc(
                self.central_server,
                rank=self.id,
                world_size=self.total_clients,
                rpc_backend_options=opts,
            )

        else:
            rpc.init_rpc(
                f"worker-{self.id}",
                rank=self.id,
                world_size=self.total_clients,
                rpc_backend_options=opts,
            )

    def aggregate(self, msg, communicate_params=True, compute_mean=True):
        if isinstance(msg, torch.nn.Module):
            # communicate either model parameters or gradients
            if communicate_params:
                updates = [param.data.detach() for param in msg.parameters()]
            else:
                updates = [param.grad.detach() for param in msg.parameters()]

            aggregated_update = rpc.rpc_sync(
                self.central_server,
                RpcServer.aggregate_updates,
                args=(updates, compute_mean),
            )
            for param, update in zip(msg.parameters(), aggregated_update):
                if communicate_params:
                    param.data.copy_(update)
                else:
                    param.grad.data.copy_(update)

            return msg
        else:
            aggregated_samples = rpc.rpc_sync(
                self.central_server,
                RpcServer.aggregate_metric,
                args=(msg, compute_mean),
            )
            return aggregated_samples

    def broadcast(self, msg, id=0):
        model_update = rpc.rpc_sync(
            self.central_server, RpcServer.server_model, args=(id,)
        )
        if isinstance(msg, torch.nn.Module):
            for param, update in zip(msg.parameters(), model_update):
                param.data.copy_(update)

            return msg

    def collect(self, msg, id, communicate_params=True):
        collected_updates = rpc.rpc_sync(
            self.central_server,
            RpcServer.collect_updates,
            args=(msg, id, communicate_params),
        )
        return collected_updates
