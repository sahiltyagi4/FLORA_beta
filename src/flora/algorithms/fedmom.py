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

import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.flora.communicator import Communicator
from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedMomTrainingParameters

from . import utils
from .BaseAlgorithm import Algorithm


class FederatedMomentum:
    """
    Implementation of Federated Momentum or FedMom

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedMomTrainingParameters,
    ):
        """
        :param model: model to train
        :param data: data to train
        :param communicator: communicator object
        :param total_clients: total number of clients / world size
        :param train_params: training hyperparameters
        """
        self.model = model
        self.train_data = train_data
        self.communicator = communicator
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.epochs = self.train_params.get_epochs()
        self.lr = self.train_params.get_lr()
        self.momentum = self.train_params.get_momentum()
        self.local_step = 0
        self.training_samples = 0
        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.diff_params = copy.deepcopy(self.model)
        self.global_model, self.diff_params = (
            self.global_model.to(self.device),
            self.diff_params.to(self.device),
        )
        self.velocity = {
            name: torch.zeros_like(param.data).to(self.device)
            for name, param in self.model.named_parameters()
        }

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def _outer_step(self):
        total_samples = self.communicator.aggregate(
            msg=torch.Tensor([self.training_samples]), compute_mean=False
        )
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.global_model.named_parameters(), self.model.named_parameters()
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                target_param = dict(self.diff_params.named_parameters())[name1]
                # scaling updates based on number of samples processed by each client
                target_param.copy_(
                    ((param1 - param2) * self.training_samples) / total_samples.item()
                )

        self.diff_params = self.communicator.aggregate(
            msg=self.diff_params, communicate_params=True, compute_mean=False
        )
        with torch.no_grad():
            for (name, param), (_, param_delta) in zip(
                self.global_model.named_parameters(),
                self.diff_params.named_parameters(),
            ):
                self.velocity[name] = self.momentum * self.velocity[name] + param_delta
                param.data -= self.lr * self.velocity[name]

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            self.training_samples += inputs.size(0)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            if self.local_step % self.comm_freq == 0:
                self._outer_step()
                self.model.load_state_dict(self.global_model.state_dict())

    def train(self):
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()


# ======================================================================================


class FedMomNew(Algorithm):
    """
    Federated Momentum (FedMom) algorithm implementation.

    FedMom applies momentum to the aggregation of model updates, improving convergence and stability in federated learning.
    """

    def __init__(
        self,
        local_model: nn.Module,
        comm: Communicator,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        super().__init__(local_model, comm)
        self.lr = lr
        self.momentum = momentum

        # ---
        self.global_model = copy.deepcopy(local_model)

        # Initialize velocity (server-side momentum) to zero
        self.velocity: Dict[str, torch.Tensor] = {}
        for name, param in local_model.named_parameters():
            if param.requires_grad:
                self.velocity[name] = torch.zeros_like(param.data)

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure the SGD optimizer for local model updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=self.lr)

    def train_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Perform a forward pass and compute the loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss, inputs.size(0)

    def round_start(self, round_idx: int) -> None:
        """
        Synchronize the local model with the global model at the start of each round.
        """
        # Receive global model via broadcast from rank 0 (server)
        self.local_model = self.comm.broadcast(self.local_model, src=0)
        # Update global model reference
        self.global_model.load_state_dict(self.local_model.state_dict())

    def round_end(self, round_idx: int) -> None:
        """
        Aggregate model parameters across clients using momentum and update the local model.
        """
        # Compute local parameter delta from global model
        local_deltas: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                global_param = dict(self.global_model.named_parameters())[name]
                # Delta = global - local (original uses global - local for momentum direction)
                local_deltas[name] = global_param.data - param.data

        # Aggregate local sample counts to compute federation total

        global_samples = self.comm.aggregate(
            torch.tensor([self.local_samples], dtype=torch.float32),
            communicate_params=False,
            compute_mean=False,
        ).item()

        # Handle edge cases safely - all nodes must participate in distributed operations
        if global_samples <= 0:
            print(
                "WARN: No samples processed across entire federation - participating with zero weight"
            )
            data_proportion = 0.0
        else:
            # Calculate data proportion for weighted aggregation of deltas
            data_proportion = self.local_samples / global_samples

        # Scale local deltas by data proportion
        for name in local_deltas:
            local_deltas[name].mul_(data_proportion)

        # Apply server-side momentum to aggregated deltas
        for name, param in self.global_model.named_parameters():
            if param.requires_grad and name in self.velocity:
                # Update velocity with momentum using in-place operations
                self.velocity[name].mul_(self.momentum).add_(local_deltas[name])

                # Update global model parameters using alpha argument
                param.data.sub_(self.velocity[name], alpha=self.lr)

        # Update local model to match updated global model
        self.local_model = copy.deepcopy(self.global_model)
