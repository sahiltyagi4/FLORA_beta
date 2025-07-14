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
from typing import Any

import rich.repr
import torch
from torch import nn

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedProxTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm


class FedProx:
    """
    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedProxTrainingParameters,
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
        self.mu = self.train_params.get_mu()
        self.local_step = 0
        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            proximal_term = 0.0
            for (name1, param1), (name2, param2) in zip(
                self.model.named_parameters(), self.global_model.named_parameters()
            ):
                proximal_term += ((param1 - param2) ** 2).sum()

            fedprox_loss = self.loss(pred, labels) + (self.mu * proximal_term) / 2

            fedprox_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            if self.local_step % self.comm_freq == 0:
                self.global_model = self.communicator.aggregate(
                    msg=self.model, compute_mean=True
                )
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


@rich.repr.auto
class FedProxNew(Algorithm):
    """
    FedProx algorithm implementation.

    FedProx extends FedAvg by adding a proximal term to the loss, which helps stabilize training in heterogeneous environments.
    The proximal term penalizes deviation from the global model during local updates.
    """

    def __init__(self, mu: float = 0.01, **kwargs):
        """Initialize FedProx algorithm with proximal term parameter."""
        super().__init__(**kwargs)
        self.mu = mu

    def _setup(self) -> None:
        """
        FedProx-specific setup: initialize global model for proximal term.
        """
        super()._setup()

        self.global_model = copy.deepcopy(self.local_model)

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _train_step(self, batch: Any, batch_idx: int) -> tuple[torch.Tensor, int]:
        """
        Forward pass and compute the FedProx loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        # Compute standard cross-entropy loss
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        # Add proximal term to penalize deviation from the global model
        prox_term = 0.0
        for (name, param), (_, global_param) in zip(
            self.local_model.named_parameters(), self.global_model.named_parameters()
        ):
            if param.requires_grad:
                prox_term += ((param - global_param).pow(2)).sum()
        loss += (self.mu / 2) * prox_term
        return loss, inputs.size(0)

    def _round_start(self, round_idx: int) -> None:
        """
        Update the reference global model at the start of each round.

        # TODO: check whether we can safely just move all this logic in round_start() for all algorithms to the end of aggregate() method and remove round_start() overrides altogether
        # TODO: should this logic be linked with the same granularity as aggregate(), rather than always on round_start?
        """
        # Update the reference global model (self.local_model already contains latest from aggregate())
        self.global_model.load_state_dict(self.local_model.state_dict())

    def _aggregate(self) -> None:
        """
        FedProx aggregation: weighted averaging of model parameters.

        NOTE: Compatible with all granularity levels.
        """
        # Aggregate local sample counts to compute federation total
        global_samples = self.comm.aggregate(
            torch.tensor([self.local_sample_count], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Handle edge cases safely - all nodes must participate in distributed operations
        if global_samples <= 0:
            data_proportion = 0.0
        else:
            # Calculate the proportion of data this client contributed
            data_proportion = self.local_sample_count / global_samples

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, data_proportion)

        # Aggregate weighted model parameters from all clients
        # NOTE: This aggregate() call returns the updated global model, so the local_model is now the aggregated global model
        self.local_model = self.comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )
