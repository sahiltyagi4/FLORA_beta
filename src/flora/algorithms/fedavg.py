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

from typing import Any, Tuple

import rich.repr
import torch
from torch import nn

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedAvgTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm


class FederatedAveraging:
    """
    Implementation of Federated Averaging

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedAvgTrainingParameters,
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
        self.local_step = 0
        self.training_samples = 0

        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

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
                # total samples processed across all clients
                total_samples = self.communicator.aggregate(
                    msg=torch.Tensor([self.training_samples]), compute_mean=False
                )
                weight_scaling = self.training_samples / total_samples.item()
                for _, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    param.data *= weight_scaling

                self.model = self.communicator.aggregate(
                    msg=self.model, compute_mean=False
                )
                self.training_samples = 0

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
class FedAvgNew(Algorithm):
    """
    Federated Averaging (FedAvg) algorithm implementation.

    FedAvg performs standard federated learning by averaging model parameters across clients after local training rounds.
    Only model parameters are aggregated; all clients synchronize with the global model at the start of each round.
    """

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _train_step(self, batch: Any) -> Tuple[torch.Tensor, int]:
        """
        Forward pass and compute the cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        batch_size = inputs.size(0)
        return loss, batch_size

    def _aggregate(self) -> None:
        """
        Aggregate model parameters across clients and update the local model with the weighted average.
        """

        # Aggregate local sample counts to compute federation total
        global_samples = self.local_comm.aggregate(
            torch.tensor([self.local_sample_count], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Handle edge cases safely - all nodes must participate in distributed operations
        if global_samples <= 0:
            data_proportion = 0.0
        else:
            # Calculate this client's data proportion for weighted aggregation
            data_proportion = self.local_sample_count / global_samples

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, data_proportion)

        # Aggregate models across all clients
        # NOTE: This aggregate() call returns the updated global model, so the local_model is now the aggregated global model
        self.local_model = self.local_comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )
        # TODO: should we require all aggregate functions to return the updated global model nn.Module? or would that be incompatible with some algorithms? that might improve clarity and consistency of the API and data flow.
