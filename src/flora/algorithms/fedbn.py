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

import rich.repr
import torch
import torch.nn as nn

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedBNTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm


class FederatedBatchNormalization:
    """
    Implementation of Federated Averaging with Batch Normalization or FedBN.
    Similar to Federated Averaging, but instead of aggregating all weights, clients keep their batch normalization
    layers local and aggregate other parameters.

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedBNTrainingParameters,
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

                # save batch normalization layer parameters
                bn_layers = {}
                for name, param in self.model.named_parameters():
                    if "bn" in name or "norm" in name:
                        bn_layers[name] = param.data

                    # scale client updates based on number of samples processed
                    param.data *= weight_scaling

                # average model parameters across clients
                self.model = self.communicator.aggregate(
                    msg=self.model, compute_mean=False
                )

                # revert back to client-local values of batch normalization layers
                for name, param in self.model.named_parameters():
                    if "bn" in name or "norm" in name:
                        param.data.copy_(bn_layers[name])

                bn_layers = None
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
class FedBNNew(Algorithm):
    """
    Federated Batch Normalization (FedBN) algorithm implementation.

    FedBN aggregates only non-batch normalization parameters across clients,
    allowing each client to maintain its own batch normalization statistics for improved personalization.
    """

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _train_step(self, batch: Any, batch_idx: int) -> tuple[torch.Tensor, int]:
        """
        Perform a forward pass and compute the loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss, inputs.size(0)

    def _aggregate(self) -> None:
        """
        FedBN aggregation: aggregate non-BatchNorm parameters while keeping BN layers local.

        NOTE: Compatible with all granularity levels.
        - Only non-BatchNorm parameters are aggregated while BatchNorm statistics remain local for domain adaptation.
        """
        # Save local BN parameters before aggregation
        local_bn_params = {}
        for name, param in self.local_model.named_parameters():
            if self._is_bn_layer(name):
                local_bn_params[name] = param.data.clone()

        # Aggregate local sample counts to compute federation total
        global_samples = self.comm.aggregate(
            torch.tensor([self.local_sample_count], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Handle edge cases safely - all nodes must participate in distributed operations
        if global_samples <= 0:
            data_proportion = 0.0
        else:
            # Calculate data proportion for weighted aggregation
            data_proportion = self.local_sample_count / global_samples

        # Scale only non-BN parameters by data proportion (all nodes participate)
        for name, param in self.local_model.named_parameters():
            if not self._is_bn_layer(name):
                param.data.mul_(data_proportion)

        # Aggregate non-BN parameters
        self.local_model = self.comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )

        # Restore local BN parameters
        for name, param in self.local_model.named_parameters():
            if self._is_bn_layer(name) and name in local_bn_params:
                param.data.copy_(local_bn_params[name])

    def _is_bn_layer(self, param_name: str) -> bool:
        """
        Check if parameter belongs to a batch normalization layer.
        """
        # TODO: More robust BN detection instead of string matching
        return "bn" in param_name or "norm" in param_name
