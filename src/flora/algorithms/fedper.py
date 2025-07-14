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
from typing import Any, Dict, Optional, Tuple

import rich.repr
import torch
import torch.nn as nn

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedPerTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm

# Example of personal_head used by FedPerModel

# class SimplePersonalModel(torch.nn.Module):
#     def __init__(self, input_dim=5408, num_classes=10):
#         super().__init__()
#         self.classifier = torch.nn.Linear(input_dim, num_classes)
#
#     def forward(self, x):
#         return self.classifier(x)


class FedPerWrapper(nn.Module):
    """
    Model wrapper for FedPer.

    Combines a shared base model and a personal head. Only the base model is aggregated across clients; the personal head remains local for client-specific adaptation.
    """

    def __init__(self, base_model: nn.Module, personal_head: nn.Module):
        super(FedPerWrapper, self).__init__()
        self.base_model = base_model
        self.personal_head = personal_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model(x)
        return self.personal_head(x)


class FedPer:
    """
    implementation of FedPer

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        personal_head: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedPerTrainingParameters,
    ):
        """
        :param model: model to train
        :param data: data to train
        :param communicator: communicator object
        :param total_clients: total number of clients / world size
        :param train_params: training hyperparameters
        """
        self.model = FedPerWrapper(base_model, personal_head)
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

                self.global_model = self.communicator.aggregate(
                    msg=self.model.base_model,
                    compute_mean=False,
                )
                self.model.base_model.load_state_dict(
                    self.global_model.base_model.state_dict()
                )
                self.training_samples = 0

    def train(self):
        self.model.train()
        self.global_model.eval()
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()


# ======================================================================================


@rich.repr.auto
class FedPerNew(Algorithm):
    """
    Federated Personalization (FedPer) algorithm implementation.

    FedPer splits the model into a shared base model and a personal head.
    Only the base model is aggregated across clients;
    each client maintains its own personal head for local adaptation.
    """

    def __init__(self, personal_layers: Optional[list[str]] = None, **kwargs):
        """Initialize FedPer algorithm with personalization layer configuration."""
        super().__init__(**kwargs)
        self.personal_layers = personal_layers or ["classifier", "head", "fc"]

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for both base and personal parameters.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _is_personal_layer(self, param_name: str) -> bool:
        """
        Check if a parameter belongs to a personal layer that should not be aggregated.
        """
        return any(layer_name in param_name for layer_name in self.personal_layers)

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
        FedPer aggregation: aggregate base model while preserving personal layers.

        NOTE: Compatible with all granularity levels.
        - Only non-personal parameters are aggregated while personal layers remain local for personalization.
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
            # Calculate data proportion for weighted aggregation
            data_proportion = self.local_sample_count / global_samples

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, data_proportion)

        # Store personal layer parameters before aggregation
        personal_params = {}
        for name, param in self.local_model.named_parameters():
            if self._is_personal_layer(name):
                personal_params[name] = param.data.clone()

        # Aggregate entire model (including personal layers)
        self.local_model = self.comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )

        # Restore personal layer parameters (keep them local)
        for name, param in self.local_model.named_parameters():
            if name in personal_params:
                param.data.copy_(personal_params[name])
