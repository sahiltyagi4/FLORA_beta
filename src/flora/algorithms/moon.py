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
from typing import Any, Dict, List, Optional, Tuple, Union

import rich.repr
import torch
import torch.nn as nn

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import MOONTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm


class MoonWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.proj_head = torch.nn.Identity()

    def forward(self, input):
        features = self.base_model.features(input)
        logits = self.base_model.classifier(features)
        representation = self.proj_head(features)

        return logits, representation


class Moon:
    """Implementation of Model-Contrastive Federated Learning or MOON

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: MOONTrainingParameters,
    ):
        """
        :param model: model to train
        :param data: data to train
        :param communicator: communicator object
        :param total_clients: total number of clients / world size
        :param train_params: training hyperparameters
        """
        self.model = MoonWrapper(model)
        self.train_data = train_data
        self.communicator = communicator
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.epochs = self.train_params.get_epochs()
        self.num_prev_models = self.train_params.get_num_prev_models()
        self.temperature = self.train_params.get_temperature()
        self.mu = self.train_params.get_mu()
        # history of previous models tracked for contrastive loss calculation
        self.prev_models = []
        self.local_step = 0
        self.training_samples = 0
        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)
        self.negative_reprs = []

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # local model prediction and representation
            pred, local_repr = self.model(inputs)
            self.training_samples += inputs.size(0)
            with torch.no_grad():
                _, global_repr = self.global_model(inputs)
                if len(self.prev_models) > 0:
                    self.negative_reprs = [
                        prev_model(inputs)[1] for prev_model in self.prev_models
                    ]

            loss = self.loss(pred, labels)
            if len(self.negative_reprs) > 0:
                local_repr = torch.nn.functional.normalize(local_repr, dim=1)
                global_repr = torch.nn.functional.normalize(global_repr, dim=1)
                self.negative_reprs = [
                    torch.nn.functional.normalize(repr, dim=1)
                    for repr in self.negative_reprs
                ]
                pos_sim = torch.exp(
                    torch.sum(local_repr * global_repr, dim=1) / self.temperature
                )
                neg_sim = torch.stack(
                    [
                        torch.exp(torch.sum(local_repr * neg, dim=1) / self.temperature)
                        for neg in self.negative_reprs
                    ],
                    dim=1,
                ).sum(dim=1)

                contrastive_loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                loss += self.mu * contrastive_loss.mean()

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
                    msg=self.model, compute_mean=False
                )
                self.model.load_state_dict(self.global_model.state_dict())
                self.training_samples = 0

                model_copy = copy.deepcopy(self.global_model)
                model_copy.eval()
                if len(self.prev_models) == self.num_prev_models:
                    self.prev_models.pop()
                self.prev_models.insert(0, model_copy)

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
class MOONWrapper(nn.Module):
    """
    Model wrapper for Model-Contrastive Federated Learning (MOON).

    Provides both logits and feature representations for use in contrastive loss calculations.
    Assumes the base model has .features and .classifier attributes.
    """

    def __init__(self, base_model):
        """
        Initialize the MOONWrapper model.
        """
        super().__init__()
        self.base_model = base_model
        self.proj_head = nn.Identity()

    def forward(self, input):
        """
        Forward pass through the base model, returning logits and feature representations.
        """
        # TODO: Current implementation assumes specific model architecture which may not be universal. May want to transition to using our ComposableModel class.
        features = self.base_model.features(input)
        logits = self.base_model.classifier(features)
        representation = self.proj_head(features)
        return logits, representation


class MOONNew(Algorithm):
    """
    Model-Contrastive Federated Learning (MOON) algorithm implementation.

    MOON uses model-level contrastive learning to align local models
    with the global model and distinguish them from previous local models,
    improving convergence and generalization in federated learning.
    """

    def __init__(
        self,
        local_model: nn.Module,
        comm: Communicator,
        max_epochs: int,
        lr: float = 0.01,
        mu: float = 1.0,
        temperature: float = 0.5,
        num_prev_models: int = 1,
    ):
        super().__init__(local_model, comm, max_epochs)
        self.lr = lr
        self.mu = mu
        self.temperature = temperature
        self.num_prev_models = num_prev_models

        # ---
        if not isinstance(local_model, MOONWrapper):
            wrapped_initial = MOONWrapper(local_model)
        else:
            wrapped_initial = local_model

        self.global_model = copy.deepcopy(wrapped_initial)
        self.global_model.eval()
        self.prev_models = []

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=self.lr)

    def train_step(self, batch: Any, batch_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Forward pass and compute the MOON loss for a single batch, including contrastive loss.
        """
        inputs, targets = batch

        # Get predictions and representations from wrapped model
        pred, local_repr = self.local_model(inputs)

        # Standard cross-entropy loss
        base_loss = torch.nn.functional.cross_entropy(pred, targets)

        # Compute contrastive loss if we have global model and previous models
        # MOON computes contrastive loss per batch to align local representations with global model while contrasting with previous models
        contrastive_loss = torch.tensor(0.0, device=inputs.device)
        with torch.no_grad():
            # Get global model representation (positive sample)
            _, global_repr = self.global_model(inputs)

            # Get negative representations from previous models
            negative_reprs = []
            if len(self.prev_models) > 0:
                negative_reprs = [
                    prev_model(inputs)[1] for prev_model in self.prev_models
                ]

        # Compute contrastive loss if we have negative samples
        if len(negative_reprs) > 0:
            # Normalize representations
            local_repr = torch.nn.functional.normalize(local_repr, dim=1)
            global_repr = torch.nn.functional.normalize(global_repr, dim=1)
            negative_reprs = [
                torch.nn.functional.normalize(repr, dim=1) for repr in negative_reprs
            ]

            # Compute similarities with temperature scaling
            pos_sim = torch.exp(
                torch.sum(local_repr * global_repr, dim=1) / self.temperature
            )
            neg_sim = torch.stack(
                [
                    torch.exp(torch.sum(local_repr * neg, dim=1) / self.temperature)
                    for neg in negative_reprs
                ],
                dim=1,
            ).sum(dim=1)

            # Contrastive loss
            contrastive_loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
            contrastive_loss = contrastive_loss.mean()

        # Combined loss
        total_loss = base_loss + self.mu * contrastive_loss

        return total_loss, inputs.size(0)

    def round_start(self, round_idx: int) -> None:
        """
        Synchronize the local model with the global model at the start of each round.
        """
        # Receive global model via broadcast from rank 0 (server)
        self.local_model = self.comm.broadcast(
            self.local_model,
            src=0,
        )

        # Update global model reference
        self.global_model.load_state_dict(self.local_model.state_dict())
        self.global_model.eval()

    def round_end(self, round_idx: int) -> None:
        """
        Aggregate model parameters across clients and update the local model, maintaining a history of previous models for contrastive learning.
        """
        # Aggregate local sample counts to compute federation total

        global_samples = self.comm.aggregate(
            torch.tensor([self.local_samples], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Handle edge cases safely - all nodes must participate in distributed operations
        if global_samples <= 0:
            data_proportion = 0.0
        else:
            # Calculate data proportion for weighted aggregation
            data_proportion = self.local_samples / global_samples

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, data_proportion)

        # Aggregate scaled models
        self.local_model = self.comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )

        # Update previous model history
        # Create a copy of the aggregated model for history
        model_copy = copy.deepcopy(self.local_model)
        model_copy.eval()

        # Maintain history of previous models
        if len(self.prev_models) == self.num_prev_models:
            self.prev_models.pop()  # Remove oldest
        self.prev_models.insert(0, model_copy)  # Add newest at front
