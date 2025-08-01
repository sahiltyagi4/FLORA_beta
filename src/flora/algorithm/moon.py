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
import torch.nn as nn

from ..communicator import AggregationOp, BaseCommunicator
from . import utils
from .base import BaseAlgorithm

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


class MOON(BaseAlgorithm):
    """
    Model-Contrastive Federated Learning (MOON) algorithm implementation.

    MOON uses model-level contrastive learning to align local models
    with the global model and distinguish them from previous local models,
    improving convergence and generalization in federated learning.

    [MOON](https://arxiv.org/abs/2103.16257) | Qinbin Li | 2021-03-30
    """

    def __init__(
        self,
        mu: float = 1.0,
        temperature: float = 0.5,
        num_prev_models: int = 1,
        **kwargs,
    ):
        """Initialize MOON algorithm with contrastive learning parameters."""
        super().__init__(**kwargs)
        self.mu = mu
        self.temperature = temperature
        self.num_prev_models = num_prev_models

    def _setup(self, *args, **kwargs) -> None:
        """
        MOON-specific setup: wrap model and initialize global model and previous models.
        """
        if not isinstance(self.local_model, MOONWrapper):
            wrapped_initial = MOONWrapper(self.local_model)
            self.local_model = wrapped_initial

        super()._setup(*args, **kwargs)

        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Global model is reference-only for contrastive learning, disable gradients and set eval mode
        self.global_model.eval()  # eval() does NOT turn off gradient tracking.
        for param in self.global_model.parameters():
            param.requires_grad = False

        # Initialize previous models history for contrastive learning
        self.prev_models = []

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
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

            # Get representations from previous models for contrastive learning
            negative_reprs = []
            if len(self.prev_models) > 0:
                negative_reprs = [
                    prev_model(inputs)[1] for prev_model in self.prev_models
                ]

        # Compute contrastive loss if we have negative samples
        if len(negative_reprs) > 0:
            # batch normalize all representations together for efficiency
            local_repr = torch.nn.functional.normalize(local_repr, dim=1)
            global_repr = torch.nn.functional.normalize(global_repr, dim=1)
            # Stack and batch normalize negative representations for better performance
            if negative_reprs:
                negative_reprs = torch.nn.functional.normalize(
                    torch.stack(negative_reprs, dim=0), dim=2
                )

            # Compute similarities with temperature scaling
            pos_sim = torch.exp(
                torch.sum(local_repr * global_repr, dim=1) / self.temperature
            )
            # negative_reprs is now [num_prev_models, batch_size, repr_dim]
            # local_repr is [batch_size, repr_dim]
            neg_sim = torch.exp(
                torch.sum(local_repr.unsqueeze(0) * negative_reprs, dim=2)
                / self.temperature
            ).sum(dim=0)  # Sum over previous models, keep batch dimension

            # Contrastive loss
            contrastive_loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
            contrastive_loss = contrastive_loss.mean()

        # Combined loss
        total_loss = base_loss + self.mu * contrastive_loss

        return total_loss

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        MOON aggregation: weighted averaging with model history for contrastive learning.
        """
        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, weight)

        # Aggregate scaled models
        aggregated_model = comm.aggregate(
            self.local_model,
            reduction=AggregationOp.SUM,
        )

        # Update previous model history
        # Create a copy of the aggregated model for history
        model_copy = copy.deepcopy(aggregated_model)
        model_copy.eval()
        for param in model_copy.parameters():
            param.requires_grad = False

        # Maintain history of previous models
        if len(self.prev_models) >= self.num_prev_models:
            self.prev_models.pop()  # Remove oldest
        self.prev_models.insert(0, model_copy)  # Add newest at front

        return aggregated_model
