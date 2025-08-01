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

from typing import Any, Optional

import rich.repr
import torch
import torch.nn as nn

from ..communicator import AggregationOp, BaseCommunicator
from . import utils
from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedPer(BaseAlgorithm):
    """
    Federated Personalization (FedPer) algorithm implementation.

    FedPer splits the model into a shared base model and a personal head.
    Only the base model is aggregated across clients;
    each client maintains its own personal head for local adaptation.

    [FedPer](https://arxiv.org/abs/1912.00818) | Muhammad Ammad-ud-din | 2020-01-01
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

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Perform a forward pass and compute the loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        FedPer aggregation: aggregate base model while preserving personal layers.
        """
        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, weight)

        # Store personal layer parameters before aggregation
        with torch.no_grad():
            personal_params = {
                name: param.data.clone()
                for name, param in self.local_model.named_parameters()
                if self._is_personal_layer(name)
            }

        # Aggregate entire model (including personal layers)
        aggregated_model = comm.aggregate(
            self.local_model,
            reduction=AggregationOp.SUM,
        )

        # Restore personal layer parameters (keep them local)
        with torch.no_grad():
            for param_name, aggregated_param in aggregated_model.named_parameters():
                if param_name in personal_params:
                    aggregated_param.data.copy_(personal_params[param_name])

        return aggregated_model
