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

from typing import Any, Dict

import rich.repr
import torch
from torch import nn

from ..communicator import AggregationOp, BaseCommunicator
from . import utils
from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedDyn(BaseAlgorithm):
    """
    Federated Dynamic Regularization (FedDyn) algorithm implementation.

    FedDyn introduces a dynamic regularization term to address objective inconsistency in federated learning and improve convergence in heterogeneous environments.

    [FedDyn](https://arxiv.org/abs/2111.04263) | Durmus Alp Emre Acar | 2021-11-08
    """

    def __init__(self, alpha: float = 0.1, **kwargs):
        """Initialize FedDyn algorithm with regularization parameter and granularity validation."""
        super().__init__(**kwargs)
        self.alpha = alpha

    def _setup(self, *args, **kwargs) -> None:
        """
        FedDyn-specific setup: initialize server momentum.
        """
        super()._setup(*args, **kwargs)

        # Initialize server momentum for dynamic regularization
        # all zero-initialized tensors based on param.data have requires_grad=False by default
        self.server_momentum: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                self.server_momentum[name] = torch.zeros_like(param.data)

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Perform a forward pass and compute the FedDyn loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)

        # Standard cross-entropy loss
        base_loss = torch.nn.functional.cross_entropy(outputs, targets)

        # Add FedDyn dynamic regularization if state is available
        regularization_loss = torch.tensor(0.0, device=inputs.device)
        # Quadratic term optimized using generator expression and tensor concatenation
        param_norm_squared = torch.sum(
            torch.stack(
                [
                    torch.sum(param**2)
                    for param in self.local_model.parameters()
                    if param.requires_grad
                ]
            )
        )
        quadratic_term = (self.alpha / 2.0) * param_norm_squared

        # Linear term
        linear_term = 0.0
        for param_name, local_param in self.local_model.named_parameters():
            if local_param.requires_grad and param_name in self.server_momentum:
                linear_term -= torch.sum(self.server_momentum[param_name] * local_param)

        regularization_loss = quadratic_term + linear_term

        total_loss = base_loss + regularization_loss
        return total_loss

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        FedDyn aggregation: weighted averaging with dynamic regularization momentum.
        """

        # Store local model before aggregation for momentum update
        local_model_params = {}
        for param_name, local_param in self.local_model.named_parameters():
            if local_param.requires_grad:
                local_model_params[param_name] = local_param.data.clone()

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, weight)

        # Aggregate weighted model parameters
        aggregated_model = comm.aggregate(
            self.local_model,
            reduction=AggregationOp.SUM,
        )

        # Update server momentum (dynamic regularizer)
        with torch.no_grad():
            for param_name, aggregated_param in aggregated_model.named_parameters():
                if (
                    aggregated_param.requires_grad
                    and param_name in self.server_momentum
                ):
                    # Compute model difference: local - global
                    model_diff = local_model_params[param_name] - aggregated_param.data

                    # Server momentum update accumulates local deviations from global model
                    # captures the "drift" direction and is used in regularization term during training
                    self.server_momentum[param_name].add_(model_diff, alpha=self.alpha)

        # Return aggregated result
        return aggregated_model
