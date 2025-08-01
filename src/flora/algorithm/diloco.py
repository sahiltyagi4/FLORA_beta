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
from typing import Any, Dict

import rich.repr
import torch
from torch import nn

from ..communicator import AggregationOp, BaseCommunicator
from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class DiLoCo(BaseAlgorithm):
    """
    Implementation of DiLoCo (Distributed Low-Communication).

    DiLoCo combines local SGD with server-side momentum updates to
    reduce communication frequency while maintaining convergence properties.

    [DiLoCo](https://arxiv.org/abs/2311.08105) | Arthur Douillard | 2023-11-14
    """

    def __init__(
        self,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        **kwargs,
    ):
        """Initialize DiLoCo algorithm with distributed optimizer parameters."""
        super().__init__(**kwargs)
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum

    def _setup(self, *args, **kwargs) -> None:
        """
        DiLoCo-specific setup: initialize global model and velocity.
        """
        super()._setup(*args, **kwargs)
        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Global model is reference-only for delta computation, set eval mode and disable gradients
        self.global_model.eval()  # eval() does NOT turn off gradient tracking.
        for param in self.global_model.parameters():
            param.requires_grad = False

        # Initialize velocity (server-side momentum)
        # all zero-initialized tensors based on param.data have requires_grad=False by default
        self.velocity: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                self.velocity[name] = torch.zeros_like(param.data)

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        Configure SGD optimizer for DiLoCo local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Forward pass and compute cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        DiLoCo aggregation: distributed low-communication with server-side momentum.
        """
        # Compute local model update (delta from global model)
        # Pre-compute global parameters dictionary once to avoid O(n^2) complexity
        global_params = dict(self.global_model.named_parameters())
        local_deltas: Dict[str, torch.Tensor] = {}
        for param_name, local_param in self.local_model.named_parameters():
            if local_param.requires_grad and param_name in self.velocity:
                global_param = global_params[param_name]  # O(1) lookup
                local_deltas[param_name] = local_param.data - global_param.data

        # DiLoCo uses mean aggregation rather than weighted aggregation
        aggregated_deltas = comm.aggregate(
            msg=local_deltas,
            reduction=AggregationOp.MEAN,
        )

        # Apply DiLoCo outer step with momentum using aggregated deltas
        # Use the same pre-computed global parameters dictionary for efficiency
        with torch.no_grad():
            for param_name, delta in aggregated_deltas.items():
                # All parameters in aggregated_deltas already passed velocity and requires_grad filters
                # Update velocity with momentum (v = momentum * v + lr_outer * delta)
                self.velocity[param_name].mul_(self.outer_momentum).add_(
                    delta, alpha=self.outer_lr
                )
                # Update global model parameters (param += v)
                global_params[param_name].data.add_(self.velocity[param_name])

        # Return updated global model as the new local model for next training period
        return copy.deepcopy(self.global_model)
