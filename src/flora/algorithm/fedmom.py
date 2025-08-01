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

from ..communicator import AggregationOp, BaseCommunicator
from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedMom(BaseAlgorithm):
    """
    Federated Momentum (FedMom) algorithm implementation.

    FedMom applies momentum to the aggregation of model updates, improving convergence and stability in federated learning.

    FedMom](https://arxiv.org/abs/2002.02090) | Zhouyuan Huo | 2020-02-06
    """

    def __init__(self, momentum: float = 0.9, **kwargs):
        """Initialize FedMom algorithm with momentum parameter and granularity validation."""
        super().__init__(**kwargs)
        self.momentum = momentum

    def _setup(self, *args, **kwargs) -> None:
        """
        FedMom-specific setup: initialize global model and velocity buffers.
        """
        super()._setup(*args, **kwargs)
        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Global model is reference-only, disable gradients and set eval mode
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
        Configure the SGD optimizer for local model updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

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
        FedMom aggregation: server-side momentum on aggregated parameter deltas.
        """
        # Compute local parameter delta from global model
        local_deltas: Dict[str, torch.Tensor] = {}
        # Pre-compute global parameters dictionary once to avoid O(n^2) complexity
        global_params = dict(self.global_model.named_parameters())
        for param_name, local_param in self.local_model.named_parameters():
            if local_param.requires_grad:
                global_param = global_params[param_name]  # O(1) lookup
                # what the client actually learned this round
                local_deltas[param_name] = local_param.data - global_param.data

        # Scale local deltas by data proportion
        for param_name in local_deltas:
            local_deltas[param_name].mul_(weight)

        # Aggregate scaled deltas
        aggregated_deltas = comm.aggregate(
            local_deltas,
            reduction=AggregationOp.SUM,
        )

        # Apply server-side momentum to aggregated deltas
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if param.requires_grad and name in self.velocity:
                    # Update velocity with momentum
                    self.velocity[name].mul_(self.momentum).add_(
                        aggregated_deltas[name]
                    )

                    # Apply server-side momentum to update global model parameters
                    # ADD because if deltas represent "what clients learned", we add them to global model
                    param.data.add_(self.velocity[name], alpha=self.local_lr)

        # Return updated global model as the new local model for next training period
        return copy.deepcopy(self.global_model)
