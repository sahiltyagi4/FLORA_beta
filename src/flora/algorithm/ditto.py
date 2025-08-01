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
class Ditto(BaseAlgorithm):
    """
    Ditto algorithm implementation for personalized federated learning.

    Ditto maintains both a global model (shared across clients) and a local personal model.
    The personal model is trained with a proximal regularization term to stay close to the global model.

    [Ditto](https://arxiv.org/abs/2012.04221) | Tian Li | 2020-12-08
    """

    def __init__(self, global_lr: float = 0.01, ditto_lambda: float = 0.1, **kwargs):
        """Initialize Ditto algorithm with global learning rate and regularization parameter."""
        super().__init__(**kwargs)
        self.global_lr = global_lr
        self.ditto_lambda = ditto_lambda

    def _setup(self, *args, **kwargs) -> None:
        """
        Ditto-specific setup: initialize global model and optimizer.
        """
        super()._setup(*args, **kwargs)

        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Ditto trains the global model actively (forward/backward/optimizer.step)

        self.global_optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=self.global_lr
        )

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Perform dual training: update both global model and personal model.
        """
        inputs, targets = batch
        batch_size = inputs.size(0)

        # Step 1: Update global model with standard loss
        global_outputs = self.global_model(inputs)
        global_loss = nn.functional.cross_entropy(global_outputs, targets)

        self.global_optimizer.zero_grad()
        global_loss.backward()
        self.global_optimizer.step()

        # Step 2: Update personal model with proximal regularization
        personal_outputs = self.local_model(inputs)
        personal_loss = nn.functional.cross_entropy(personal_outputs, targets)

        # Add proximal regularization term
        proximal_reg = 0.0
        for local_param, global_param in zip(
            self.local_model.parameters(), self.global_model.parameters()
        ):
            diff = local_param - global_param.detach()
            # power is generally more expensive than multiplication
            proximal_reg += torch.sum(diff * diff)

        total_loss = personal_loss + 0.5 * self.ditto_lambda * proximal_reg

        return total_loss

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        Ditto aggregation: aggregate global models while keeping personal models local.
        """
        # All nodes participate regardless of sample count
        utils.scale_params(self.global_model, weight)

        # Aggregate global models (personal models remain local)
        self.global_model = comm.aggregate(
            self.global_model,
            reduction=AggregationOp.SUM,
        )

        # Return the personal local model, not the aggregated global model
        return self.local_model
