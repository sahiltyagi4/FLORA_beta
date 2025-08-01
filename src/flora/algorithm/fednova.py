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
import torch.nn as nn

from ..communicator import AggregationOp, BaseCommunicator
from . import utils
from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedNova(BaseAlgorithm):
    """
    Implementation of Federated Normalized Averaging (FedNova).

    FedNova normalizes local updates to address objective inconsistency in federated learning,
    accounting for varying numbers of local steps and learning dynamics across clients.

    [FedNova](https://arxiv.org/abs/2007.07481) | Jianyu Wang | 2020-07-15
    """

    def __init__(self, weight_decay: float = 0.0, **kwargs):
        """Initialize FedNova algorithm with weight decay parameter."""
        super().__init__(**kwargs)
        self.weight_decay = weight_decay

    def _setup(self, *args, **kwargs) -> None:
        """
        FedNova-specific setup: initialize global model and step counter.
        """
        super()._setup(*args, **kwargs)

        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Global model is reference-only for delta computation, set eval mode and disable gradients
        self.global_model.eval()  # eval() does NOT turn off gradient tracking.
        for param in self.global_model.parameters():
            param.requires_grad = False

        self.optimizer_steps: int = 0

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer with weight decay.
        """
        return torch.optim.SGD(
            self.local_model.parameters(), lr=local_lr, weight_decay=self.weight_decay
        )

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Perform a forward pass and compute cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss

    def _optimizer_step(self) -> None:
        """
        Perform an optimizer step and increment the local step counter for normalization.
        """
        self.local_optimizer.step()
        self.optimizer_steps += 1

    def _compute_alpha(self, lr: float, local_steps: int) -> float:
        """
        Compute the normalization coefficient alpha.
        """
        if local_steps <= 0:
            return lr
        momentum_term = 1 - lr * self.weight_decay
        alpha = 0.0
        for j in range(local_steps):
            alpha += momentum_term**j
        alpha *= lr
        return alpha

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        FedNova aggregation: normalized averaging based on local training steps.
        """
        lr = self.local_optimizer.param_groups[0]["lr"]
        alpha = self._compute_alpha(lr, self.optimizer_steps)

        # Compute normalized parameter deltas for each trainable parameter
        normalized_deltas: Dict[str, torch.Tensor] = {}
        # Pre-compute global parameters dictionary once to avoid O(n^2) complexity
        global_params = dict(self.global_model.named_parameters())
        for param_name, local_param in self.local_model.named_parameters():
            if local_param.requires_grad:
                global_param = global_params[param_name]  # O(1) lookup
                normalized_deltas[param_name] = (
                    local_param.data - global_param.data
                ) / alpha

        # Scale normalized deltas by the data proportion for weighted aggregation
        with torch.no_grad():
            for param_name, delta in normalized_deltas.items():
                delta.mul_(weight)

        # Aggregate normalized deltas
        aggregated_deltas = comm.aggregate(
            msg=normalized_deltas,
            reduction=AggregationOp.SUM,
        )

        # Apply the aggregated normalized updates to the global model parameters
        utils.add_model_deltas(self.global_model, aggregated_deltas, alpha=lr)

        # Reset optimizer steps counter after aggregation since we're starting new local training
        # so that control variates are properly normalized for the next aggregation period
        self.optimizer_steps = 0

        # Return updated global model as the new local model for next training period
        return copy.deepcopy(self.global_model)
