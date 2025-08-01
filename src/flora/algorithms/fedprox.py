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
from torch import nn

from ..communicator import ReductionType
from . import utils
from .BaseAlgorithm import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedProx(BaseAlgorithm):
    """
    FedProx algorithm implementation.

    FedProx extends FedAvg by adding a proximal term to the loss, which helps stabilize training in heterogeneous environments.
    The proximal term penalizes deviation from the global model during local updates.

    [FedProx](https://arxiv.org/abs/1812.06127) | Tian Li | 2018-12-14
    """

    def __init__(self, mu: float = 0.01, **kwargs):
        """Initialize FedProx algorithm with proximal term parameter."""
        super().__init__(**kwargs)
        self.mu = mu

    def _setup(self, device: torch.device) -> None:
        """
        FedProx-specific setup: initialize global model for proximal term.
        """
        super()._setup(device=device)

        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Global model is reference-only for proximal term, set eval mode and disable gradients
        self.global_model.eval()  # eval() does NOT turn off gradient tracking.
        for param in self.global_model.parameters():
            param.requires_grad = False

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _train_step(self, batch: Any) -> tuple[torch.Tensor, int]:
        """
        Forward pass and compute the FedProx loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        # Compute standard cross-entropy loss
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        # Add proximal term to penalize deviation from the global model
        prox_term = 0.0
        for (_, local_param), (_, global_param) in zip(
            self.local_model.named_parameters(), self.global_model.named_parameters()
        ):
            if local_param.requires_grad:
                diff = local_param - global_param
                # power is generally more expensive than multiplication
                prox_term += torch.sum(diff * diff)
        loss += (self.mu / 2) * prox_term
        return loss, inputs.size(0)

    def _aggregate(self) -> nn.Module:
        """
        FedProx aggregation: weighted averaging of model parameters.
        """
        # Aggregate local sample counts to compute federation total
        global_samples = self.local_comm.aggregate(
            torch.tensor([self.local_sample_count], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Calculate this client's data proportion for weighted aggregation
        data_proportion = self.local_sample_count / max(global_samples, 1)

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, data_proportion)

        # Aggregate weighted model parameters from all clients
        aggregated_model = self.local_comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )

        return aggregated_model
