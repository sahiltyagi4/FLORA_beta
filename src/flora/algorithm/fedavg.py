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

from typing import Any

import rich.repr
import torch
from torch import nn

from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedAvg(BaseAlgorithm):
    """
    Federated Averaging (FedAvg) algorithm implementation.

    FedAvg performs standard federated learning by averaging model parameters across clients after local training rounds.
    Only model parameters are aggregated; all clients synchronize with the global model at the start of each round.

    [FedAvg](https://arxiv.org/abs/1602.05629) | H. Brendan McMahan | 2016-02-17
    """

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Forward pass and compute the cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        return loss
