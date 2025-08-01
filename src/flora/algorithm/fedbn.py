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
import torch.nn as nn

from ..communicator import AggregationOp, BaseCommunicator
from . import utils
from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class FedBN(BaseAlgorithm):
    """
    Federated Batch Normalization (FedBN) algorithm implementation.

    FedBN aggregates only non-batch normalization parameters across clients,
    allowing each client to maintain its own batch normalization statistics for improved personalization.

    [FedBN](https://arxiv.org/abs/2102.07623) | Xiaoxiao Li | 2021-02-15
    """

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
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
        FedBN aggregation: aggregate non-BatchNorm parameters while keeping BN layers local.

        BatchNorm layers are kept local because they capture client-specific data statistics.
        Aggregating BN parameters would mix statistics from different data distributions.
        """
        # Save local BN parameters before aggregation
        local_bn_params = {}
        for param_name, local_param in self.local_model.named_parameters():
            if self._is_bn_layer(param_name):
                local_bn_params[param_name] = local_param.data.clone()

        # Scale only non-BN parameters by data proportion (all nodes participate)
        utils.scale_params(
            self.local_model,
            weight,
            filter_fn=lambda name, tensor: not self._is_bn_layer(name),
        )

        # Aggregate non-BN parameters
        aggregated_model = comm.aggregate(
            self.local_model,
            reduction=AggregationOp.SUM,
        )

        # Restore local BN parameters
        with torch.no_grad():
            for param_name, aggregated_param in aggregated_model.named_parameters():
                if self._is_bn_layer(param_name) and param_name in local_bn_params:
                    aggregated_param.data.copy_(local_bn_params[param_name])

        return aggregated_model

    def _is_bn_layer(self, param_name: str) -> bool:
        """
        Check if parameter belongs to a batch normalization layer.
        """
        # TODO: More robust BN detection instead of string matching
        return "bn" in param_name or "norm" in param_name
