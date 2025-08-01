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
from typing import Any, Tuple

import rich.repr
import torch
from torch import nn

from ..communicator import AggregationOp, BaseCommunicator
from . import utils
from .base import BaseAlgorithm

# ======================================================================================


@rich.repr.auto
class Scaffold(BaseAlgorithm):
    """
    SCAFFOLD (Stochastic Controlled Averaging) algorithm implementation.

    SCAFFOLD addresses client drift in federated learning by maintaining control variates
    that estimate the update direction difference between local and global objectives.
    Gradient correction and control variate updates are performed each round.
    """

    def _setup(self, *args, **kwargs) -> None:
        """
        SCAFFOLD-specific setup: initialize control variates and tracking structures.
        """
        super()._setup(*args, **kwargs)

        # Deep-copy retains requires_grad state from local_model
        self.global_model = copy.deepcopy(self.local_model)
        # Global model is reference-only for delta computation, set eval mode and disable gradients
        self.global_model.eval()  # eval() does NOT turn off gradient tracking.
        for param in self.global_model.parameters():
            param.requires_grad = False

        self.server_cv = {}
        self.client_cv = {}
        self.old_client_cv = {}
        self.model_delta = {}
        self.cv_delta = {}
        self.optimizer_steps = 0

        # Initialize control variates and deltas
        # all zero-initialized tensors based on param.data have requires_grad=False by default
        for name, param in self.local_model.named_parameters():
            self.server_cv[name] = torch.zeros_like(param.data)
            self.client_cv[name] = torch.zeros_like(param.data)
            self.old_client_cv[name] = torch.zeros_like(param.data)
            self.model_delta[name] = torch.zeros_like(param.data)
            self.cv_delta[name] = torch.zeros_like(param.data)

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
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

    def _optimizer_step(self) -> None:
        """
        Track optimizer steps for control variate normalization.
        """
        self.local_optimizer.step()
        self.optimizer_steps += 1

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """
        Apply SCAFFOLD gradient correction after backward pass.
        """
        loss.backward()
        for name, param in self.local_model.named_parameters():
            if param.grad is not None and name in self.server_cv:
                param.grad.add_(self.server_cv[name] - self.client_cv[name])

    def _aggregate_within_group(
        self, comm: BaseCommunicator, weight: float
    ) -> nn.Module:
        """
        SCAFFOLD aggregation: aggregate model deltas and control variate deltas.
        """
        effective_comm_freq = max(1, self.optimizer_steps)
        lr = self.local_optimizer.param_groups[0]["lr"]

        # Update client control variates and compute deltas
        with torch.no_grad():
            for (global_param_name, global_param), (
                local_param_name,
                local_param,
            ) in zip(
                self.global_model.named_parameters(),
                self.local_model.named_parameters(),
            ):
                assert global_param_name == local_param_name, (
                    f"Parameter mismatch: {global_param_name} vs {local_param_name}"
                )
                param_name = global_param_name

                # Save current control variate state before updating
                self.old_client_cv[param_name].copy_(self.client_cv[param_name])

                # Client control variate update
                update_term = (local_param - global_param) / (effective_comm_freq * lr)
                self.client_cv[param_name].sub_(self.server_cv[param_name]).add_(
                    update_term
                )

                # Compute model delta and control variate delta for aggregation
                self.model_delta[param_name].copy_(local_param.data).sub_(
                    global_param.data
                )
                self.cv_delta[param_name].copy_(self.client_cv[param_name]).sub_(
                    self.old_client_cv[param_name]
                )

        # SCAFFOLD uses mean aggregation rather than weighted aggregation
        aggregated_model_deltas = comm.aggregate(
            msg=self.model_delta, reduction=AggregationOp.MEAN
        )
        aggregated_cv_deltas = comm.aggregate(
            msg=self.cv_delta, reduction=AggregationOp.MEAN
        )

        # Update Global model with aggregated deltas and control variates
        utils.add_model_deltas(self.global_model, aggregated_model_deltas, alpha=lr)
        # Update server control variates with aggregated deltas
        with torch.no_grad():
            for name in self.server_cv:
                if name in aggregated_cv_deltas:
                    self.server_cv[name].add_(aggregated_cv_deltas[name])

        # Reset optimizer steps counter after aggregation since we're starting new local training
        # so that control variates are properly normalized for the next aggregation period
        self.optimizer_steps = 0

        # Return updated global model as the new local model for next training period
        return copy.deepcopy(self.global_model)
