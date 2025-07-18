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
from torch import nn

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import ScaffoldTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm


class Scaffold:
    """Implementation of Stochastic Controlled Averaging algorithm or SCAFFOLD

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: ScaffoldTrainingParameters,
    ):
        """
        :param model: model to train
        :param data: data to train
        :param communicator: communicator object
        :param total_clients: total number of clients / world size
        :param train_params: training hyperparameters
        """
        self.model = model
        self.train_data = train_data
        self.communicator = communicator
        self.total_clients = total_clients
        self.train_params = train_params
        self.optimizer = self.train_params.get_optimizer()
        self.comm_freq = self.train_params.get_comm_freq()
        self.loss = self.train_params.get_loss()
        self.epochs = self.train_params.get_epochs()
        self.local_step = 0
        self.training_samples = 0

        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)
        self.server_control_variates = {
            name: torch.zeros_like(param.data).to(self.device)
            for name, param in self.model.named_parameters()
        }
        self.client_control_variates = {
            name: torch.zeros_like(param.data).to(self.device)
            for name, param in self.model.named_parameters()
        }
        self.old_client_control_variates = {
            name: torch.zeros_like(param.data).to(self.device)
            for name, param in self.model.named_parameters()
        }
        self.model_delta = {
            name: torch.zeros_like(param.data).to(self.device)
            for name, param in self.model.named_parameters()
        }
        self.control_variate_delta = {
            name: torch.zeros_like(param.data).to(self.device)
            for name, param in self.model.named_parameters()
        }

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            self.training_samples += inputs.size(0)
            loss.backward()
            # update gradients to account for client drift
            for name, param in self.model.named_parameters():
                param.grad.add_(
                    self.server_control_variates[name]
                    - self.client_control_variates[name]
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            lr = self.optimizer.param_groups[0]["lr"]

            # update client-local control variate
            for (name1, param1), (name2, param2) in zip(
                self.global_model.named_parameters(), self.model.named_parameters()
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                # update old state of client control-variate
                self.old_client_control_variates[name1].copy_(
                    self.client_control_variates[name1]
                )
                self.client_control_variates[name1].copy_(
                    self.client_control_variates[name1]
                    - self.server_control_variates[name1]
                    + (param2 - param1) / (self.comm_freq * lr)
                )

            if self.local_step % self.comm_freq == 0:
                # compute client specific model-delta and control variate delta
                for (name1, param1), (name2, param2) in zip(
                    self.global_model.named_parameters(), self.model.named_parameters()
                ):
                    assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                    self.model_delta[name1].copy_(param2.data - param1.data)
                    self.control_variate_delta[name1].copy_(
                        self.client_control_variates[name1]
                        - self.old_client_control_variates[name1]
                    )

                # average model-deltas and control-variate deltas across clients
                avg_model_delta = self.communicator.aggregate(
                    msg=self.model_delta, compute_mean=True
                )
                avg_control_variate_delta = self.communicator.aggregate(
                    msg=self.control_variate_delta, compute_mean=True
                )

                # update the global model and server control variates
                for name, param in self.global_model.named_parameters():
                    param.add_(lr * avg_model_delta[name])
                    self.server_control_variates[name].add_(
                        avg_control_variate_delta[name]
                    )

                self.model.load_state_dict(self.global_model.state_dict())
                del avg_model_delta, avg_control_variate_delta

    def train(self):
        self.model = self.broadcast_model(model=self.model)
        if self.epochs is not None and isinstance(self.epochs, int) and self.epochs > 0:
            for epoch in range(self.epochs):
                self.train_loop()
        else:
            while True:
                self.train_loop()


# ======================================================================================


@rich.repr.auto
class ScaffoldNew(Algorithm):
    """
    SCAFFOLD (Stochastic Controlled Averaging) algorithm implementation.

    SCAFFOLD addresses client drift in federated learning by maintaining control variates
    that estimate the update direction difference between local and global objectives.
    Gradient correction and control variate updates are performed each round.
    """

    def __init__(self, **kwargs):
        """Initialize SCAFFOLD algorithm with granularity validation."""
        super().__init__(**kwargs)

        # SCAFFOLD requires ROUND-level aggregation for proper control variate mathematics
        if self.agg_level != "ROUND":
            raise ValueError(
                f"SCAFFOLD requires agg_level='ROUND' for proper control variate mathematics. "
                f"Got agg_level='{self.agg_level}'. EPOCH/ITER levels would break the optimizer_steps "
                f"counter and control variate calculations across batches."
            )

    def _setup(self, device: torch.device) -> None:
        """
        SCAFFOLD-specific setup: initialize control variates and tracking structures.
        """
        super()._setup(device=device)

        self.server_cv = {}
        self.client_cv = {}
        self.old_client_cv = {}
        self.global_model = copy.deepcopy(self.local_model)
        self.model_delta = {}
        self.cv_delta = {}
        self.optimizer_steps = 0

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

    def _train_step(
        self,
        batch: Any,
    ) -> Tuple[torch.Tensor, int]:
        """
        Forward pass and compute cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss, inputs.size(0)

    def _optimizer_step(self) -> None:
        """
        Track optimizer steps for control variate normalization.
        """
        self.local_optimizer.step()
        self.optimizer_steps += 1

    def _round_start(self) -> None:
        """
        Reset optimizer step count and update global model reference at the start of each round.

        # TODO: check whether we can safely just move all this logic in round_start() for all algorithms to the end of aggregate() method and remove round_start() overrides altogether
        # TODO: should this logic be linked with the same granularity as aggregate(), rather than always on round_start?
        """
        # Update global model reference (self.local_model already contains latest from aggregate())
        self.global_model.load_state_dict(self.local_model.state_dict())
        # Reset optimizer step counter for SCAFFOLD control variate calculations
        self.optimizer_steps = 0

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """
        Apply SCAFFOLD gradient correction after backward pass.
        """
        loss.backward()
        for name, param in self.local_model.named_parameters():
            if param.grad is not None and name in self.server_cv:
                param.grad.add_(self.server_cv[name] - self.client_cv[name])

    def _aggregate(self) -> None:
        """
        SCAFFOLD aggregation: aggregate model deltas and control variate deltas.

        NOTE: SCAFFOLD REQUIRES ROUND-level aggregation ONLY (agg_level=ROUND, agg_freq=1)
        - Control variate mathematics depend on complete local training cycles
        - optimizer_steps counter assumes full round completion
        - EPOCH/ITER level aggregation would break control variate calculations
        """
        effective_comm_freq = max(1, self.optimizer_steps)
        lr = self.local_optimizer.param_groups[0]["lr"]

        # Update client control variates
        for (name1, param1), (name2, param2) in zip(
            self.global_model.named_parameters(), self.local_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
            self.old_client_cv[name1].copy_(self.client_cv[name1])
            update_term = (param2 - param1) / (effective_comm_freq * lr)
            # TODO: Verify control variate update formula against SCAFFOLD paper
            self.client_cv[name1].sub_(self.server_cv[name1]).add_(update_term)

        # Compute model delta and control variate delta
        for (name1, param1), (name2, param2) in zip(
            self.global_model.named_parameters(), self.local_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
            self.model_delta[name1].copy_(param2.data).sub_(param1.data)
            self.cv_delta[name1].copy_(self.client_cv[name1]).sub_(
                self.old_client_cv[name1]
            )

        # SCAFFOLD uses mean aggregation rather than weighted aggregation
        aggregated_model_deltas = self.local_comm.aggregate(
            msg=self.model_delta, reduction=ReductionType.MEAN
        )
        aggregated_cv_deltas = self.local_comm.aggregate(
            msg=self.cv_delta, reduction=ReductionType.MEAN
        )

        lr = self.local_optimizer.param_groups[0]["lr"]
        utils.apply_model_delta(self.global_model, aggregated_model_deltas, scale=lr)
        for name in self.server_cv:
            if name in aggregated_cv_deltas:
                self.server_cv[name].add_(aggregated_cv_deltas[name])

        self.local_model.load_state_dict(self.global_model.state_dict())
