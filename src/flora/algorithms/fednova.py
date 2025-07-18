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

from src.flora.helper.node_config import NodeConfig
from src.flora.helper.training_params import FedNovaTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm


class FedNova:
    """
    Implementation of Federated Normalized Averaging or FedNova

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedNovaTrainingParameters,
    ):
        """
        Initialize FedNova algorithm instance.

        Args:
            model (torch.nn.Module): Model to train.
            train_data (torch.utils.data.DataLoader): Local training data.
            communicator (Communicator): Communication interface.
            total_clients (int): Number of clients in the federation.
            train_params (FedNovaTrainingParameters): Training hyperparameters.
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
        self.weight_decay = self.train_params.get_weight_decay()
        self.local_step = 0
        self.training_samples = 0
        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.diff_params = copy.deepcopy(self.model)
        self.global_model, self.diff_params = (
            self.global_model.to(self.device),
            self.diff_params.to(self.device),
        )

    def broadcast_model(self, model):
        # broadcast model from central server with id 0
        model = self.communicator.broadcast(msg=model, id=0)
        return model

    def compute_alpha(self, lr):
        momentum_term = 1 - lr * self.weight_decay
        alpha = 0.0
        for j in range(self.comm_freq):
            alpha += momentum_term**j
        alpha *= lr
        return alpha

    def normalized_update(self, weight_scaling: float):
        """sends normalized updates and receives scaled, aggregated update"""
        lr = self.optimizer.param_groups[0]["lr"]
        alpha = self.compute_alpha(lr)
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.global_model.named_parameters(), self.model.named_parameters()
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                target_param = dict(self.diff_params.named_parameters())[name1]
                target_param.copy_((weight_scaling * (param1 - param2)) / alpha)

        self.diff_params = self.communicator.aggregate(
            msg=self.diff_params, compute_mean=False
        )

    def model_update(self):
        lr = self.optimizer.param_groups[0]["lr"]
        with torch.no_grad():
            for (name1, param1), (name2, param_delta) in zip(
                self.global_model.named_parameters(),
                self.diff_params.named_parameters(),
            ):
                assert name1 == name2, f"Parameter mismatch: {name1} vs {name2}"
                param1 -= lr * param_delta

    def train_loop(self):
        for inputs, labels in self.train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model(inputs)
            loss = self.loss(pred, labels)
            self.training_samples += inputs.size(0)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            if self.local_step % self.comm_freq == 0:
                # total samples processed across all clients
                total_samples = self.communicator.aggregate(
                    msg=torch.Tensor([self.training_samples]), compute_mean=False
                )
                weight_scaling = self.training_samples / total_samples.item()
                self.normalized_update(weight_scaling)
                self.model_update()
                self.model.load_state_dict(self.global_model.state_dict())

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
class FedNovaNew(Algorithm):
    """
    Implementation of Federated Normalized Averaging (FedNova).

    FedNova normalizes local updates to address objective inconsistency in federated learning,
    accounting for varying numbers of local steps and learning dynamics across clients.
    """

    def __init__(self, weight_decay: float = 0.0, **kwargs):
        """Initialize FedNova algorithm with weight decay parameter."""
        super().__init__(**kwargs)
        self.weight_decay = weight_decay

    def _setup(self, device: torch.device) -> None:
        """
        FedNova-specific setup: initialize global model and step counter.
        """
        super()._setup(device=device)

        self.global_model = copy.deepcopy(self.local_model)
        self.local_steps_this_round: int = 0

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer with weight decay.
        """
        return torch.optim.SGD(
            self.local_model.parameters(), lr=local_lr, weight_decay=self.weight_decay
        )

    def _train_step(self, batch: Any) -> tuple[torch.Tensor, int]:
        """
        Perform a forward pass and compute cross-entropy loss for a batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        return loss, inputs.size(0)

    def _round_start(self) -> None:
        """
        Update global model reference and reset step counter at the start of each round.

        # TODO: check whether we can safely just move all this logic in round_start() for all algorithms to the end of aggregate() method and remove round_start() overrides altogether
        # TODO: should this logic be linked with the same granularity as aggregate(), rather than always on round_start?
        """
        # Update global model reference (self.local_model already contains latest from aggregate())
        self.global_model.load_state_dict(self.local_model.state_dict())
        # Reset local step counter for normalization calculations
        self.local_steps_this_round = 0

    def _optimizer_step(self) -> None:
        """
        Perform an optimizer step and increment the local step counter for normalization.
        """
        self.local_optimizer.step()
        self.local_steps_this_round += 1

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

    def _aggregate(self) -> None:
        """
        FedNova aggregation: normalized averaging based on local training steps.

        NOTE: Compatible with all granularity levels.
        - Normalization factor adapts automatically based on actual steps taken since last aggregation.
        """
        lr = self.local_optimizer.param_groups[0]["lr"]
        alpha = self._compute_alpha(lr, self.local_steps_this_round)

        # Compute normalized parameter deltas for each trainable parameter
        normalized_deltas: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                global_param = dict(self.global_model.named_parameters())[name]
                normalized_deltas[name] = (global_param.data - param.data) / alpha

        # Aggregate local sample counts to compute federation total

        global_samples = self.local_comm.aggregate(
            torch.tensor([self.local_sample_count], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Handle edge cases safely - all nodes must participate in distributed operations
        if global_samples <= 0:
            data_proportion = 0.0
        else:
            # Calculate the proportion of data this client contributed
            data_proportion = self.local_sample_count / global_samples

        # Scale normalized deltas by the data proportion for weighted aggregation
        for name, delta in normalized_deltas.items():
            delta.mul_(data_proportion)

        # Aggregate normalized deltas
        aggregated_deltas = self.local_comm.aggregate(
            msg=normalized_deltas,
            reduction=ReductionType.SUM,
        )

        # Apply the aggregated normalized updates to the global model parameters
        utils.apply_model_delta(self.global_model, aggregated_deltas, scale=lr)

        # Update the local model to match the updated global model
        self.local_model = copy.deepcopy(self.global_model)
