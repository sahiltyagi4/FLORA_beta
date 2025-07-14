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
from src.flora.helper.training_params import FedDynTrainingParameters

from ..communicator import Communicator, ReductionType
from . import utils
from .BaseAlgorithm import Algorithm


class FedDyn:
    """
    Implementation of Federated Dynamic-Regularizer or FedDyn

    NOTE: Original implementation kept for reference purposes
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        communicator: Communicator,
        total_clients: int,
        train_params: FedDynTrainingParameters,
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
        self.regularizer_alpha = self.train_params.get_regularizer_alpha()
        self.local_step = 0
        self.training_samples = 0
        dev_id = NodeConfig().get_gpus() % self.total_clients
        self.device = torch.device(
            "cuda:" + str(dev_id) if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.dynamic_correction = torch.zeros_like(
            torch.nn.utils.parameters_to_vector(self.model.parameters())
        ).to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)

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

            param_2_vec = torch.nn.utils.parameters_to_vector(self.model.parameters())
            regularization_loss = (
                self.regularizer_alpha * torch.norm(param_2_vec) ** 2
            ) / 2 - torch.dot(self.dynamic_correction, param_2_vec)
            fed_dyn_loss = loss + regularization_loss
            del param_2_vec

            fed_dyn_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.local_step += 1
            self.dynamic_correction += self.regularizer_alpha * (
                torch.nn.utils.parameters_to_vector(self.model.parameters())
                - torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            )

            if self.local_step % self.comm_freq == 0:
                total_samples = self.communicator.aggregate(
                    msg=torch.Tensor([self.training_samples]), compute_mean=False
                )

                weight_scaling = self.training_samples / total_samples.item()
                for _, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    param.data *= weight_scaling

                self.global_model = self.communicator.aggregate(
                    msg=self.model, compute_mean=False
                )
                self.model.load_state_dict(self.global_model.state_dict())
                self.training_samples = 0

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
class FedDynNew(Algorithm):
    """
    Federated Dynamic Regularization (FedDyn) algorithm implementation.

    FedDyn introduces a dynamic regularization term to address objective inconsistency in federated learning and improve convergence in heterogeneous environments.
    """

    def __init__(self, alpha: float = 0.1, **kwargs):
        """Initialize FedDyn algorithm with regularization parameter and granularity validation."""
        super().__init__(**kwargs)
        self.alpha = alpha

        # FedDyn works best at ROUND level for optimal dynamic regularization mathematics
        if self.agg_level != "ROUND":
            import warnings

            warnings.warn(
                f"FedDyn designed for agg_level='ROUND' but got '{self.agg_level}'. "
                f"EPOCH/ITER levels may work but could affect regularization convergence.",
                UserWarning,
            )

    def _setup(self) -> None:
        """
        FedDyn-specific setup: initialize server momentum.
        """
        super()._setup()

        self.server_momentum: Dict[str, torch.Tensor] = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                self.server_momentum[name] = torch.zeros_like(param.data)

    def _configure_local_optimizer(self, local_lr: float) -> torch.optim.Optimizer:
        """
        SGD optimizer for local updates.
        """
        return torch.optim.SGD(self.local_model.parameters(), lr=local_lr)

    def _train_step(self, batch: Any, batch_idx: int) -> tuple[torch.Tensor, int]:
        """
        Perform a forward pass and compute the FedDyn loss for a single batch.
        """
        inputs, targets = batch
        outputs = self.local_model(inputs)

        # Standard cross-entropy loss
        base_loss = torch.nn.functional.cross_entropy(outputs, targets)

        # Add FedDyn dynamic regularization if state is available
        regularization_loss = torch.tensor(0.0, device=inputs.device)
        # Quadratic term
        param_norm_squared = 0.0
        for param in self.local_model.parameters():
            if param.requires_grad:
                param_norm_squared += torch.sum(param**2)
        quadratic_term = (self.alpha / 2.0) * param_norm_squared

        # Linear term
        linear_term = 0.0
        for name, param in self.local_model.named_parameters():
            if param.requires_grad and name in self.server_momentum:
                linear_term -= torch.sum(self.server_momentum[name] * param)

        regularization_loss = quadratic_term + linear_term

        total_loss = base_loss + regularization_loss
        return total_loss, inputs.size(0)

    def _aggregate(self) -> None:
        """
        FedDyn aggregation: weighted averaging with dynamic regularization momentum.

        NOTE: Works best at ROUND level but may be compatible with other granularities
        - Dynamic regularization mathematics designed for complete training cycles
        - EPOCH/ITER levels may work but could affect regularization convergence
        """

        # Store local model before aggregation for momentum update
        local_model_params = {}
        for name, param in self.local_model.named_parameters():
            if param.requires_grad:
                local_model_params[name] = param.data.clone()

        # Aggregate local sample counts to compute federation total
        global_samples = self.comm.aggregate(
            torch.tensor([self.local_sample_count], dtype=torch.float32),
            reduction=ReductionType.SUM,
        ).item()

        # Handle edge cases safely - all nodes must participate in distributed operations
        if global_samples <= 0:
            data_proportion = 0.0
        else:
            # Calculate data proportion for weighted aggregation
            data_proportion = self.local_sample_count / global_samples

        # All nodes participate regardless of sample count
        utils.scale_params(self.local_model, data_proportion)

        # Aggregate weighted model parameters
        aggregated_model = self.comm.aggregate(
            self.local_model,
            reduction=ReductionType.SUM,
        )

        # Update server momentum (dynamic regularizer)
        for name, param in aggregated_model.named_parameters():
            if param.requires_grad and name in self.server_momentum:
                # Compute model difference: local - global
                model_diff = local_model_params[name] - param.data

                # TODO: Verify server momentum update direction against FedDyn paper
                # Update server momentum: h = h + alpha * (local - global)
                self.server_momentum[name].add_(model_diff, alpha=self.alpha)

        # Update local model to aggregated result
        self.local_model = aggregated_model
