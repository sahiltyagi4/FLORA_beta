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

import torch
import torch.nn as nn


class TrainingParameters:
    def __init__(self, **kwargs):
        self.optimizer = kwargs.get("optimizer", None)
        self.loss = kwargs.get("loss", None)
        self.epochs = kwargs.get("epochs", None)

    def get_optimizer(self):
        return self.optimizer

    def get_loss(self):
        return self.loss

    def get_epochs(self):
        return self.epochs


class FedAvgTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)

    def get_comm_freq(self):
        return self.comm_freq


class DiLocoTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.outer_lr = kwargs.get("outer_lr", None)
        self.outer_momentum = kwargs.get("outer_momentum", None)

    def get_comm_freq(self):
        return self.comm_freq

    def get_outer_lr(self):
        return self.outer_lr

    def get_outer_momentum(self):
        return self.outer_momentum


class FedProxTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.mu = kwargs.get("mu", None)

    def get_comm_freq(self):
        return self.comm_freq

    def get_mu(self):
        return self.mu


class FedMomTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.lr = kwargs.get("lr", None)
        self.momentum = kwargs.get("momentum", None)

    def get_comm_freq(self):
        return self.comm_freq

    def get_lr(self):
        return self.lr

    def get_momentum(self):
        return self.momentum


class FedBNTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)

    def get_comm_freq(self):
        return self.comm_freq


class FedNovaTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.weight_decay = kwargs.get("weight_decay", None)

    def get_comm_freq(self):
        return self.comm_freq

    def get_weight_decay(self):
        return self.weight_decay


class ScaffoldTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)

    def get_comm_freq(self):
        return self.comm_freq


class FedDynTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.regularizer_alpha = kwargs.get("regularizer_alpha", None)

    def get_comm_freq(self):
        return self.comm_freq

    def get_regularizer_alpha(self):
        return self.regularizer_alpha


class MOONTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        """
        :param comm_freq: iterations after which collect/aggregate updates
        :param num_prev_models: num of previous models to keep track
        :param temperature: temperature parameter between representations
        :param mu: contrastive loss weight
        """
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.num_prev_models = kwargs.get("num_prev_models", None)
        self.temperature = kwargs.get("temperature", None)
        self.mu = kwargs.get("mu", None)

    def get_comm_freq(self):
        return self.comm_freq

    def get_num_prev_models(self):
        return self.num_prev_models

    def get_temperature(self):
        return self.temperature

    def get_mu(self):
        return self.mu


class FedPerTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)

    def get_comm_freq(self):
        return self.comm_freq


class DittoTrainingParameters(TrainingParameters):
    def __init__(self, **kwargs):
        """
        :param comm_freq: iterations after which collect/aggregate updates
        :param ditto_regularizer: regularization parameter controlling closeness between local and global model
        """
        super().__init__(**kwargs)
        self.comm_freq = kwargs.get("comm_freq", None)
        self.ditto_regularizer = kwargs.get("ditto_regularizer", None)
        self.global_loss = kwargs.get("global_loss", None)
        self.global_optimizer = kwargs.get("global_optimizer", None)

    def get_comm_freq(self):
        return self.comm_freq

    def get_ditto_regularizer(self):
        return self.ditto_regularizer

    def get_global_loss(self):
        return self.global_loss

    def get_global_optimizer(self):
        return self.global_optimizer


class MLPModel(nn.Module):
    """basic fully-connected network"""

    def __init__(self, grad_dim, hidden_dim=1024, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(grad_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, grad_dim),
        )

    def forward(self, large_batch_update):
        return self.net(large_batch_update)


if __name__ == "__main__":
    model = MLPModel(grad_dim=10)
    optimer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    comm_freq = 50

    params = TrainingParameters(optimizer=optimer, comm_freq=comm_freq, loss=loss)
    print(params.epochs)
