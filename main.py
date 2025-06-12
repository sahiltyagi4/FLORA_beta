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


import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from hydra.utils import instantiate

from src.flora.config.hydra_configs import FLConfig


class SimpleModel(nn.Module):
    """Dummy demo model"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dummy_data(num_samples: int = 100) -> TensorDataset:
    """Dummy demo random data"""
    x = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(x, y)


@hydra.main(version_base=None, config_path="test_configs", config_name="test1")
def main(cfg: FLConfig) -> None:
    print("#" * 80)
    print(OmegaConf.to_yaml(cfg))

    # Create a simple model and dataset for demonstration
    model = SimpleModel()
    data = create_dummy_data(100)

    data_loader = DataLoader(data, batch_size=32, shuffle=True)

    # Instantiate algorithm using Hydra
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/
    algorithm = instantiate(
        cfg.algorithm,
        # Passing in the model and train_data manually.
        model=model,
        train_data=data_loader,
    )

    # Start training
    algorithm.train()


if __name__ == "__main__":
    main()
