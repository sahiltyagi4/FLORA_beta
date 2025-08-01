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

from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """
    Base structured config for PyTorch neural network models.
    
    The _target_ must point to a torch.nn.Module subclass that can be instantiated
    by Hydra. Model-specific parameters are added through config inheritance.
    
    Examples:
    - ComposableModel (backbone + head architecture)
    - Pre-trained models (ResNet, EfficientNet, etc.)
    - Custom neural networks
    """
    
    _target_: str = MISSING  # Must be torch.nn.Module subclass path