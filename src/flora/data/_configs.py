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
from typing import Optional, Any
from omegaconf import MISSING


@dataclass
class DataModuleConfig:
    """
    Structured config for FL data loading and preprocessing.
    
    Both train and eval DataLoaders are optional to support different
    FL scenarios (some nodes only train, others only evaluate).
    """
    
    _target_: str = "src.flora.data.DataModule.DataModule"
    
    train: Optional[Any] = None  # DataLoader config for training data
    eval: Optional[Any] = None   # DataLoader config for evaluation data