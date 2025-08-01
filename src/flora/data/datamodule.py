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

from typing import Any, Optional

import rich.repr
from torch.utils.data import DataLoader
from typeguard import typechecked


@rich.repr.auto
class DataModule:
    """
    Data loading container for federated learning experiments.

    Encapsulates training and evaluation DataLoaders for a single node.
    Provides consistent interface for algorithm access to local data.

    Typically created by Hydra configuration and passed to Node constructors.
    Algorithms access data via node.datamodule.train and node.datamodule.eval.

    See conf/datamodule/ for configuration examples.
    """

    @typechecked
    def __init__(
        self,
        train: Optional[DataLoader[Any]] = None,
        eval: Optional[DataLoader[Any]] = None,
    ):
        """
        Initialize data module with PyTorch DataLoaders.

        Args:
            train: DataLoader for training data (local to this node)
            eval: DataLoader for evaluation data (local to this node)
        """
        print("[DATAMODULE-INIT]")

        self.train: Optional[DataLoader[Any]] = train
        self.eval: Optional[DataLoader[Any]] = eval
