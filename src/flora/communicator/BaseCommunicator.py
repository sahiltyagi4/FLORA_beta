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

from abc import ABC, abstractmethod
from typing import Union, TypeVar, Dict

import rich.repr
import torch
import torch.nn as nn


# ======================================================================================


@rich.repr.auto
class Communicator(ABC):
    """
    Abstract communication interface.

    - Defines standard communication primitives
    - Abstracts away protocol-specific implementation details
    - Abstractions should be topology-agnostic
    """

    MsgT = TypeVar("MsgT", nn.Module, torch.Tensor, Dict[str, torch.Tensor])

    @abstractmethod
    def setup(self):
        """
        Initialize the communication layer.
        This method should be called before any communication operations.
        """
        pass

    @abstractmethod
    def broadcast(
        self,
        msg: MsgT,
        src: int = 0,
    ) -> MsgT:
        """
        Broadcast a model to all nodes in the communication layer.

        Args:
            model: The model to broadcast
            src: The source node index (default is 0)
        Returns:
            The broadcasted model on all nodes
        """
        pass

    @abstractmethod
    def aggregate(
        self,
        msg: MsgT,
        communicate_params: bool = True,
        compute_mean: bool = True,
        **kwargs,
    ) -> MsgT:
        """
        Aggregate an object (model or tensor) across nodes.

        Args:
            obj: The object to aggregate (model or tensor)
            communicate_params: Whether to communicate parameters (True) or gradients (False)
            compute_mean: Whether to compute the mean (default is True)
            **kwargs: Additional protocol-specific arguments (e.g., batch_samples for weighted aggregation)
        Returns:
            The aggregated object
        """
        pass

    @abstractmethod
    def send(
        self,
        msg: MsgT,
        dst: int,
        communicate_params: bool = True,
    ) -> MsgT:
        """
        Send model parameters or tensor to a given destination rank.
        """
        pass

    @abstractmethod
    def receive(
        self,
        msg: MsgT,
        src: int,
        communicate_params: bool = True,
    ) -> MsgT:
        """
        Receive model parameters or tensor from a given source rank.
        """
        pass

    @abstractmethod
    def collect(
        self,
        msg: Union[nn.Module, torch.Tensor, float, int],
        communicate_params: bool = True,
    ) -> list:
        """
        Gather an object from all ranks and return a list of (rank, data).
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up any resources used by the communicator.
        """
        pass
