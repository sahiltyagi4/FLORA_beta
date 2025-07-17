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
from enum import Enum
from typing import Union, TypeVar, Dict

import rich.repr
import torch
import torch.nn as nn

from ..mixins import SetupMixin


# ======================================================================================


class ReductionType(str, Enum):
    """Aggregation reduction operations."""

    SUM = "SUM"
    MEAN = "MEAN"
    MAX = "MAX"


class Communicator(SetupMixin, ABC):
    """
    Abstract communication interface for federated learning message transport.

    Provides protocol-agnostic message passing operations (broadcast, aggregate)
    across different communication backends. Algorithms handle aggregation logic;
    communicators handle pure transport.
    """

    MsgT = TypeVar("MsgT", nn.Module, torch.Tensor, Dict[str, torch.Tensor])

    @abstractmethod
    def broadcast(
        self,
        msg: MsgT,
        src: int = 0,
    ) -> MsgT:
        """Broadcast message from source to all ranks."""
        pass

    @abstractmethod
    def aggregate(
        self,
        msg: MsgT,
        reduction: ReductionType,
    ) -> MsgT:
        """Aggregate message across all ranks with specified reduction."""
        pass

    # @abstractmethod
    # def send(
    #     self,
    #     msg: MsgT,
    #     dst: int,
    # ) -> MsgT:
    #     """
    #     Send model parameters or tensor to a given destination rank.
    #     """
    #     pass

    # @abstractmethod
    # def receive(
    #     self,
    #     msg: MsgT,
    #     src: int,
    # ) -> MsgT:
    #     """
    #     Receive model parameters or tensor from a given source rank.
    #     """
    #     pass

    # @abstractmethod
    # def collect(
    #     self,
    #     msg: Union[nn.Module, torch.Tensor, float, int],
    # ) -> list:
    #     """
    #     Gather an object from all ranks and return a list of (rank, data).
    #     """
    #     pass

    @abstractmethod
    def close(self):
        """Clean up communication resources."""
        pass

    def get_msg_info(self, msg: MsgT) -> dict:
        info = {}

        # Extract tensor metadata for logging
        if isinstance(msg, torch.Tensor):
            tensor = msg
        elif isinstance(msg, nn.Module):
            tensor = next(msg.parameters(), None)
        elif isinstance(msg, dict):
            tensor = next(
                (t for t in msg.values() if isinstance(t, torch.Tensor)), None
            )
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

        info["dtype"] = str(tensor.dtype)
        info["device"] = str(tensor.device)

        return info
