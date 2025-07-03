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

from typing import Union

import rich.repr
import torch
import torch.nn as nn

from .BaseCommunicator import Communicator


# ======================================================================================


@rich.repr.auto
class DummyCommunicator(Communicator):
    """
    Mock communicator for development with no-ops.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        group_name: str = "default",
        **kwargs,  # Accept additional parameters for compatibility with other communicators
    ):
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        print(f"DummyCommunicator | rank={rank}/{world_size} | group={group_name}")

    def setup(self):
        print("DummyCommunicator | setup called")

    def broadcast(
        self,
        msg: Communicator.MsgT,
        src: int = 0,
    ) -> Communicator.MsgT:
        print(f"DummyCommunicator | broadcast called from src={src}")
        return msg

    def aggregate(
        self,
        msg: Communicator.MsgT,
        communicate_params: bool = True,
        compute_mean: bool = True,
    ) -> Communicator.MsgT:
        print("DummyCommunicator | aggregate called")
        return msg

    def send(
        self,
        msg: Communicator.MsgT,
        dst: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        print(f"DummyCommunicator | send called to dst={dst}")
        return msg

    def receive(
        self,
        msg: Communicator.MsgT,
        src: int,
        communicate_params: bool = True,
    ) -> Communicator.MsgT:
        print(f"DummyCommunicator | receive called from src={src}")
        return msg

    def collect(
        self,
        msg: Union[nn.Module, torch.Tensor, float, int],
        communicate_params: bool = True,
    ) -> list[tuple[int, Communicator.MsgT]]:
        print("DummyCommunicator | collect called")
        return [(self.rank, msg)]

    def close(self):
        print("DummyCommunicator | close called")
