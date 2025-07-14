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
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        print(f"[COMM-INIT] rank={rank}/{world_size} | group={group_name}")

    def _setup(self):
        print("[COMM-SETUP] no-op")

    def broadcast(
        self,
        msg: Communicator.MsgT,
        src: int = 0,
    ) -> Communicator.MsgT:
        print(f"[COMM-BCAST] src=rank{src} | {type(msg).__name__}")
        return msg

    def aggregate(
        self,
        msg: Communicator.MsgT,
        compute_mean: bool = True,
    ) -> Communicator.MsgT:
        print(f"[COMM-AGG] {type(msg).__name__} | no-op")
        return msg

    def send(
        self,
        msg: Communicator.MsgT,
        dst: int,
    ) -> Communicator.MsgT:
        print(f"[COMM-SEND] dst=rank{dst}")
        return msg

    def receive(
        self,
        msg: Communicator.MsgT,
        src: int,
    ) -> Communicator.MsgT:
        print(f"[COMM-RECV] src=rank{src}")
        return msg

    def collect(
        self,
        msg: Union[nn.Module, torch.Tensor, float, int],
    ) -> list[tuple[int, Communicator.MsgT]]:
        print("[COMM-COLLECT] no-op")
        return [(self.rank, msg)]

    def close(self):
        print("[COMM-CLOSE] cleanup")
