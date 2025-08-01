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

from dataclasses import replace
from typing import Any, Dict, List, Optional

import rich.repr
from omegaconf import OmegaConf

from ..communicator import BaseCommunicatorConfig
from ..node import NodeConfig
from .base import BaseTopology

# ======================================================================================


@rich.repr.auto
class CentralizedTopology(BaseTopology):
    """
    Classic federated learning: one server coordinating multiple clients.

    When to use: All participants can communicate directly with a central server.
    For cross-institutional setups, use MultiGroupTopology instead.

    How it works:
    - Server (rank 0) aggregates model updates, never trains locally
    - Clients (ranks 1+) train on local data, send updates to server
    - All communication stays within this single group

    Example config:
    ```yaml
    topology:
      _target_: src.flora.topology.CentralizedTopology
      num_clients: N
      local_comm:
        _target_: src.flora.communicator.TorchDistCommunicator
        backend: "gloo"
    ```

    Example with N clients (N+1 total nodes):
    - 0.0 (server): Receives updates from clients → averages → sends back.
    - 0.1, 0.2, ... 0.N (clients): Train locally → send updates → receive new model.
    """

    def __init__(
        self,
        num_clients: int,
        local_comm: BaseCommunicatorConfig,
        overrides: Optional[Dict[int, NodeConfig]] = None,
        **kwargs: Any,
    ):
        """
        Set up server-client FL topology.

        Args:
            num_clients: How many client nodes to create (server added automatically)
            local_comm: Communication config for server-client coordination
            overrides: Custom settings per rank (0=server, 1+=clients).
                      Example: {0: {"device_hint": "cpu"}, 1: {"device_hint": "cuda:X"}}
        """
        super().__init__(**kwargs)
        self.num_clients: int = num_clients
        self.local_comm_cfg: BaseCommunicatorConfig = local_comm
        self.overrides: Dict[int, NodeConfig] = overrides or {}

    def _create_node_configs(self) -> List[NodeConfig]:
        """
        Create server and client node configurations.

        Server gets rank 0, clients get ranks 1, 2, 3, etc.
        Each node gets communication settings and can have custom overrides.

        Returns all nodes ready for the Engine to launch as Ray actors.
        """
        world_size: int = self.num_clients + 1
        node_configs: List[NodeConfig] = []

        for rank in range(world_size):
            # Create local comm config with rank-specific parameters
            # Use structured config to preserve type information
            local_comm_cfg: BaseCommunicatorConfig = OmegaConf.structured(
                self.local_comm_cfg
            )
            local_comm_cfg.rank = rank
            local_comm_cfg.world_size = world_size

            node_cfg = NodeConfig(
                name=f"Node0.{rank}",
                local_comm=local_comm_cfg,
                global_comm=None,
            )

            # Apply overrides using dataclass replace to preserve types
            override_cfg = self.overrides.get(rank, {})
            node_cfg = replace(node_cfg, **override_cfg)

            node_configs.append(node_cfg)

        return node_configs
