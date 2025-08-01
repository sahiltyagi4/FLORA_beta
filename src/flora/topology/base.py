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
from typing import Any, List, Optional

import rich.repr
from rich.pretty import pprint

from .. import utils
from ..node import NodeConfig
from ..utils import print

# ======================================================================================


@rich.repr.auto
class BaseTopology(ABC):
    """
    Base class for federated learning network topologies.

    Defines how nodes are arranged and communicate in distributed FL experiments.
    Concrete implementations include CentralizedTopology and MultiGroupTopology.

    Quick decision guide:
    - Use CentralizedTopology: Single-site FL (all nodes can talk directly)
    - Use MultiGroupTopology: Multi-site FL (hospitals, institutions, etc.)
    - Extend BaseTopology: Custom communication patterns (advanced users)

    How it works:
    - Subclasses implement create_node_configs() to define network structure
    - Returns NodeConfig objects that the Engine launches as Ray actors
    - Provides iteration interface for easy access to all nodes
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize topology base class.

        Args:
            **kwargs: Topology-specific parameters passed to subclasses
        """
        utils.print_rule()

        # Lazy-initialized node configurations list
        self.__node_configs: Optional[List[NodeConfig]] = None

    @property
    def node_configs(self) -> List[NodeConfig]:
        """
        Get all node configurations for this topology.

        Lazily calls _create_node_configs() on first access.
        """
        if self.__node_configs is None:
            print("Lazy-initializing node configurations...")
            # Call the protected method to create node configurations
            self.__node_configs = self._create_node_configs()
            # Log the created node configurations (only on creation)
            # pprint(self.__node_configs)

        if not self.__node_configs:
            raise ValueError(
                "_create_node_configs() must return at least one node configuration"
            )

        return self.__node_configs

    @abstractmethod
    def _create_node_configs(self) -> List[NodeConfig]:
        """
        Create node configurations defining this topology's network structure.

        Protected method - called internally by node_configs property.
        External code should use the node_configs property, not call this directly.

        Subclasses must implement this to define:
        - How many nodes and their roles (server, client, etc.)
        - Communication patterns between nodes
        - Node naming conventions
        - Device assignments and resource requirements

        Returns:
            List of NodeConfig objects ready for Engine to launch as Ray actors
        """
        pass

    def __len__(self) -> int:
        """
        Get the number of nodes in this topology.
        """
        return len(self.node_configs)

    def __getitem__(self, index: int) -> NodeConfig:
        """
        Get a node configuration by index.
        """
        return self.node_configs[index]

    def __iter__(self):
        """
        Iterate over all node configurations in this topology.
        """
        return iter(self.node_configs)
