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

import os
from typing import Any, Dict, List

import rich.repr
import torch
from omegaconf import DictConfig

from ..Node import Node
from .BaseTopology import Topology
from .CentralizedTopology import CentralizedTopology

# ======================================================================================


@rich.repr.auto
class MultiGroupTopology(Topology):
    """
    Multi-group federated learning topology for cross-institutional FL.

    Coordinates multiple independent federated learning groups,
    where each group runs as a standard centralized topology internally.
    Group servers can communicate across institutional boundaries.

    Each group maintains its own TorchDistributed communication for local training,
    while group servers use gRPC to coordinate global aggregation.

    The topology composes multiple CentralizedTopology instances.
    """

    def __init__(self, groups: List[CentralizedTopology]):
        """
        Initialize with a list of group configurations.

        Args:
            groups: Each group defines how many clients participate in that
                   institution's local federated learning setup.

        Raises:
            ValueError: If no groups provided
        """
        super().__init__()

        if not groups:
            raise ValueError("At least one group must be specified")

        self.groups: List[CentralizedTopology] = groups

    def create_nodes(
        self,
        local_comm_cfg: DictConfig,
        global_comm_cfg: DictConfig | None,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        log_dir: str,
        node_rayopts: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Node]:
        """
        Create nodes for all groups and configure cross-group communication.

        Each group gets its own TorchDistributed setup for local training.
        Group servers additionally get gRPC configuration for coordinating
        with other institutions.

        Args:
            local_comm_cfg: TorchDistributed config for intra-group communication
            global_comm_cfg: gRPC config for inter-group coordination
            algo_cfg: Algorithm configuration
            model_cfg: Model configuration
            data_cfg: Data configuration
            log_dir: Base logging directory

        Returns:
            Flattened list of all nodes across all groups
        """
        assert global_comm_cfg is not None, (
            "global_comm_cfg required for inter-group communication"
        )

        # Prevent GPU over-allocation when multiple groups share infrastructure
        total_nodes = sum(group.num_clients + 1 for group in self.groups)
        if torch.cuda.is_available():
            node_rayopts.setdefault("num_gpus", 1.0 / total_nodes)

        all_nodes: List[Node] = []

        for group_idx, group_topology in enumerate(self.groups):
            # Inject group's global rank into gRPC configuration
            group_global_comm_cfg = DictConfig(
                {
                    **global_comm_cfg,
                    "rank": group_idx,  # Group index becomes global rank
                    "world_size": len(self.groups),
                }
            )

            # Let CentralizedTopology handle the heavy lifting
            group_nodes = group_topology.create_nodes(
                local_comm_cfg=local_comm_cfg,
                global_comm_cfg=group_global_comm_cfg,
                algo_cfg=algo_cfg,
                model_cfg=model_cfg,
                data_cfg=data_cfg,
                log_dir=os.path.join(log_dir, f"Group{group_idx}"),
                node_rayopts=node_rayopts,
                **kwargs,
            )

            all_nodes.extend(group_nodes)

        return all_nodes
