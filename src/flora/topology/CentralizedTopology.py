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
from typing import Any, Dict, List, Optional

import time

import ray
import rich.repr
import torch
from omegaconf import DictConfig

from .. import utils
from ..Node import Node
from .BaseTopology import Topology

# ======================================================================================


@rich.repr.auto
class CentralizedTopology(Topology):
    """
    Template for centralized federated learning topology with aggregator-trainer architecture.

    - One aggregator node (rank 0) collects and combines model updates
    - Multiple trainer nodes (ranks 1+) perform local training
    - All communication flows through the aggregator
    """

    def __init__(self, num_clients: int, init_delay: float = 1.0):
        """
        Initialize centralized topology.

        Args:
            num_clients (int): Number of client nodes (server node is added automatically)
            init_delay (float): Simulated delay in seconds between node initializations (default: 0.5s)
        """
        super().__init__()
        self.num_clients: int = num_clients
        self.num_nodes: int = num_clients + 1  # 1 server + N clients
        self.init_delay: float = init_delay

    def create_nodes(
        self,
        comm_cfg: DictConfig,
        algo_cfg: DictConfig,
        model_cfg: DictConfig,
        data_cfg: DictConfig,
        log_dir: str,
    ) -> List[Node]:
        """
        Create nodes for centralized topology.

        In centralized topology:
        - Rank 0: Aggregator (no training data, coordinates aggregation)
        - Ranks 1+: Trainers (training data, perform local training)

        Returns:
            List of configured nodes
        """
        print(
            f"[TOPOLOGY-CREATE] Creating {self.num_clients} clients + 1 server = {self.num_nodes} total nodes",
            flush=True,
        )

        nodes: List[Node] = []

        # ----------------------------------------------------------------
        # INIT ALL NODES

        for rank in range(self.num_nodes):
            # Configure Ray actor options
            node_rayopts: Dict[str, Any] = {}

            # Request GPU resources if available, but don't assign specific devices here
            if torch.cuda.is_available():
                node_rayopts["num_gpus"] = 1.0 / self.num_nodes

            if rank == 0:
                node_id = f"N{rank}-SERVER"
                # Create node-specific data config for server (no training data)
                node_data_cfg = data_cfg.copy()
                # node_data_cfg = data_cfg
                node_data_cfg.train = None
            else:
                node_id = f"N{rank}-Client"  # Purposefully using different casing for log readability
                node_data_cfg = data_cfg

            node = Node.options(**node_rayopts).remote(
                id=node_id,
                comm_cfg=comm_cfg,
                model_cfg=model_cfg,
                algo_cfg=algo_cfg,
                data_cfg=node_data_cfg,
                local_rank=rank,
                world_size=self.num_nodes,
                log_dir=os.path.join(log_dir, node_id),
            )

            nodes.append(node)

            # Add simulated delay between node initializations
            time.sleep(self.init_delay)

        # ----------------------------------------------------------------
        # SETUP ALL NODES (PARALLEL)

        # Start all setups simultaneously to avoid TorchDist coordination deadlock
        setup_futures = [node.setup.remote() for node in nodes]
        ray.get(setup_futures)

        return nodes
