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

import logging
import time

import ray
import rich.repr
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from rich import box
from rich.table import Table

from . import utils
from .mixins import SetupMixin
from .topology.BaseTopology import Topology


@rich.repr.auto
class Engine(SetupMixin):
    """
    Core orchestration engine for federated learning experiments.

    - Orchestrate distributed federated learning experiments
    - Manage node lifecycle and resource allocation
    - Collect, analyze, and report metrics and statistics.

    Integration:
    - Coordinates with Topology definitions for network structure
    - Manages Node actors running Algorithm implementations
    - Instantiated through Hydra configuration
    """

    def __init__(
        self,
        flora_cfg: DictConfig,
    ):
        super().__init__()
        utils.log_sep(f"{self.__class__.__name__} Init", color="blue")

        self.flora_cfg: DictConfig = flora_cfg
        self.hydra_cfg: HydraConf = HydraConfig.get()

        self.topology: Topology = instantiate(self.flora_cfg.topology)
        self.global_rounds: int = flora_cfg.global_rounds

    def _setup(self):
        utils.log_sep("FLORA Engine Setup", color="blue")

        # Initialize Ray with more verbose logging and explicit namespace
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=True,  # NOTE: when false, Task and Actor logs are not copied to the driver stdout.
            # logging_level=logging.INFO,  # TODO: Tie this to Hydra's logging level
            # namespace="federated_learning",
        )

        self.topology.setup(
            comm_cfg=self.flora_cfg.comm,
            algo_cfg=self.flora_cfg.algo,
            model_cfg=self.flora_cfg.model,
            data_cfg=self.flora_cfg.data,
            log_dir=self.hydra_cfg.runtime.output_dir,
        )

    def start(self):
        """
        NOTE: stuffed everything in here for now, plan to refactor into smaller generalized methods.
        """
        try:
            # ----------------------------------------------------------------
            utils.log_sep("FLORA Engine Start", color="blue")

            summaries = []
            for round_idx in range(self.global_rounds):
                # utils.log_sep(f"Round {round_idx + 1}/{self.global_rounds}")
                print()
                print(f"# Round {round_idx + 1}/{self.global_rounds}", flush=True)
                _t_start_round = time.time()

                results_futures = []
                for node in self.topology:
                    future = node.round_exec.remote(round_idx)
                    results_futures.append(future)

                results = ray.get(results_futures)

                # ---
                _t_round = time.time() - _t_start_round

                _ct_total = len(results)

                # ---
                summaries.append(
                    {
                        "round_idx": round_idx,
                        "duration": _t_round,
                        "total_count": _ct_total,
                    }
                )
                print(f"# Round Complete | {summaries[-1]}", flush=True)
                time.sleep(3)  # Give time for logs to flush

            utils.log_sep("FL Rounds End", color="blue")

            # ----------------------------------------------------------------

            table = Table(
                title="Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )

            for key in summaries[0].keys():
                table.add_column(
                    key.replace("_", " ").title(),
                    justify="center",
                    style="cyan",
                )

            for metrics in summaries:
                table.add_row(*[str(metrics[key]) for key in metrics.keys()])

            utils.console.print(table)

        finally:
            print("Engine shutting down")
            ray.shutdown()
