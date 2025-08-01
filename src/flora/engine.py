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

import json
import os
import pickle
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import ray
import rich.repr
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import MISSING
from rich.pretty import pprint
from tqdm.auto import tqdm

from . import utils
from .algorithm import BaseAlgorithmConfig
from .data import DataModuleConfig
from .model import ModelConfig
from .node import Node, NodeConfig
from .topology import BaseTopology, BaseTopologyConfig
from .utils import ExperimentResultsDisplay, MetricFormatter, RequiredSetup, print

LOG_FLUSH_DELAY = 2.0


@dataclass
class RayConfig:
    """Ray cluster configuration for distributed federated learning."""

    # ─────────────────────────────────────────
    # Cluster Connection & Resource Allocation
    # ─────────────────────────────────────────

    # Cluster connection (null = auto-detect local cluster)
    # Use "ray://host:port" for remote clusters, "local" to force local
    address: Optional[str] = None

    # Resource allocation - CRITICAL for proper GPU/CPU distribution
    # null = auto-detect based on hardware, explicit numbers override detection
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None

    # Custom resources: {"accelerator_type": "V100", "high_memory": 2}
    resources: Optional[Dict[str, Any]] = None

    # ─────────────────────────────────────────
    # Memory & Performance
    # ─────────────────────────────────────────

    # Object store memory for large model sharing (null = 30% of system memory)
    object_store_memory: Optional[int] = None

    # ─────────────────────────────────────────
    # Monitoring & Development
    # ─────────────────────────────────────────

    # Essential for FL: forward all distributed node logs to main process
    log_to_driver: bool = True

    # Ray dashboard (null = auto-start if dependencies available)
    include_dashboard: Optional[bool] = None
    dashboard_host: str = "127.0.0.1"  # Use "0.0.0.0" for external access
    dashboard_port: Optional[int] = None  # null = auto-find port starting from 8265

    # Development convenience - allow multiple ray.init() calls without error
    ignore_reinit_error: bool = True

    # ─────────────────────────────────────────
    # Advanced Configuration
    # ─────────────────────────────────────────

    # Experiment isolation (null = anonymous namespace)
    namespace: Optional[str] = None

    # Runtime environment for distributed workers (empty = inherit from main process)
    # Example: {"pip": ["torch==1.12.0"], "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}}
    runtime_env: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.resources is None:
            self.resources = {}
        if self.runtime_env is None:
            self.runtime_env = {}


@dataclass
class EngineConfig:
    """Main configuration for FLORA federated learning experiments."""

    # Required experiment parameters
    global_rounds: int = MISSING

    # Optional experiment parameters
    overwrite: bool = False

    # Component configurations - these will be resolved by Hydra defaults
    topology: BaseTopologyConfig = MISSING
    algorithm: BaseAlgorithmConfig = MISSING
    model: ModelConfig = MISSING
    datamodule: DataModuleConfig = MISSING

    # Infrastructure configurations
    ray: RayConfig = field(default_factory=RayConfig)


# Register the config with Hydra's ConfigStore for structured configs
cs = ConfigStore.instance()
cs.store(name="base_config", node=EngineConfig)


@rich.repr.auto
class Engine(RequiredSetup):
    """
    Main engine for federated learning experiments.

    Coordinates distributed Ray actors (nodes) to run FL algorithms across different topologies.
    Handles experiment setup, execution, and results collection with automatic
    GPU allocation and output management.

    Use this as the main entry point for running FL experiments with Hydra configurations.
    See working examples in the conf/ directory.
    """

    def __init__(
        self,
        cfg: EngineConfig,
    ):
        """
        Initialize the federated learning experiment engine.

        Args:
            cfg: Complete FLORA configuration including topology, algorithm,
                 model, and datamodule specifications
        """
        super().__init__()
        utils.print_rule()

        self.cfg: EngineConfig = cfg
        self.hydra_cfg: HydraConf = HydraConfig.get()

        self.topology: BaseTopology = instantiate(cfg.topology, _recursive_=False)
        self.global_rounds: int = cfg.global_rounds
        self.overwrite: bool = cfg.overwrite

        # Convert Ray configuration to dict for ray.init()
        self.ray_cfg: RayConfig = cfg.ray

        self.output_dir: str = self.hydra_cfg.runtime.output_dir
        self.engine_dir: str = os.path.join(self.output_dir, "engine")
        self.results_dir: str = os.path.join(self.engine_dir, "node_results")

        self._metric_formatter: MetricFormatter = MetricFormatter()
        self._results_display: ExperimentResultsDisplay = ExperimentResultsDisplay()

        self._ray_actor_refs: List[Node] = []

    def _setup_output_directories(self) -> None:
        """
        Create and validate output directories for experiment data.

        Creates engine/ and node_results/ directories under Hydra's output path.
        Issues warnings if conflicting experiment files already exist unless overwrite=True.
        """
        # Check for pre-existing files that could overwrite results (ignore Hydra standard files)
        if os.path.exists(self.output_dir):
            hydra_standard_files = {".hydra", "main.log", ".gitignore"}
            existing_files = [
                f
                for f in os.listdir(self.output_dir)
                if not f.startswith(".") and f not in hydra_standard_files
            ]
            if existing_files:
                if self.overwrite:
                    warnings.warn(
                        f"Output directory contains existing files: {self.output_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"Proceeding with overwrite=True - previous experiment results may be overwritten.",
                        UserWarning,
                    )
                else:
                    raise RuntimeError(
                        f"Output directory contains existing files: {self.output_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"This could overwrite previous experiment results. "
                        f"Use a fresh Hydra output directory, clean the existing one, or set overwrite=true."
                    )

        # Create engine directory
        os.makedirs(self.engine_dir, exist_ok=True)

        # Check if engine directory is not empty (indicates conflicting experiment)
        if os.path.exists(self.engine_dir):
            existing_files = [
                f for f in os.listdir(self.engine_dir) if not f.startswith(".")
            ]
            if existing_files:
                if self.overwrite:
                    warnings.warn(
                        f"Engine directory is not empty: {self.engine_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"Proceeding with overwrite=True - conflicting experiment files may be overwritten.",
                        UserWarning,
                    )
                else:
                    raise RuntimeError(
                        f"Engine directory is not empty: {self.engine_dir}\n"
                        f"Found: {existing_files[:5]}{'...' if len(existing_files) > 5 else ''}\n"
                        f"This indicates a conflicting experiment setup. Set overwrite=true to proceed anyway."
                    )

        print(f"Created engine directory: {self.engine_dir}")

    def _setup(self) -> None:
        """
        Initialize Ray cluster and launch distributed nodes.

        Sets up the distributed infrastructure including:
        - Output directory validation
        - Ray cluster initialization
        - Smart GPU allocation (fractional for single-node, full for multi-node)
        - Node actor creation and setup
        """
        utils.print_rule()

        # Setup directories first
        self._setup_output_directories()

        # Initialize Ray cluster
        ray.init(**self.ray_cfg)

        # Smart GPU allocation: detect single-node vs multi-node scenarios
        ray_available_resources = ray.available_resources()
        print("ray.available_resources()")
        pprint(ray_available_resources)

        # Check if single node
        ray_nodes = ray.nodes()
        print("ray.nodes()")
        pprint(ray_nodes)

        # Save Ray cluster information to engine directory
        ray_resources_path = os.path.join(
            self.engine_dir, "ray_available_resources.json"
        )
        ray_nodes_path = os.path.join(self.engine_dir, "ray_nodes.json")

        with open(ray_resources_path, "w") as f:
            json.dump(ray_available_resources, f, indent=2, default=str)
            print(f"Saved Ray resources info to: {ray_resources_path}")

        with open(ray_nodes_path, "w") as f:
            json.dump(ray_nodes, f, indent=2, default=str)
            print(f"Saved Ray nodes info to: {ray_nodes_path}")

        # print(f"Saved Ray cluster info to: {self.engine_dir}")

        ray_nodes_alive = [node for node in ray_nodes if node["Alive"]]
        is_single_node = len(ray_nodes_alive) == 1

        available_gpus = ray_available_resources.get("GPU", 0)
        total_actors = len(list(self.topology))

        # Determine GPU allocation strategy
        use_fractional_gpu = (
            is_single_node and total_actors > available_gpus and available_gpus > 0
        )

        print(f"Launching {total_actors} Actors")
        node_config: NodeConfig
        for node_config in self.topology:
            # Set log directory
            node_config.log_dir_base = (
                node_config.log_dir_base or self.hydra_cfg.runtime.output_dir
            )

            # Configure GPU allocation
            ray_actor_options = asdict(node_config.ray_actor_options)
            if ray_actor_options.get("num_gpus") is None and available_gpus > 0:
                if use_fractional_gpu:
                    # Single-node with GPU shortage: use fractional allocation
                    ray_actor_options["num_gpus"] = available_gpus / total_actors
                else:
                    # Multi-node or sufficient GPUs: request 1 GPU per actor
                    ray_actor_options["num_gpus"] = 1

            pprint(node_config)

            node_actor = Node.options(**ray_actor_options).remote(
                **asdict(node_config),  # type: ignore - Ray's remote() typing doesn't understand dataclass unpacking
                algorithm=self.cfg.algorithm,
                model=self.cfg.model,
                datamodule=self.cfg.datamodule,
            )
            self._ray_actor_refs.append(node_actor)

        print(f"Calling setup() on {len(self._ray_actor_refs)} Nodes")
        setup_futures = [node.setup.remote() for node in self._ray_actor_refs]
        ray.get(setup_futures)

    def _save_node_results(self, results: List) -> None:
        """
        Save individual node results as pickle files for debugging.

        Creates node_results/ directory and saves each node's results as
        node_000_results.pkl, node_001_results.pkl, etc.
        Non-fatal - logs errors but doesn't crash the experiment.

        Args:
            results: List of result dictionaries from each node
        """
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Saving node results to: {self.results_dir}")

        # Save node results with progress bar
        for node_idx, node_result in tqdm(
            enumerate(results),
            desc="Saving node results",
            unit="file",
            total=len(results),
        ):
            filename = f"node_{node_idx:03d}_results.pkl"
            filepath = os.path.join(self.results_dir, filename)

            with open(filepath, "wb") as f:
                pickle.dump(node_result, f)

        print(
            f":heavy_check_mark: Saved {len(results)} node result files successfully!"
        )

    def run_experiment(self) -> None:
        """
        Run the federated learning experiment.

        Coordinates experiment execution across all nodes:
        - Launches experiment on all Ray actors
        - Waits for completion with progress feedback
        - Saves node results for debugging
        - Displays formatted experiment results
        - Handles Ray cluster shutdown
        """
        try:
            utils.print_rule()
            print(f"Starting Experiment with {len(self._ray_actor_refs)} Nodes")

            experiment_start_time = time.time()

            node_results_futures = []
            for node in self._ray_actor_refs:
                future = node.run_experiment.remote(self.global_rounds)
                node_results_futures.append(future)

            print(
                f"Waiting for {len(node_results_futures)} nodes to complete experiments...",
                flush=True,
            )

            # Wait for all nodes to complete
            results = ray.get(node_results_futures)

            print(
                f":heavy_check_mark: All {len(results)} nodes completed successfully!",
                flush=True,
            )

            # Save node results for debugging
            try:
                self._save_node_results(results)
            except Exception as e:
                warnings.warn(
                    f"Failed to save node results for debugging: {e}. "
                    f"Experiment results will still be displayed normally.",
                    UserWarning,
                )

            if results:
                print("=" * 80)
                print("DEBUG: First node's returned data structure:")
                print("=" * 80)

                print(json.dumps(results[0], indent=2, default=str))
                print("=" * 80)

            experiment_end_time = time.time()
            experiment_duration = experiment_end_time - experiment_start_time

            utils.print_rule()
            time.sleep(
                LOG_FLUSH_DELAY
            )  # Ensure async Ray logs complete before displaying results

            self._results_display.show_experiment_results(
                results,
                experiment_duration,
                self.global_rounds,
                len(self.topology),
            )

        finally:
            print("Shutting down...", flush=True)
            ray.shutdown()
