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

"""
Hydra:
- built on top of the OmegaConf library

References:

Structured Config:

- https://hydra.cc/docs/tutorials/structured_config/intro/
- https://hydra.cc/docs/tutorials/structured_config/minimal_example/
- https://hydra.cc/docs/tutorials/structured_config/hierarchical_static_config/

Instantiating Objects:

- https://hydra.cc/docs/advanced/instantiate_objects/overview/
- https://hydra.cc/docs/advanced/instantiate_objects/structured_config/

"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class CommunicationProtocol:
    """Base configuration for communicators"""

    _target_: str = MISSING


@dataclass
class GRPCConfig(CommunicationProtocol):
    """Configuration for gRPC communicator"""

    _target_: str = "src.flora.communicator.GRPCCommunicator"
    # Where this node's gRPC server binds to receive connections
    bind_address: str = "localhost"
    bind_port: int = MISSING

    # Optional connection parameters
    timeout_ms: int = 5000
    max_retries: int = 3


@dataclass
class MPIConfig(CommunicationProtocol):
    """Configuration for MPI protocol"""

    _target_: str = "src.flora.communicator.MPICommunicator"


@dataclass
class CommunicationProtocols:
    """Node communication protocol configurations

    Contains configuration for each protocol supported by a node.
    Each field represents a different protocol type with its specific configuration.

    TODO:
    - Should a node support multiple instances of the same protocol?
    """

    grpc: Optional[GRPCConfig] = None
    mpi: Optional[MPIConfig] = None
    # NOTE: Future protocols can be added as fields here


# class MatchOperator(Enum):
#     """Operators for field matching in connection rules"""

#     EQUALS = "eq"
#     NOT_EQUALS = "ne"
#     CONTAINS = "contains"
#     REGEX = "regex"
#     IN = "in"
#     NOT_IN = "not_in"
#     GREATER_THAN = "gt"
#     LESS_THAN = "lt"


@dataclass
class FieldMatcher:
    """Advanced field matching with multiple operators"""

    field_path: str = MISSING  # Support nested fields like "metadata.region"
    # operator: MatchOperator = MatchOperator.EQUALS
    value: Any = MISSING
    case_sensitive: bool = True


@dataclass
class ConnectionConfig:
    """
    Generalized rule-based connection specification using dictionary matching

    NOTE:
    - Potential alternative to explicitly defining edges.

    TODO:
    - Generalized matching rules (e.g., regex, wildcard matching?)
    - Node exclusion logic?
    """

    protocol: str = MISSING

    # Node selection using flexible dictionary matching
    # Each key-value pair specifies a field name and required value
    filter: Dict[str, Any] = field(default_factory=dict)

    # Connection modifiers
    exclude_self: bool = True
    # bidirectional: bool = False
    max_connections: Optional[int] = None

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeConfig:
    """Node configuration"""

    id: str = MISSING
    host: str = "localhost"

    # TODO: Decide whether to define connections here or separate from nodes (as edges)
    peers: List[ConnectionConfig] = field(default_factory=list)

    protocols: CommunicationProtocols = field(default_factory=CommunicationProtocols)

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeConfig:
    """
    Communication edge between nodes

    TODO:
    - Figure out if replacing this with a Node-centric rule-based system is better (e.g., ConnectionRule).
    - Defining protocol here vs. in NodeConfig? Advantages/Disadvantages?
    """

    src: str = MISSING
    tgt: str = MISSING
    protocol: str = MISSING

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class Edge:
    """Runtime representation of an edge in the topology"""

    def __init__(
        self, src: str, tgt: str, protocol: str, metadata: Dict[str, Any] = dict()
    ):
        self.src = src
        self.tgt = tgt
        self.protocol = protocol
        self.metadata = metadata


@dataclass
class FLTopologyConfig:
    nodes: Dict[str, NodeConfig] = field(default_factory=dict)
    edges: List[EdgeConfig] = field(default_factory=list)

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FLAlgorithmConfig:
    """Configuration for federated learning algorithms"""

    # Dot-path to the class to instantiate
    _target_: str = MISSING
    # Generic required fields for all algorithms
    comm_freq: int = MISSING
    epochs: int = MISSING


@dataclass
class FLConfig:
    """Main federated learning configuration"""

    topology: FLTopologyConfig = MISSING
    algorithm: FLAlgorithmConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=FLConfig)
