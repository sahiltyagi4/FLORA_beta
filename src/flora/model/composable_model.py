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

import rich.repr
import torch
import torch.nn as nn

# ======================================================================================


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for model heads (output layers).
    
    Defines interface for final processing layers.
    Converts backbone features to task-specific outputs (classification, regression, etc.).
    """

    @property
    @abstractmethod
    def in_channels(self) -> int:
        """Number of input channels expected by this head."""
        pass


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for feature extraction backbones.
    
    Defines interface for feature extraction components that process raw inputs
    into meaningful representations for heads to consume.
    """

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of output channels produced by this backbone."""
        pass


@rich.repr.auto
class ComposableModel(nn.Module):
    """
    Modular neural network with swappable backbone and head components.

    Combines feature extraction (backbone) with task-specific processing (head)
    while enforcing channel compatibility between components.

    Enables flexible model architectures through component composition rather
    than monolithic model definitions.

    Example:
    ```python
    backbone = SimpleCNNBackbone(...)
    head = ClassificationHead(...)
    model = ComposableModel(backbone=backbone, head=head)
    ```
    """

    def __init__(
        self,
        backbone: BaseBackbone,
        head: BaseHead,
    ):
        """
        Initialize composable model with backbone and head components.

        Args:
            backbone: Feature extraction component
            head: Task-specific output component

        Raises:
            ValueError: If backbone output channels don't match head input channels
        """
        super().__init__()

        # Validate component compatibility
        if backbone.out_channels != head.in_channels:
            raise ValueError(
                f"Channel mismatch: backbone outputs {backbone.out_channels} channels "
                f"but head expects {head.in_channels} channels"
            )

        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone then head.

        Args:
            x: Input tensor

        Returns:
            Task-specific output tensor
        """
        features = self.backbone(x)
        output = self.head(features)
        return output
