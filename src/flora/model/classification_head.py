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

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .composable_model import BaseHead

# ======================================================================================


class ClassificationHead(BaseHead):
    """
    Classification head for converting spatial features to class logits.

    Processes backbone features through optional transformation layers,
    applies global pooling to create fixed-size vectors, then projects
    to final class predictions.

    Architecture: features → [transforms] → global_pool → classifier → logits
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        feature_layers: List[int] = [],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize classification head.

        Args:
            in_channels: Input channels from backbone
            num_classes: Number of classification classes
            feature_layers: Optional intermediate layer sizes (1x1 convs)
            norm_layer: Normalization factory (applied after convs)
            activation_layer: Activation function factory
            dropout_rate: Dropout probability for feature layers
        """
        super().__init__()
        self.__in_channels = in_channels

        # Build feature layers
        layers = []
        prev_channels = in_channels

        for feature_size in feature_layers:
            # Add 1x1 convolution
            layers.append(nn.Conv2d(prev_channels, feature_size, kernel_size=1))

            # Add normalization if specified
            if norm_layer is not None:
                layers.append(norm_layer(feature_size))

            # Add activation if specified
            if activation_layer is not None:
                layers.append(activation_layer(inplace=True))

            # Add dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))

            prev_channels = feature_size

        self.feature_transform = nn.Sequential(*layers) if layers else nn.Identity()

        # Global average pooling (size-agnostic)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification layer (linear instead of conv)
        self.classifier = nn.Linear(prev_channels, num_classes)

    @property
    def in_channels(self) -> int:
        """Input channels expected by this head."""
        return self.__in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert spatial features to class logits.

        Args:
            x: Feature tensor from backbone (B, C, H, W)

        Returns:
            Class logits (B, num_classes)
        """
        x = self.feature_transform(x)
        x = self.global_pool(x)  # (B, C, H, W) -> (B, C, 1, 1)
        x = x.flatten(1)  # (B, C, 1, 1) -> (B, C)
        x = self.classifier(x)  # (B, C) -> (B, num_classes)
        return x
