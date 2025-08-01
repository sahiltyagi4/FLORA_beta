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

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torchvision import ops

from .composable_model import BaseBackbone


class SimpleCNNBackbone(BaseBackbone):
    """
    Configurable CNN backbone for feature extraction.

    Builds a series of convolutional blocks (conv + norm + activation + pool)
    followed by a 1x1 output projection layer. Supports flexible architecture
    configuration through layer lists and function parameters.

    Common use cases:
    - Image classification feature extraction
    - Transfer learning base models
    - Federated learning backbone components
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        kernel_sizes: Union[int, List[int]] = 3,
        paddings: Optional[Union[int, List[int]]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        pool_layer: Optional[Callable[..., nn.Module]] = nn.MaxPool2d,
    ):
        """
        Initialize configurable CNN backbone.

        Args:
            in_channels: Input channels (1=grayscale, 3=RGB)
            hidden_channels: Output channels for each convolutional block
            out_channels: Final output channels (must match head input)
            kernel_sizes: Convolutional kernel sizes (int or list per layer)
            paddings: Padding values (int, list, or None for auto)
            norm_layer: Normalization layer factory (default: BatchNorm2d)
            activation_layer: Activation function factory (default: ReLU)
            pool_layer: Pooling layer factory (default: MaxPool2d)
        """
        super().__init__()

        # Normalize kernel_sizes and paddings to lists if they are integers
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(hidden_channels)
        if isinstance(paddings, int):
            paddings = [paddings] * len(hidden_channels)

        # Create convolutional and pooling stages
        self.stages = nn.ModuleList()

        __hid_prev_chans = in_channels
        for i, __hid_out_chans in enumerate(hidden_channels):
            # Create a sequential block with Conv2dNormActivation and pooling
            stage = nn.Sequential()

            # Add Conv2dNormActivation
            stage.append(
                ops.Conv2dNormActivation(
                    in_channels=__hid_prev_chans,
                    out_channels=__hid_out_chans,
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i] if paddings is not None else None,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # Add pooling if specified
            if pool_layer is not None:
                stage.append(
                    pool_layer(2)
                )  # NOTE: Assuming fixed 2x2 pooling (for now)

            self.stages.append(stage)
            __hid_prev_chans = __hid_out_chans

        # Final output projection layer
        self.out_proj = nn.Conv2d(
            in_channels=__hid_prev_chans,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 conv to adjust channels
        )

        # Store output channels for head compatibility check
        self.__out_channels = out_channels

    @property
    def out_channels(self) -> int:
        """Output channels produced by this backbone."""
        return self.__out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Feature tensor with out_channels dimensions
        """
        for stage in self.stages:
            x = stage(x)
        x = self.out_proj(x)
        return x
