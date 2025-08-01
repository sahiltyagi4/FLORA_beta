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

from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn

from . import grpc_pb2


def tensordict_to_proto(
    tensordict: Dict[str, torch.Tensor],
) -> grpc_pb2.TensorDict:
    """
    Convert tensor dictionary to protobuf format for gRPC transmission.

    Serializes PyTorch tensors to byte format with metadata for exact reconstruction.
    Includes device information and data type preservation.

    Args:
        tensordict: Dictionary mapping parameter names to tensor values

    Returns:
        TensorDict protobuf message ready for gRPC transmission
    """
    entries = []

    for key, tensor in tensordict.items():
        # Store original device before CPU conversion
        original_device = str(tensor.device)

        # Convert to CPU for serialization (required for protobuf)
        tensor_cpu = tensor.cpu()

        # Serialize tensor data to bytes for transmission
        tensor_bytes = tensor_cpu.numpy().tobytes()

        entry = grpc_pb2.TensorEntry(
            key=key,
            data=tensor_bytes,
            shape=list(tensor_cpu.shape),
            dtype=str(tensor_cpu.dtype),
            device=original_device,
            data_size=len(tensor_bytes),
        )
        entries.append(entry)

    return grpc_pb2.TensorDict(entries=entries)


def proto_to_tensordict(
    proto_tensordict: grpc_pb2.TensorDict,
) -> Dict[str, torch.Tensor]:
    """
    Convert protobuf tensor dictionary back to PyTorch tensors.

    Deserializes byte data back to PyTorch tensors with original shapes,
    data types, and device placement preserved.

    Args:
        proto_tensordict: TensorDict protobuf message from gRPC

    Returns:
        Dictionary mapping parameter names to reconstructed tensors

    Raises:
        ValueError: If data size mismatch or unsupported dtype
    """
    tensordict = {}
    for entry in proto_tensordict.entries:
        # Validate data size
        if len(entry.data) != entry.data_size:
            raise ValueError(
                f"Data size mismatch for tensor {entry.key}: expected {entry.data_size}, got {len(entry.data)}"
            )

        # Dtype mapping: string -> numpy_dtype
        dtype_mapping = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.bool": np.bool_,
        }

        if entry.dtype not in dtype_mapping:
            supported_dtypes = list(dtype_mapping.keys())
            raise ValueError(
                f"Unsupported dtype: {entry.dtype}. Supported: {supported_dtypes}"
            )

        numpy_dtype = dtype_mapping[entry.dtype]

        # Reconstruct tensor from serialized bytes
        numpy_array = np.frombuffer(entry.data, dtype=numpy_dtype)
        numpy_array = numpy_array.reshape(tuple(entry.shape))

        # Create tensor and restore to original device
        # Note: .copy() needed because np.frombuffer creates read-only arrays
        tensor = torch.from_numpy(numpy_array.copy()).to(entry.device)

        tensordict[entry.key] = tensor

    return tensordict


def get_msg_info(
    msg: Union[torch.Tensor, nn.Module, Dict[str, Any], Any],
) -> Dict[str, Any]:
    """
    Extract metadata from message for logging and debugging.

    Provides structured information about tensors, models, or dictionaries
    for communication operation logging and troubleshooting.

    Args:
        msg: Message to analyze (tensor, model, or dict)

    Returns:
        Dictionary with type, shape, device, and size information

    Raises:
        TypeError: If message type is not supported
    """
    info: Dict[str, Any] = {
        "type": type(msg).__name__,
    }

    # Extract tensor metadata for logging
    if isinstance(msg, torch.Tensor):
        info.update(
            {
                "shape": list(msg.shape),
                "numel": msg.numel(),
                "dtype": str(msg.dtype),
                "device": str(msg.device),
            }
        )
    elif isinstance(msg, nn.Module):
        info["params"] = sum(p.numel() for p in msg.parameters())
        tensor = next(msg.parameters(), None)
        if tensor is not None:
            info.update(
                {
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
            )
    elif isinstance(msg, dict):
        keys = list(msg.keys())
        # Limit keys output to prevent log overflow
        if len(keys) <= 5:
            info["keys"] = keys
        else:
            info["keys"] = keys[:3] + ["...", f"({len(keys)} total)"]

        # Add tensor-specific information if dictionary contains tensors
        tensor_values = [v for v in msg.values() if isinstance(v, torch.Tensor)]
        if tensor_values:
            info["tensors"] = len(tensor_values)
            info["total_params"] = sum(t.numel() for t in tensor_values)

            # Use first tensor for dtype/device info
            first_tensor = tensor_values[0]
            info.update(
                {
                    "dtype": str(first_tensor.dtype),
                    "device": str(first_tensor.device),
                }
            )
    else:
        raise TypeError(f"Unsupported message type: {type(msg)}")

    return info
