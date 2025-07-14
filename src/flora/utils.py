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

import copy
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

import rich.repr
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from torch import nn
from torch.utils.data import DataLoader

from .algorithms import utils as alg_utils
from .algorithms.BaseAlgorithm import Algorithm

console = Console()


def setup_rich_logging(level=logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        omit_repeated_times=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )

    formatter = logging.Formatter(fmt="[%(name)s] %(message)s")
    rich_handler.setFormatter(formatter)

    root_logger.addHandler(rich_handler)
    root_logger.setLevel(level)


def log_sep(title: str = "", style: str = "═", color: str = "yellow") -> None:
    """full-width separator for different logging stages"""
    if title:
        console.print()
        console.print(
            Rule(f"[bold {color}]{title}[/bold {color}]", style=color, characters=style)
        )
    else:
        console.print(Rule(style="dim", characters=style))
