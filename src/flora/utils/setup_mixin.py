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
from typing import Any


class RequiredSetup(ABC):
    """
    Template Method pattern for deferred component initialization.

    Provides consistent setup/teardown lifecycle for FLORA components that need
    runtime initialization (device placement, network connections, etc.).

    Pattern enforces clean separation:
    - External callers use `setup()` (public interface)
    - Classes override `_setup()` (private implementation)
    - Built-in duplicate setup protection with state tracking

    Usage pattern:
    ```python
    class MyComponent(SetupMixin):
        def _setup(self):
            # Runtime initialization logic here
            pass

    # External usage
    component = MyComponent()
    component.setup()  # Calls _setup() internally
    component.setup()  # Subsequent calls ignored
    ```
    """

    @property
    def is_ready(self) -> bool:
        """True if component has completed setup and is ready for operations."""
        if not hasattr(self, "_setup_complete"):
            self._setup_complete = False
        return self._setup_complete

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize component with duplicate setup protection.

        Template method that calls _setup() if not already initialized.
        Subsequent calls are ignored with a notification message.

        Args:
            *args: Positional arguments passed to _setup()
            **kwargs: Keyword arguments passed to _setup()
        """
        if not hasattr(self, "_setup_complete"):
            self._setup_complete = False

        if self._setup_complete:
            print(f"NOTE: {self.__class__.__name__} is already set up. Skipping setup.")
            return

        self._setup(*args, **kwargs)
        self._setup_complete = True

    @abstractmethod
    def _setup(self, *args: Any, **kwargs: Any) -> None:
        """
        Component-specific setup logic (override required).

        Called exactly once by setup() method. Contains the actual initialization
        logic specific to each component type.

        Args:
            *args: Setup arguments (component-specific)
            **kwargs: Setup keyword arguments (component-specific)
        """
        pass
