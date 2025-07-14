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


class SetupMixin(ABC):
    """
    Mixin for components that require setup before use.

    Provides consistent setup state management and template method pattern.
    Automatically initializes state without requiring super().__init__() calls.
    """

    @property
    def is_ready(self) -> bool:
        """True if component is ready for operations."""
        if not hasattr(self, "_setup_complete"):
            self._setup_complete = False
        return self._setup_complete

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Initialize component. Template method pattern."""
        if not hasattr(self, "_setup_complete"):
            self._setup_complete = False

        if self._setup_complete:
            print(f"NOTE: {self.__class__.__name__} is already set up. Skipping setup.")
            return

        self._setup(*args, **kwargs)
        self._setup_complete = True

    @abstractmethod
    def _setup(self, *args: Any, **kwargs: Any) -> None:
        """Implementation-specific setup logic."""
        pass
