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

from abc import ABC


class LifecycleHooks(ABC):
    """
    Optional lifecycle hooks for federated learning algorithm customization.

    Provides standardized extension points during FL execution for algorithm-specific
    logic like custom metrics, logging, or state management.

    All hooks are optional - override only what your algorithm needs.
    Default implementations are no-ops.

    Hook execution order:
    ```
    _round_start()
      _train_epoch_start()
        _train_batch_start()
        _train_batch_end()
      _train_epoch_end()
      _eval_epoch_start()
        _eval_batch_start()
        _eval_batch_end()
      _eval_epoch_end()
    _round_end()
    ```
    """

    def _round_start(self) -> None:
        """Called at start of each FL round.
        
        **Override for round-level setup** like custom metrics initialization or algorithm state reset.
        """
        pass

    def _round_end(self) -> None:
        """Called at end of each FL round.
        
        **Override for round-level cleanup** like custom metrics finalization or algorithm state updates.
        """
        pass

    def _train_epoch_start(self) -> None:
        """Called at start of each training epoch.
        
        **Override for epoch-level training setup** like learning rate scheduling or epoch-specific initialization.
        """
        pass

    def _train_epoch_end(self) -> None:
        """Called at end of each training epoch.
        
        **Override for epoch-level training cleanup** like custom metrics aggregation or epoch-specific updates.
        """
        pass

    def _train_batch_start(self) -> None:
        """Called before processing each training batch.
        
        **Override for batch-level training setup** like custom data preprocessing or batch-specific initialization.
        """
        pass

    def _train_batch_end(self) -> None:
        """Called after processing each training batch.
        
        **Override for batch-level training cleanup** like custom metrics logging or batch-specific updates.
        """
        pass

    def _eval_epoch_start(self) -> None:
        """Called at start of each evaluation epoch.
        
        **Override for epoch-level evaluation setup** like evaluation-specific initialization or mode switching.
        """
        pass

    def _eval_epoch_end(self) -> None:
        """Called at end of each evaluation epoch.
        
        **Override for epoch-level evaluation cleanup** like custom evaluation metrics aggregation.
        """
        pass

    def _eval_batch_start(self) -> None:
        """Called before processing each evaluation batch.
        
        **Override for batch-level evaluation setup** like custom evaluation preprocessing.
        """
        pass

    def _eval_batch_end(self) -> None:
        """Called after processing each evaluation batch.
        
        **Override for batch-level evaluation cleanup** like custom evaluation metrics logging.
        """
        pass
