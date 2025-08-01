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

import atexit
import csv
import os
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from .metric_format import MetricFormatter


class MeanAccumulator:
    """Simple mean accumulator."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.update_count = 0

    def update(self, value, weight=1):
        """Update with a new value and optional weight."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.sum += float(value) * weight
        self.count += weight
        self.update_count += 1

    def compute(self):
        """Compute the mean value."""
        if self.count == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.sum / self.count)

    def reset(self):
        """Reset the accumulator."""
        self.sum = 0.0
        self.count = 0
        self.update_count = 0


class SumAccumulator:
    """Simple sum accumulator."""

    def __init__(self):
        self.sum = 0.0
        self.update_count = 0

    def update(self, value, weight=1):
        """Update with a new value and optional weight."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.sum += float(value) * weight
        self.update_count += 1

    def compute(self):
        """Compute the sum value."""
        return torch.tensor(self.sum)

    def reset(self):
        """Reset the accumulator."""
        self.sum = 0.0
        self.update_count = 0


class MetricAggType(str, Enum):
    """Enumeration of metric aggregation strategies."""

    MEAN = "mean"
    SUM = "sum"


class MetricLogger:
    """Mixin for metrics collection and logging.

    Organizes metrics by aggregation context (train/eval/sync) and
    logs to TensorBoard and CSV files.

    Key features:
        - Separate aggregation contexts for different training phases
        - Built-in timing measurement
        - Multiple output formats
        - Context managers and class-level decorators
        - Flexible coordinate extraction (round/epoch/batch/etc.)

    Note: Not thread-safe. Use one instance per process/actor.
    """

    def __init__(
        self,
        log_dir: str,
        global_step_fn: Callable[[], int],
        metadata_fields: Optional[Dict[str, Callable[[], Any]]] = None,
    ):
        """Initialize metrics collection system.

        Creates log_dir if needed. Overwrites existing metric files.
        Call this directly from inheriting classes, not via super().

        Args:
            log_dir: Directory for metric output files (TensorBoard logs, CSV files)
            global_step_fn: Lambda function that returns current global step
                           e.g., lambda: self.global_step
            metadata_fields: Optional dict mapping field names to lambda functions
                           e.g., {"round_idx": lambda: self.round_idx}
        """

        self._metric_log_dir = log_dir
        self._global_step_fn = global_step_fn
        self._metadata_fields = metadata_fields or {}

        self._tb_writer = SummaryWriter(log_dir)
        self._csv_file = open(
            os.path.join(log_dir, "metrics_full.csv"), "w", newline=""
        )
        self._csv_writer = csv.writer(self._csv_file)

        # Build dynamic CSV header based on metadata fields
        header = (
            ["global_step"]
            + list(self._metadata_fields.keys())
            + [
                "agg_ctx",
                "agg_type",
                "agg_count",
                "metric_key",
                "metric_val",
            ]
        )
        self._csv_writer.writerow(header)

        self.current_agg_context: str = "default"
        self._agg_ctx_accumulators: Dict[str, Dict[MetricAggType, defaultdict]] = {}
        self._agg_ctx_csv_writers: Dict[str, csv.writer] = {}
        self._agg_ctx_csv_files: Dict[str, Any] = {}
        self._formatter = MetricFormatter()

        atexit.register(self.close_metrics)

    @contextmanager
    def logging_context(
        self,
        context_key: str,
        *,
        log_duration: bool = True,
        duration_key: str = "time",
        print_progress: bool = True,
    ):
        """Create a metric context for organized logging (context manager only).

        Groups metrics and automatically measures duration. Flushes when exiting.

        Args:
            context_key: Context name for grouping metrics
            log_duration: Whether to automatically log execution duration (default: True)
            duration_key: Custom duration metric name (default: "time")
            print_progress: Print start/end messages (default: True)

        Example:
            # As context manager
            with self.logging_context("training"):
                self.log_metric("loss", 0.5)
                self.log_metric("accuracy", 0.9)

            with self.logging_context("evaluation", log_duration=False, print_progress=False):
                self.log_metric("eval_loss", 0.3)
        """
        # Enter logic - set context
        prev_context = self.current_agg_context
        self.current_agg_context = context_key

        if print_progress:
            print(
                f"[{context_key}] START @ {self.progress_info_str}",
                flush=True,
            )

        # Use log_duration for timing if enabled, otherwise just yield
        duration_context = (
            self.log_duration(
                duration_key,
                print_progress=False,  # We handle our own progress printing
                agg_context=context_key,
            )
            if log_duration
            else None
        )

        try:
            if duration_context:
                with duration_context:
                    yield
            else:
                yield
        finally:
            # Flush metrics to storage
            try:
                flushed_metrics = self.flush_metrics(context_key)
                flushed_metrics = {
                    key: self._formatter.format(key, value)
                    for key, value in flushed_metrics.items()
                }
            except Exception as e:
                warnings.warn(
                    f"Failed to flush metrics for context '{context_key}': {e}"
                )
                flushed_metrics = "<flush failed>"

            # Print progress if enabled
            if print_progress:
                print(
                    f"[{context_key}] END @ {self.progress_info_str} | {flushed_metrics}",
                    flush=True,
                )

            # Restore previous context (CRITICAL - must always succeed)
            self.current_agg_context = prev_context

    @classmethod
    def context(cls, context_key: Optional[str] = None, **kwargs):
        """Create a decorator for metric context switching (class-level decorator syntax).

        Args:
            name: Context name for grouping metrics (defaults to function name)
            **kwargs: Same arguments as logging_context()

        Example:
            @MetricLogger.context()  # Uses function name
            def train_epoch(self):
                self.log_metric("loss", 0.5)

            @MetricLogger.context("custom_training")  # Uses custom name
            def train_epoch(self):
                self.log_metric("loss", 0.5)
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **func_kwargs):
                context_name = context_key or func.__name__
                with self.logging_context(context_name, **kwargs):
                    return func(self, *args, **func_kwargs)

            return wrapper

        return decorator

    @property
    def progress_info_str(self) -> str:
        """Current status line with global step and all metadata."""
        global_step = self._global_step_fn()
        parts = []

        for field_name, field_fn in self._metadata_fields.items():
            try:
                value = field_fn()
                parts.append(f"{field_name}={value}")
            except Exception:
                pass

        return f"global_step={global_step} ({' '.join(parts)})".upper()

    def log_metric(
        self,
        key: str,
        val: float,
        agg_type: MetricAggType = MetricAggType.MEAN,
        agg_context: Optional[str] = None,
    ) -> None:
        """Log a metric value in the specified or current aggregation context.

        Args:
            key: Metric name (e.g., "loss", "accuracy")
            val: Value to log
            agg_type: MEAN for averages, SUM for totals
            agg_context: Optional aggregation context to override current context

        Example:
            self.log_metric("loss", 0.5)
            self.log_metric("samples", 32, MetricAggType.SUM)
            self.log_metric("cross_validation", 0.8, agg_context="test")
        """
        # Use provided context or fall back to current context
        target_agg_context = agg_context or self.current_agg_context

        # Create full metric name by joining metric name with aggregation context
        full_metric_name = f"{target_agg_context}/{key}"

        # Initialize target aggregation context if not exists
        if target_agg_context not in self._agg_ctx_accumulators:
            self._agg_ctx_accumulators[target_agg_context] = {
                MetricAggType.MEAN: defaultdict(MeanAccumulator),
                MetricAggType.SUM: defaultdict(SumAccumulator),
            }

        agg_ctx = self._agg_ctx_accumulators[target_agg_context]
        match agg_type:
            case MetricAggType.MEAN:
                agg_ctx[MetricAggType.MEAN][full_metric_name].update(val)
            case MetricAggType.SUM:
                agg_ctx[MetricAggType.SUM][full_metric_name].update(val)
            case _:
                raise ValueError(f"Unknown aggregation type: {agg_type}")

    @contextmanager
    def log_duration(
        self,
        key: str,
        *,
        print_progress: bool = True,
        agg_context: Optional[str] = None,
    ):
        """Time a code block and log the duration.

        Args:
            key: Name for the timing metric
            print_progress: Print start/end messages (default: False)
            agg_context: Optional aggregation context to override current context

        Example:
            with self.log_duration("training_time"):
                run_training_epoch()

            with self.log_duration("sync_time", print_progress=True):
                perform_synchronization()

            with self.log_duration("validation_time", agg_context="test"):
                run_validation()
        """
        if print_progress:
            print(f"[{key}] START @ {self.progress_info_str}", flush=True)

        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Log duration metric (best effort)
            try:
                self.log_metric(key, duration, agg_context=agg_context)
            except Exception as e:
                warnings.warn(f"Failed to log duration metric '{key}': {e}")

            if print_progress:
                print(
                    f"[{key}] END @ {self.progress_info_str} | duration={duration:.3f}s",
                    flush=True,
                )

    def flush_metrics(self, agg_context: Optional[str] = None) -> Dict[str, float]:
        """Write accumulated metrics and reset counters.

        Usually called automatically by logging_context() and @MetricLogger.context() decorator.
        Call manually only when changing agg_contexts without using these methods.

        Args:
            agg_context: Aggregation context to flush (defaults to current agg_context)

        Returns:
            Dictionary of computed metrics that were flushed

        Note:
            Skips agg_contexts with no metrics. Resets all counters after writing.
        """
        agg_context = agg_context or self.current_agg_context

        # Skip if aggregation context doesn't exist (no metrics were accumulated)
        if agg_context not in self._agg_ctx_accumulators:
            return {}

        metrics = []
        return_dict = {}
        agg_ctx_accumulators = self._agg_ctx_accumulators[agg_context]

        # Process both MEAN and SUM metrics
        for agg_type in [MetricAggType.MEAN, MetricAggType.SUM]:
            for metric_key, metric_accumulator in agg_ctx_accumulators[
                agg_type
            ].items():
                if metric_accumulator.update_count > 0:
                    # Compute value before reset
                    metric_val = metric_accumulator.compute().item()
                    metric_count = metric_accumulator.update_count

                    # Store for writing
                    metrics.append((metric_key, metric_val, agg_type, metric_count))

                    # Store for return
                    return_dict[metric_key] = metric_val

                # Reset accumulator
                metric_accumulator.reset()

        # Write metrics
        self._write_metrics(metrics, agg_context)

        return return_dict

    def _extract_metadata(self) -> Dict[str, int | float]:
        """Extract metadata fields with graceful error handling."""
        metadata: Dict[str, int | float] = {"global_step": self._global_step_fn()}
        for field_name, field_fn in self._metadata_fields.items():
            try:
                metadata[field_name] = field_fn()
            except Exception:
                # Fallback value for failed metadata extraction
                metadata[field_name] = -1
        return metadata

    def _get_context_metric_names(self, agg_context: str) -> set[str]:
        """Get all metric names for a given aggregation context."""
        all_metric_names = set()
        if agg_context in self._agg_ctx_accumulators:
            for agg_type in [MetricAggType.MEAN, MetricAggType.SUM]:
                all_metric_names.update(
                    self._agg_ctx_accumulators[agg_context][agg_type].keys()
                )
        return all_metric_names

    def _write_metrics(
        self,
        metrics: List[tuple[str, float, MetricAggType, int]],
        agg_context: str,
    ) -> None:
        """Write metrics to TensorBoard and CSV files.

        Args:
            metrics: List of (metric_name, metric_val, agg_type, agg_count) tuples
            agg_context: Aggregation context for these metrics
        """
        # Skip if no metrics to write
        if not metrics:
            return

        metadata = self._extract_metadata()

        # Write to TensorBoard and long-format CSV
        for name, metric_val, agg_type, agg_count in metrics:
            # Log to TensorBoard (with error handling)
            try:
                self._tb_writer.add_scalar(
                    name, metric_val, global_step=metadata["global_step"]
                )
            except (OSError, RuntimeError) as e:
                warnings.warn(f"Failed to write metric '{name}' to TensorBoard: {e}")

            # Write to long format CSV using field order from header
            try:
                row = [
                    metadata["global_step"],
                    *[metadata[field] for field in self._metadata_fields.keys()],
                    agg_context,
                    agg_type,
                    agg_count,
                    name,
                    metric_val,
                ]
                self._csv_writer.writerow(row)
            except (OSError, IOError) as e:
                warnings.warn(f"Failed to write metric '{name}' to main CSV: {e}")

        # Flush main CSV (with error handling)
        try:
            self._csv_file.flush()
        except (OSError, IOError) as e:
            warnings.warn(f"Failed to flush main CSV file: {e}")

        # Handle context-specific CSV (direct writing for crash safety)
        if metrics:  # Only proceed if we have metrics
            try:
                # Get all metric names once (used for both header and row writing)
                all_metric_names = self._get_context_metric_names(agg_context)
                sorted_metric_names = sorted(all_metric_names)

                # Initialize context CSV writer if needed
                if agg_context not in self._agg_ctx_csv_writers:
                    csv_path = os.path.join(
                        self._metric_log_dir, f"metrics_{agg_context}.csv"
                    )
                    csv_file = open(csv_path, "w", newline="")
                    csv_writer = csv.writer(csv_file)

                    # Write header with all known metrics for this context
                    header = list(metadata.keys()) + sorted_metric_names
                    csv_writer.writerow(header)

                    self._agg_ctx_csv_files[agg_context] = csv_file
                    self._agg_ctx_csv_writers[agg_context] = csv_writer

                # Build row: metadata + metric values in sorted order
                metric_values = {name: val for name, val, _, _ in metrics}
                row = list(metadata.values()) + [
                    metric_values.get(metric_name, "")
                    for metric_name in sorted_metric_names
                ]

                self._agg_ctx_csv_writers[agg_context].writerow(row)
                self._agg_ctx_csv_files[agg_context].flush()
            except (OSError, IOError) as e:
                warnings.warn(f"Failed to write context CSV for '{agg_context}': {e}")

    def get_experiment_data(self) -> Dict[str, Any]:
        """Extract experiment timeline data for display purposes.

        Returns:
            Dictionary containing all context data organized by agg_context.
            Format: {agg_context: [list of metric rows with metadata]}
        """
        experiment_data = {}

        for agg_context in self._agg_ctx_csv_files.keys():
            csv_path = os.path.join(self._metric_log_dir, f"metrics_{agg_context}.csv")
            try:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    experiment_data[agg_context] = df.to_dict("records")
            except Exception as e:
                warnings.warn(
                    f"Failed to read experiment data for context '{agg_context}': {e}"
                )
                experiment_data[agg_context] = []

        return experiment_data

    def close_metrics(self) -> None:
        """Close TensorBoard writer and CSV files.

        Note:
            Called automatically on exit. Safe to call multiple times.
        """

        # Close TensorBoard writer
        self._tb_writer.close()

        # Close main CSV file
        self._csv_file.close()

        # Close all context-specific CSV files
        for csv_file in self._agg_ctx_csv_files.values():
            csv_file.close()
        self._agg_ctx_csv_files.clear()
        self._agg_ctx_csv_writers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_metrics()
