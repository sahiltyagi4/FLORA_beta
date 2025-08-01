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

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import numpy as np
import warnings
import re
from typeguard import typechecked


class MetricDirection(str, Enum):
    """Optimization direction for metrics (higher/lower is better)."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    NEUTRAL = "neutral"


class MetricGroup(str, Enum):
    """Metric categories for table organization."""

    LOSS_METRICS = "Loss & Error"
    PERFORMANCE_METRICS = "Performance"
    GRADIENT_METRICS = "Gradients"
    TIMING_METRICS = "Timing"
    DATASET_METRICS = "Dataset"
    REPORTING_METADATA = "Reporting Metadata"
    OTHER = "Other"


@dataclass
class MetricFormatRule:
    """Formatting rules for FL metrics.

    Rules are matched by regex pattern in order.
    """

    regex: str
    precision: int = 3
    units: str = ""
    optimization_goal: MetricDirection = MetricDirection.NEUTRAL
    format_as_integer: bool = False
    emoji: str = ""
    group: str = MetricGroup.OTHER
    group_order: int = 50
    description: str = "Default formatting"

    show_sum: bool = False
    show_mean: bool = True
    show_std: bool = True
    show_min: bool = True
    show_max: bool = True
    show_median: bool = True
    show_cv: bool = True


DEFAULT_FORMAT_RULES = [
    MetricFormatRule(
        regex=r"time",
        precision=4,
        units="s",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":stopwatch:",
        group=MetricGroup.TIMING_METRICS,
        group_order=40,
        description="Timing metrics",
        show_sum=False,
    ),
    MetricFormatRule(
        regex=r"(loss|error|mse|mae|rmse)",
        precision=4,
        units="",
        optimization_goal=MetricDirection.MINIMIZE,
        emoji=":chart_decreasing:",
        group=MetricGroup.LOSS_METRICS,
        group_order=10,
        description="Loss and error metrics",
        show_sum=False,
    ),
    MetricFormatRule(
        regex=r"(accuracy|precision|recall|f1)",
        precision=4,
        units="",
        optimization_goal=MetricDirection.MAXIMIZE,
        emoji=":dart:",
        group=MetricGroup.PERFORMANCE_METRICS,
        group_order=20,
        description="Performance metrics",
        show_sum=False,
    ),
    MetricFormatRule(
        regex=r"(count|num_)",
        precision=0,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        format_as_integer=True,
        emoji=":package:",
        group=MetricGroup.DATASET_METRICS,
        group_order=5,
        description="Count metrics",
        show_sum=True,  # Sum makes sense for counts
    ),
    MetricFormatRule(
        regex=r"grad",
        precision=4,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        emoji=":bar_chart:",
        group=MetricGroup.GRADIENT_METRICS,
        group_order=30,
        description="Gradient metrics",
        show_sum=False,
    ),
    MetricFormatRule(
        regex=r"(batch_idx|epoch_idx|global_step|round_idx)",
        precision=0,
        units="",
        optimization_goal=MetricDirection.NEUTRAL,
        format_as_integer=True,
        emoji=":spiral_calendar:",
        group=MetricGroup.REPORTING_METADATA,
        group_order=90,  # Show last
        description="Training progression coordinates",
        show_sum=False,  # Averaging makes more sense than summing
        show_mean=True,
        show_std=True,
        show_min=True,
        show_max=True,
        show_median=True,
        show_cv=True,
    ),
]


@typechecked
class MetricFormatter:
    """FL metrics formatter with validation and caching."""

    def __init__(self, rules: List[MetricFormatRule] = DEFAULT_FORMAT_RULES):
        """Initialize formatter.

        Args:
            rules: Custom formatting rules. Uses defaults if None.
        """
        if not rules:
            raise ValueError("Rules list cannot be empty")

        for i, rule in enumerate(rules):
            if not rule.regex.strip():
                raise ValueError(f"Rule {i} has invalid regex: {rule.regex}")

        self.rules = rules
        self._rule_cache = {}
        self._validation_stats = {"cache_hits": 0, "cache_misses": 0, "fallbacks": 0}

    def find_rule(self, metric_name: str) -> MetricFormatRule:
        """Find matching formatting rule for metric."""
        if not metric_name.strip():
            metric_name = "empty_metric"
            self._validation_stats["fallbacks"] += 1

        if metric_name in self._rule_cache:
            self._validation_stats["cache_hits"] += 1
            return self._rule_cache[metric_name]

        self._validation_stats["cache_misses"] += 1

        try:
            for rule in self.rules:
                try:
                    if re.search(rule.regex, metric_name, re.IGNORECASE):
                        self._rule_cache[metric_name] = rule
                        return rule
                except re.error as e:
                    warnings.warn(f"Invalid regex in rule for {rule.regex}: {e}")
                    continue
        except Exception as e:
            warnings.warn(f"Error searching rules for metric '{metric_name}': {e}")

        default_rule = MetricFormatRule(
            regex=r".*",
            group_order=80,
            description=f"Auto-generated default for '{metric_name}'",
        )
        self._rule_cache[metric_name] = default_rule
        self._validation_stats["fallbacks"] += 1
        return default_rule

    def format(self, metric_name: str, value: float) -> str:
        """Format metric value according to rules."""
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "+∞" if value > 0 else "-∞"

        try:
            rule = self.find_rule(metric_name)

            if rule.format_as_integer:
                try:
                    formatted_value = f"{int(round(value)):,}{rule.units}"
                except (ValueError, OverflowError):
                    formatted_value = f"{value:.0f}{rule.units}"
            else:
                precision = max(0, min(10, rule.precision))
                formatted_value = f"{value:.{precision}f}{rule.units}"

            return formatted_value

        except Exception as e:
            raise ValueError(
                f"Failed to format metric '{metric_name}' with value {value}: {e}"
            ) from e

    def group_metric_names(self, metric_names: List[str]) -> Dict[str, List[str]]:
        """Group metrics by category."""
        if not metric_names:
            return {}

        validated_names = []
        for i, name in enumerate(metric_names):
            if not name.strip():
                name = f"empty_metric_{i}"
                self._validation_stats["fallbacks"] += 1
            validated_names.append(name.strip())

        try:
            groups = defaultdict(list)
            group_orders = {}

            for metric_name in validated_names:
                rule = self.find_rule(metric_name)
                group = rule.group.value
                groups[group].append(metric_name)
                if group not in group_orders:
                    group_orders[group] = rule.group_order

            for group in groups:
                groups[group].sort()

            sorted_groups = dict(
                sorted(groups.items(), key=lambda x: group_orders[x[0]])
            )
            return sorted_groups

        except Exception as e:
            raise RuntimeError(f"Failed to group metrics: {e}") from e

    def get_applicable_stats(self, metric_name: str) -> Dict[str, bool]:
        """Get which statistics to show for a metric.

        Args:
            metric_name: Metric name

        Returns:
            Dict of stat name -> show flag
        """
        if not metric_name.strip():
            metric_name = "empty_metric"
            self._validation_stats["fallbacks"] += 1

        try:
            rule = self.find_rule(metric_name)
            return {
                "sum": rule.show_sum,
                "mean": rule.show_mean,
                "std": rule.show_std,
                "min": rule.show_min,
                "max": rule.show_max,
                "median": rule.show_median,
                "cv": rule.show_cv,
            }
        except Exception as e:
            raise RuntimeError(
                f"Failed to get applicable stats for metric '{metric_name}': {e}"
            ) from e

    def get_delta_color(
        self, metric_name: str, delta: float, threshold: float = 1e-4
    ) -> str:
        """Get color for metric change.

        Args:
            metric_name: Metric name
            delta: Change value (new - old)
            threshold: Minimum significant change

        Returns:
            Rich color string
        """
        if np.isnan(delta) or np.isinf(delta):
            return "dim white"

        if threshold <= 0:
            threshold = 1e-4

        try:
            if abs(delta) < threshold:
                return "dim white"

            rule = self.find_rule(metric_name)
            goal = rule.optimization_goal

            if goal == MetricDirection.NEUTRAL:
                return "dim white"

            is_good_change = (delta < 0 and goal == MetricDirection.MINIMIZE) or (
                delta > 0 and goal == MetricDirection.MAXIMIZE
            )
            return "bright_green" if is_good_change else "bright_red"

        except Exception as e:
            raise RuntimeError(
                f"Failed to get delta color for metric '{metric_name}': {e}"
            ) from e
