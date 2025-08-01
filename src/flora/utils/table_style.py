"""
Styling and theming for FLORA experiment displays.

Centralized colors, emojis, table configurations, and column definitions.
"""

from dataclasses import dataclass
from typing import Dict, Any
from rich import box
from typeguard import typechecked


@dataclass(frozen=True)
class ColorScheme:
    """Centralized color definitions for all display components."""

    # Primary colors for different data types
    round_color: str = "bright_blue"
    epoch_color: str = "bright_magenta"
    batch_color: str = "bright_yellow"
    step_color: str = "bright_green"

    # Table styling colors
    header_primary: str = "bold bright_blue"
    header_secondary: str = "bold bright_white"
    header_tertiary: str = "bold bright_cyan"

    # Text colors
    metric_name: str = "bright_white"
    metric_emoji: str = "bold bright_green"
    group_header: str = "bold bright_blue"
    group_count: str = "dim"

    # Status colors
    success: str = "bold bright_green"
    warning: str = "bold bright_yellow"
    error: str = "bold bright_red"
    info: str = "bright_cyan"

    # Data colors
    data_primary: str = "white"
    data_secondary: str = "dim white"
    data_highlight: str = "bold white"

    # Change indicators
    positive_change: str = "italic bright_green"
    negative_change: str = "italic bright_red"


@dataclass(frozen=True)
class EmojiScheme:
    """Centralized emoji definitions for consistent iconography."""

    # Table types
    experiment_summary: str = ":clipboard:"
    progression: str = ":chart_with_upwards_trend:"
    node_statistics: str = ":busts_in_silhouette:"

    # Coordinate levels
    round: str = ":arrows_clockwise:"
    epoch: str = ":hourglass_not_done:"
    batch: str = ":package:"
    step: str = ":footprints:"

    # Metrics and data
    metric: str = ":bar_chart:"
    sum: str = ":heavy_plus_sign:"
    mean: str = ":bar_chart:"
    std: str = ":straight_ruler:"
    min: str = ":red_triangle_pointed_down:"
    max: str = ":red_triangle_pointed_up:"
    median: str = ":chart_with_upwards_trend:"
    cv: str = ":chart_with_upwards_trend:"
    nodes: str = ":busts_in_silhouette:"

    # Status indicators
    success: str = ":white_check_mark:"
    warning: str = ":warning:"
    error: str = ":cross_mark:"
    info: str = ":information:"

    # General
    value: str = ":clipboard:"
    duration: str = ":stopwatch:"
    rounds: str = ":arrows_counterclockwise:"


@dataclass(frozen=True)
class TableConfiguration:
    """Standard table configuration settings."""

    # Table styling
    box_style: box.Box = box.ROUNDED
    show_header: bool = True

    # Title styling
    title_style: str = "bold bright_white"
    title_justify: str = "left"

    # Caption styling
    caption_style: str = "italic dim"
    caption_justify: str = "right"

    # Column styling
    justify_left: str = "left"
    justify_center: str = "center"
    justify_right: str = "right"
    vertical_middle: str = "middle"


@dataclass(frozen=True)
class ColumnDefinitions:
    """Standardized column definitions for consistent table structure."""

    # Standard column configurations (icon, text, style, justification)
    metric: tuple = (EmojiScheme.metric, "Metric", ColorScheme.header_tertiary, "left")

    # Coordinate columns
    round: tuple = (EmojiScheme.round, "Round", ColorScheme.round_color, "center")
    epoch: tuple = (EmojiScheme.epoch, "Epoch", ColorScheme.epoch_color, "center")
    batch: tuple = (EmojiScheme.batch, "Batch", ColorScheme.batch_color, "center")
    step: tuple = (EmojiScheme.step, "Step", ColorScheme.step_color, "center")

    # Statistics columns
    nodes: tuple = (EmojiScheme.nodes, "Nodes", ColorScheme.data_secondary, "center")
    sum: tuple = (EmojiScheme.sum, "Sum", ColorScheme.data_primary, "right")
    mean: tuple = (EmojiScheme.mean, "Mean", ColorScheme.data_primary, "right")
    std: tuple = (EmojiScheme.std, "Std", ColorScheme.data_primary, "right")
    min: tuple = (EmojiScheme.min, "Min", ColorScheme.data_primary, "right")
    max: tuple = (EmojiScheme.max, "Max", ColorScheme.data_primary, "right")
    median: tuple = (EmojiScheme.median, "Median", ColorScheme.data_primary, "right")
    cv: tuple = (EmojiScheme.cv, "CV", ColorScheme.data_primary, "right")

    # Summary columns
    value: tuple = (EmojiScheme.value, "Value", ColorScheme.success, "center")


class TableStyleManager:
    """Manager for table styling and configuration."""

    def __init__(self):
        """Initialize style manager."""
        self.colors = ColorScheme()
        self.emojis = EmojiScheme()
        self.config = TableConfiguration()
        self.columns = ColumnDefinitions()

    @typechecked
    def get_table_config(
        self, title: str, caption: str, title_emoji: str
    ) -> Dict[str, Any]:
        """Get table configuration."""
        # Validate title
        if not title.strip():
            raise ValueError("Title must be a non-empty string")

        return {
            "title": f"{title_emoji} {title}".strip(),
            "title_style": self.config.title_style,
            "title_justify": self.config.title_justify,
            "caption": caption,
            "caption_style": self.config.caption_style,
            "caption_justify": self.config.caption_justify,
            "box": self.config.box_style,
            "show_header": self.config.show_header,
        }

    @typechecked
    def get_column_config(self, column_type: str) -> Dict[str, Any]:
        """Get column configuration."""
        # Validate column type
        if not column_type.strip():
            raise ValueError("Column type must be a non-empty string")

        # Map column types to definitions
        available_column_types = {
            "metric": self.columns.metric,
            "round": self.columns.round,
            "epoch": self.columns.epoch,
            "batch": self.columns.batch,
            "step": self.columns.step,
            "nodes": self.columns.nodes,
            "sum": self.columns.sum,
            "mean": self.columns.mean,
            "std": self.columns.std,
            "min": self.columns.min,
            "max": self.columns.max,
            "median": self.columns.median,
            "cv": self.columns.cv,
            "value": self.columns.value,
        }

        if column_type not in available_column_types:
            available_types = list(available_column_types.keys())
            raise ValueError(
                f"Unknown column type: '{column_type}'. Available: {available_types}"
            )

        try:
            column_def = available_column_types[column_type]
            if not isinstance(column_def, tuple) or len(column_def) != 4:
                raise ValueError(
                    f"Invalid column definition for '{column_type}': {column_def}"
                )

            emoji, text, style, justify = column_def

            return {
                "header": f"{emoji} {text}".strip(),
                "justify": justify,
                "style": f"bold {style}" if justify in ["left", "center"] else style,
                "vertical": self.config.vertical_middle,
            }

        except Exception as e:
            return {
                "header": f"❓ {column_type}",
                "justify": "left",
                "style": "white",
                "vertical": self.config.vertical_middle,
            }

    @typechecked
    def format_coordinate_label(
        self, level_name: str, value: int, is_boundary: bool = False
    ) -> str:
        """Format coordinate labels."""
        try:
            # Handle empty level name
            if not level_name.strip():
                level_name = "Unknown"

            # Convert to 1-based for display
            display_value = value + 1

            # Get color for this coordinate level
            color_map = {
                "round": self.colors.round_color,
                "epoch": self.colors.epoch_color,
                "batch": self.colors.batch_color,
                "step": self.colors.step_color,
            }

            color = color_map.get(level_name.lower(), self.colors.info)
            base_label = f"{level_name.title()} {display_value}"

            # Apply boundary formatting if needed
            if is_boundary:
                return f"[bold underline {color}]{base_label}[/bold underline {color}]"
            else:
                return f"[{color}]{base_label}[/{color}]"

        except Exception as e:
            # Provide fallback label instead of failing
            return f"[{self.colors.info}]Step {value + 1}[/{self.colors.info}]"

    @typechecked
    def format_metric_name(self, metric_name: str, emoji: str) -> str:
        """Format metric names."""
        if not metric_name.strip():
            metric_name = "Unknown Metric"

        return f"[{self.colors.metric_emoji}]{emoji}[/{self.colors.metric_emoji}] [{self.colors.metric_name}]{metric_name}[/{self.colors.metric_name}]"

    @typechecked
    def format_group_header(self, group_name: str, count: int) -> str:
        """Format group headers."""
        if not group_name.strip():
            group_name = "Unknown Group"

        if count < 0:
            count = 0

        return f"[{self.colors.group_header}]{group_name}[/{self.colors.group_header}] [{self.colors.group_count}]({count})[/{self.colors.group_count}]"

    def get_header_style(self, level: str = "primary") -> str:
        """Get header style for different hierarchy levels."""
        style_map = {
            "primary": self.colors.header_primary,
            "secondary": self.colors.header_secondary,
            "tertiary": self.colors.header_tertiary,
        }
        return style_map.get(level, self.colors.header_primary)


# Global style manager instance
STYLE_MANAGER = TableStyleManager()


# Convenience functions for easy access
@typechecked
def get_table_config(title: str, caption: str, title_emoji: str) -> Dict[str, Any]:
    """Get standardized table configuration."""
    return STYLE_MANAGER.get_table_config(title, caption, title_emoji)


@typechecked
def get_column_config(column_type: str) -> Dict[str, Any]:
    """Get standardized column configuration."""
    return STYLE_MANAGER.get_column_config(column_type)


@typechecked
def format_coordinate_label(
    level_name: str, value: int, is_boundary: bool = False
) -> str:
    """Format coordinate labels with consistent styling."""
    return STYLE_MANAGER.format_coordinate_label(level_name, value, is_boundary)


@typechecked
def format_metric_name(metric_name: str, emoji: str) -> str:
    """Format metric names with consistent styling."""
    return STYLE_MANAGER.format_metric_name(metric_name, emoji)


@typechecked
def format_group_header(group_name: str, count: int) -> str:
    """Format group headers with consistent styling."""
    return STYLE_MANAGER.format_group_header(group_name, count)


def get_colors() -> ColorScheme:
    """Get the current color scheme."""
    return STYLE_MANAGER.colors


def get_emojis() -> EmojiScheme:
    """Get the current emoji scheme."""
    return STYLE_MANAGER.emojis
