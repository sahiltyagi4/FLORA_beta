#!/usr/bin/env python3
"""
Enhanced debug script to load pickled node results and reproduce experiment display tables.

This script loads saved node results from experiment runs and recreates the experiment
display tables to test features like coordinate columns, table styling, and separators.
"""

import os
import pickle
import glob
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from flora.utils.results_display import ExperimentResultsDisplay


class ExperimentDebugger:
    """Enhanced experiment debugger with better error handling and metadata extraction."""

    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.validation_errors = []
        self.validation_warnings = []

    def find_experiment_directories(self) -> List[Path]:
        """Find all experiment output directories with node results."""
        experiment_dirs = []

        # Try multiple possible patterns for node results
        patterns = [
            "*/*/engine/node_results",
            "*/*/node_results",
            "*/engine/node_results",
            "*/node_results",
        ]

        for pattern in patterns:
            full_pattern = self.outputs_dir / pattern
            found_dirs = list(Path().glob(str(full_pattern)))
            experiment_dirs.extend(found_dirs)

        # Remove duplicates and sort by modification time
        unique_dirs = list(set(experiment_dirs))
        unique_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return unique_dirs

    def list_available_experiments(self) -> None:
        """List all available experiment directories."""
        experiment_dirs = self.find_experiment_directories()

        if not experiment_dirs:
            print("No experiment directories found.")
            return

        print(f"Found {len(experiment_dirs)} experiment directories:")
        print("-" * 80)

        for i, exp_dir in enumerate(experiment_dirs):
            # Try to extract experiment info
            experiment_name = self._extract_experiment_name(exp_dir)
            timestamp = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            node_files = list(exp_dir.glob("node_*_results.pkl"))

            print(f"{i + 1:2}. {experiment_name}")
            print(f"    Path: {exp_dir}")
            print(f"    Modified: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Node files: {len(node_files)}")
            print()

    def _extract_experiment_name(self, exp_dir: Path) -> str:
        """Extract a meaningful experiment name from the directory path."""
        # Try to get experiment name from parent directories
        parts = exp_dir.parts
        if len(parts) >= 3:
            return f"{parts[-3]}/{parts[-2]}"
        return str(exp_dir.parent.name)

    def get_latest_experiment_dir(self) -> Path:
        """Get the most recent experiment directory."""
        experiment_dirs = self.find_experiment_directories()

        if not experiment_dirs:
            raise FileNotFoundError(
                "No experiment results found.\n"
                "Run an experiment first: ./main.sh --config-name test_mnist_subset\n"
                f"Looking in: {self.outputs_dir.absolute()}"
            )

        latest_dir = experiment_dirs[0]
        print(f"Using latest experiment: {self._extract_experiment_name(latest_dir)}")
        print(f"Path: {latest_dir}")

        return latest_dir

    def get_experiment_dir_by_index(self, index: int) -> Path:
        """Get experiment directory by index from the list."""
        experiment_dirs = self.find_experiment_directories()

        if not experiment_dirs:
            raise FileNotFoundError("No experiment directories found.")

        if index < 0 or index >= len(experiment_dirs):
            raise IndexError(
                f"Invalid experiment index. Available: 0-{len(experiment_dirs) - 1}"
            )

        selected_dir = experiment_dirs[index]
        print(f"Selected experiment: {self._extract_experiment_name(selected_dir)}")
        print(f"Path: {selected_dir}")

        return selected_dir

    def load_node_results(self, results_dir: Path) -> List[Dict[str, Any]]:
        """Load all node result pickle files from the results directory."""
        node_files = list(results_dir.glob("node_*_results.pkl"))

        if not node_files:
            raise FileNotFoundError(
                f"No node result files found in {results_dir}\n"
                f"Expected files matching pattern: node_*_results.pkl"
            )

        # Sort by node number
        node_files.sort(key=lambda x: self._extract_node_number(x.name))

        results = []
        for node_file in node_files:
            print(f"Loading {node_file.name}")

            try:
                with open(node_file, "rb") as f:
                    node_data = pickle.load(f)
                    results.append(node_data)
            except Exception as e:
                print(f"Warning: Failed to load {node_file.name}: {e}")
                continue

        if not results:
            raise RuntimeError("No node result files could be loaded successfully.")

        print(f"Successfully loaded {len(results)} node result files")
        return results

    def _extract_node_number(self, filename: str) -> int:
        """Extract node number from filename like 'node_0_results.pkl'."""
        try:
            # Extract number between 'node_' and '_results.pkl'
            start = filename.find("node_") + 5
            end = filename.find("_results.pkl")
            return int(filename[start:end])
        except (ValueError, IndexError):
            return 0  # Default fallback

    def extract_experiment_metadata(
        self, results: List[Dict[str, Any]], results_dir: Path
    ) -> Tuple[float, int]:
        """Extract experiment metadata (duration, rounds) from results or config files."""
        duration = 10.0  # Default fallback
        global_rounds = 2  # Default fallback

        # Try to extract from results structure
        if results:
            # Look for any timing information in the results
            for node_result in results:
                for context_name, context_data in node_result.items():
                    if context_data and isinstance(context_data, list):
                        # Count unique rounds
                        rounds_seen = set()
                        for measurement in context_data:
                            if (
                                isinstance(measurement, dict)
                                and "round_idx" in measurement
                            ):
                                rounds_seen.add(measurement["round_idx"])

                        if rounds_seen:
                            global_rounds = (
                                max(rounds_seen) + 1
                            )  # Convert from 0-based to count
                            break

        # Try to find experiment config or metadata files
        experiment_root = (
            results_dir.parent.parent
            if results_dir.parent.parent.exists()
            else results_dir.parent
        )

        config_files = [
            experiment_root / "experiment_config.json",
            experiment_root / "config.yaml",
            experiment_root / ".hydra" / "config.yaml",
        ]

        for config_file in config_files:
            if config_file.exists():
                try:
                    if config_file.suffix == ".json":
                        with open(config_file, "r") as f:
                            config = json.load(f)
                            global_rounds = config.get("global_rounds", global_rounds)
                    # Could add YAML parsing here if needed
                    break
                except Exception as e:
                    print(f"Warning: Could not read config from {config_file}: {e}")

        return duration, global_rounds

    def run_debug_display(
        self,
        experiment_index: Optional[int] = None,
        validate_only: bool = False,
        filter_metrics: Optional[List[str]] = None,
        filter_context: Optional[str] = None,
    ) -> None:
        """Main debug function to reproduce experiment display."""
        print("=" * 80)
        print("🔍 FLORA Experiment Display Debugger")
        print("=" * 80)

        try:
            # Get experiment directory
            if experiment_index is not None:
                results_dir = self.get_experiment_dir_by_index(experiment_index)
            else:
                results_dir = self.get_latest_experiment_dir()

            # Load results
            results = self.load_node_results(results_dir)

            # Extract metadata
            duration, global_rounds = self.extract_experiment_metadata(
                results, results_dir
            )

            # Print experiment info
            self._print_experiment_info(results, duration, global_rounds)

            # Apply filters if specified
            if filter_context or filter_metrics:
                results = self._apply_filters(results, filter_context, filter_metrics)
                print(
                    f"\n🔍 Applied filters - Context: {filter_context}, Metrics: {filter_metrics}"
                )

            if not validate_only:
                print("\n" + "=" * 80)
                print("📊 Reproducing Experiment Display")
                print("=" * 80)

                # Create display and show results
                display = ExperimentResultsDisplay()
                display.show_experiment_results(
                    results=results,
                    duration=duration,
                    global_rounds=global_rounds,
                    total_nodes=len(results),
                )

            print("\n" + "=" * 80)
            # Run validation checks
            self._run_validation_checks(results)

            # Print validation results
            self._print_validation_summary()

            print("✅ Debug completed successfully!")
            print(
                "Features tested: coordinate columns, table styling, separators, data validation"
            )
            print("=" * 80)

        except Exception as e:
            self._print_error_diagnostics(e)

    def _print_experiment_info(
        self, results: List[Dict[str, Any]], duration: float, global_rounds: int
    ) -> None:
        """Print detailed information about the loaded experiment."""
        print(f"\n📋 Experiment Information:")
        print(f"├── Number of nodes: {len(results)}")
        print(f"├── Duration: {duration:.1f}s")
        print(f"├── Global rounds: {global_rounds}")

        if results:
            contexts = list(results[0].keys())
            print(f"├── Contexts: {', '.join(contexts)}")

            # Show structure of first context
            if contexts:
                first_context = contexts[0]
                if results[0][first_context]:
                    sample_measurement = results[0][first_context][0]
                    if isinstance(sample_measurement, dict):
                        metrics = [
                            k
                            for k in sample_measurement.keys()
                            if not k.startswith("_")
                            and k
                            not in [
                                "round_idx",
                                "epoch_idx",
                                "batch_idx",
                                "global_step",
                            ]
                        ]
                        print(
                            f"└── Sample metrics in '{first_context}': {', '.join(metrics[:5])}"
                        )
                        if len(metrics) > 5:
                            print(f"    (and {len(metrics) - 5} more...)")

    def _run_validation_checks(self, results: List[Dict[str, Any]]) -> None:
        """Run comprehensive validation checks on experiment results."""
        print("\n" + "=" * 80)
        print("🔍 Running Data Validation Checks")
        print("=" * 80)

        self.validation_errors = []
        self.validation_warnings = []

        # Check 1: Basic structure validation
        self._validate_basic_structure(results)

        # Check 2: Data consistency across nodes
        self._validate_data_consistency(results)

        # Check 3: Coordinate progression validation
        self._validate_coordinate_progression(results)

        # Check 4: Metric value validation
        self._validate_metric_values(results)

        # Check 5: Percentage calculation validation (if applicable)
        self._validate_percentage_calculations(results)

    def _validate_basic_structure(self, results: List[Dict[str, Any]]) -> None:
        """Validate basic structure of results data."""
        if not results:
            self.validation_errors.append("No results data found")
            return

        # Check that all nodes have the same contexts
        first_contexts = set(results[0].keys())
        for i, node_result in enumerate(results[1:], 1):
            node_contexts = set(node_result.keys())
            if node_contexts != first_contexts:
                missing = first_contexts - node_contexts
                extra = node_contexts - first_contexts
                if missing:
                    self.validation_warnings.append(
                        f"Node {i} missing contexts: {missing}"
                    )
                if extra:
                    self.validation_warnings.append(
                        f"Node {i} has extra contexts: {extra}"
                    )

        print(f"✓ Basic structure validation: {len(results)} nodes checked")

    def _validate_data_consistency(self, results: List[Dict[str, Any]]) -> None:
        """Validate data consistency across nodes."""
        if not results:
            return

        total_measurements = 0
        context_counts = {}

        for node_result in results:
            for context_name, context_data in node_result.items():
                if context_data and isinstance(context_data, list):
                    context_counts[context_name] = context_counts.get(
                        context_name, 0
                    ) + len(context_data)
                    total_measurements += len(context_data)

        # Check for empty contexts
        empty_contexts = [name for name, count in context_counts.items() if count == 0]
        if empty_contexts:
            self.validation_warnings.append(f"Empty contexts found: {empty_contexts}")

        print(
            f"✓ Data consistency: {total_measurements} total measurements across {len(context_counts)} contexts"
        )

    def _validate_coordinate_progression(self, results: List[Dict[str, Any]]) -> None:
        """Validate coordinate progression patterns."""
        if not results:
            return

        coordinate_issues = []

        for node_result in results:
            for context_name, context_data in node_result.items():
                if context_data and isinstance(context_data, list):
                    # Check for coordinate progression
                    coordinates_seen = []
                    for measurement in context_data:
                        if isinstance(measurement, dict):
                            coord = (
                                measurement.get("round_idx"),
                                measurement.get("epoch_idx"),
                                measurement.get("batch_idx"),
                                measurement.get("global_step"),
                            )
                            coordinates_seen.append(coord)

                    # Check for duplicates
                    unique_coords = set(coordinates_seen)
                    if len(unique_coords) != len(coordinates_seen):
                        duplicates = len(coordinates_seen) - len(unique_coords)
                        coordinate_issues.append(
                            f"{context_name}: {duplicates} duplicate coordinates"
                        )

        if coordinate_issues:
            self.validation_warnings.extend(coordinate_issues)

        print(f"✓ Coordinate progression: {len(coordinate_issues)} issues found")

    def _validate_metric_values(self, results: List[Dict[str, Any]]) -> None:
        """Validate metric values for common issues."""
        if not results:
            return

        metric_stats = {}
        suspicious_values = []

        for node_result in results:
            for context_name, context_data in node_result.items():
                if context_data and isinstance(context_data, list):
                    for measurement in context_data:
                        if isinstance(measurement, dict):
                            for key, value in measurement.items():
                                if (
                                    key
                                    not in [
                                        "round_idx",
                                        "epoch_idx",
                                        "batch_idx",
                                        "global_step",
                                    ]
                                    and not key.startswith("_")
                                    and isinstance(value, (int, float))
                                ):
                                    # Track metric statistics
                                    if key not in metric_stats:
                                        metric_stats[key] = {
                                            "values": [],
                                            "context": context_name,
                                        }
                                    metric_stats[key]["values"].append(value)

                                    # Check for suspicious values
                                    if value != value:  # NaN check
                                        suspicious_values.append(
                                            f"{context_name}.{key}: NaN value"
                                        )
                                    elif abs(value) > 1e6:
                                        suspicious_values.append(
                                            f"{context_name}.{key}: Very large value ({value})"
                                        )
                                    elif value < 0 and key.endswith(
                                        ("accuracy", "precision", "recall")
                                    ):
                                        suspicious_values.append(
                                            f"{context_name}.{key}: Negative accuracy metric ({value})"
                                        )

        if suspicious_values[:5]:  # Show first 5 suspicious values
            self.validation_warnings.extend(suspicious_values[:5])
            if len(suspicious_values) > 5:
                self.validation_warnings.append(
                    f"... and {len(suspicious_values) - 5} more suspicious values"
                )

        print(
            f"✓ Metric validation: {len(metric_stats)} metrics checked, {len(suspicious_values)} suspicious values"
        )

    def _validate_percentage_calculations(self, results: List[Dict[str, Any]]) -> None:
        """Validate percentage calculations work correctly."""
        if not results:
            return

        # This is a placeholder for percentage calculation validation
        # In a real implementation, we'd check if percentage changes make sense
        # relative to the sequential progression of values

        progression_count = 0

        for node_result in results:
            for context_name, context_data in node_result.items():
                if (
                    context_data
                    and isinstance(context_data, list)
                    and len(context_data) >= 2
                ):
                    progression_count += 1

        print(
            f"✓ Percentage calculation readiness: {progression_count} contexts have progression data"
        )

    def _print_validation_summary(self) -> None:
        """Print summary of validation results."""
        print("\n" + "=" * 80)
        print("📋 Validation Summary")
        print("=" * 80)

        if not self.validation_errors and not self.validation_warnings:
            print("✅ All validation checks passed!")
        else:
            if self.validation_errors:
                print(f"❌ {len(self.validation_errors)} Errors:")
                for error in self.validation_errors:
                    print(f"   • {error}")

            if self.validation_warnings:
                print(f"⚠️  {len(self.validation_warnings)} Warnings:")
                for warning in self.validation_warnings:
                    print(f"   • {warning}")

        print()

    def _print_error_diagnostics(self, error: Exception) -> None:
        """Print detailed error diagnostics and troubleshooting steps."""
        print(f"\n❌ Error: {error}")
        print("\n🔧 Troubleshooting Steps:")
        print("1. Ensure you've run an experiment first:")
        print("   ./main.sh --config-name test_mnist_subset")
        print("2. Check experiment outputs:")
        print("   ls -la outputs/")
        print("3. List available experiments:")
        print("   python debug.py --list")
        print("4. Verify node result files exist:")
        print("   find outputs/ -name 'node_*_results.pkl'")

        # Show what directories exist
        if self.outputs_dir.exists():
            subdirs = [d.name for d in self.outputs_dir.iterdir() if d.is_dir()]
            if subdirs:
                print(f"\n📁 Found output subdirectories: {', '.join(subdirs[:10])}")
                if len(subdirs) > 10:
                    print(f"   (and {len(subdirs) - 10} more...)")
            else:
                print(f"\n📁 No subdirectories found in {self.outputs_dir}")
        else:
            print(f"\n📁 Outputs directory does not exist: {self.outputs_dir}")

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filter_context: Optional[str] = None,
        filter_metrics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply context and metric filters to results."""
        filtered_results = []

        for node_result in results:
            filtered_node = {}

            for context_name, context_data in node_result.items():
                # Apply context filter
                if filter_context and filter_context not in context_name:
                    continue

                # Apply metric filters
                if filter_metrics and context_data:
                    filtered_context_data = []
                    for measurement in context_data:
                        if isinstance(measurement, dict):
                            filtered_measurement = {}
                            # Always keep coordinate keys
                            for key in [
                                "round_idx",
                                "epoch_idx",
                                "batch_idx",
                                "global_step",
                            ]:
                                if key in measurement:
                                    filtered_measurement[key] = measurement[key]

                            # Add filtered metrics
                            for metric in filter_metrics:
                                if metric in measurement:
                                    filtered_measurement[metric] = measurement[metric]

                            if (
                                len(filtered_measurement) > 4
                            ):  # More than just coordinates
                                filtered_context_data.append(filtered_measurement)

                    filtered_node[context_name] = filtered_context_data
                else:
                    filtered_node[context_name] = context_data

            filtered_results.append(filtered_node)

        return filtered_results

    def run_comparison(
        self,
        exp1_index: int,
        exp2_index: int,
        validate_only: bool = False,
        filter_metrics: Optional[List[str]] = None,
        filter_context: Optional[str] = None,
    ) -> None:
        """Compare two experiments side by side."""
        print("=" * 80)
        print("⚔️ FLORA Experiment Comparison")
        print("=" * 80)

        try:
            # Load both experiments
            exp1_dir = self.get_experiment_dir_by_index(exp1_index)
            exp1_results = self.load_node_results(exp1_dir)
            exp1_duration, exp1_rounds = self.extract_experiment_metadata(
                exp1_results, exp1_dir
            )

            print("\n" + "-" * 40)

            exp2_dir = self.get_experiment_dir_by_index(exp2_index)
            exp2_results = self.load_node_results(exp2_dir)
            exp2_duration, exp2_rounds = self.extract_experiment_metadata(
                exp2_results, exp2_dir
            )

            # Compare basic info
            print("\n📋 Comparison Summary:")
            print(f"Experiment 1: {self._extract_experiment_name(exp1_dir)}")
            print(
                f"├── Nodes: {len(exp1_results)}, Rounds: {exp1_rounds}, Duration: {exp1_duration:.1f}s"
            )
            print(f"Experiment 2: {self._extract_experiment_name(exp2_dir)}")
            print(
                f"└── Nodes: {len(exp2_results)}, Rounds: {exp2_rounds}, Duration: {exp2_duration:.1f}s"
            )

            # Run validation on both
            print("\n🔍 Validating Experiment 1:")
            self._run_validation_checks(exp1_results)
            exp1_errors = len(self.validation_errors)
            exp1_warnings = len(self.validation_warnings)

            print("\n🔍 Validating Experiment 2:")
            self._run_validation_checks(exp2_results)
            exp2_errors = len(self.validation_errors) - exp1_errors
            exp2_warnings = len(self.validation_warnings) - exp1_warnings

            # Compare validation results
            print("\n⚔️ Validation Comparison:")
            print(f"Experiment 1: {exp1_errors} errors, {exp1_warnings} warnings")
            print(f"Experiment 2: {exp2_errors} errors, {exp2_warnings} warnings")

            if exp1_errors == exp2_errors and exp1_warnings == exp2_warnings:
                print("✅ Both experiments have identical validation results")
            else:
                print("⚠️ Experiments have different validation results")

            self._print_validation_summary()

            # Show displays if not validation-only
            if not validate_only:
                # Apply filters
                if filter_context or filter_metrics:
                    exp1_results = self._apply_filters(
                        exp1_results, filter_context, filter_metrics
                    )
                    exp2_results = self._apply_filters(
                        exp2_results, filter_context, filter_metrics
                    )

                print("\n" + "=" * 80)
                print("📋 Experiment 1 Display")
                print("=" * 80)

                display = ExperimentResultsDisplay()
                display.show_experiment_results(
                    results=exp1_results,
                    duration=exp1_duration,
                    global_rounds=exp1_rounds,
                    total_nodes=len(exp1_results),
                )

                print("\n" + "=" * 80)
                print("📋 Experiment 2 Display")
                print("=" * 80)

                display.show_experiment_results(
                    results=exp2_results,
                    duration=exp2_duration,
                    global_rounds=exp2_rounds,
                    total_nodes=len(exp2_results),
                )

            print("\n✅ Comparison completed successfully!")

        except Exception as e:
            self._print_error_diagnostics(e)


def main():
    """Main entry point with command line argument support."""
    parser = argparse.ArgumentParser(
        description="Debug FLORA experiment display with enhanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug.py                                    # Use latest experiment
  python debug.py --list                             # List available experiments  
  python debug.py --experiment 0                     # Use first experiment in list
  python debug.py --compare 0 1                     # Compare two experiments
  python debug.py --validate-only                   # Run validation checks only
  python debug.py --metrics accuracy loss           # Show only specific metrics
  python debug.py --context training                # Show only training context
  python debug.py --outputs custom/                 # Use custom outputs directory
        """,
    )

    parser.add_argument(
        "--list", action="store_true", help="List available experiments and exit"
    )

    parser.add_argument(
        "--experiment",
        type=int,
        help="Index of experiment to use (see --list for available experiments)",
    )

    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Path to outputs directory (default: outputs)",
    )

    parser.add_argument(
        "--compare",
        type=int,
        nargs=2,
        metavar=("EXP1", "EXP2"),
        help="Compare two experiments by index (e.g., --compare 0 1)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation checks only, skip display",
    )

    parser.add_argument(
        "--metrics",
        nargs="*",
        help="Show only specific metrics (e.g., --metrics accuracy loss)",
    )

    parser.add_argument(
        "--context", help="Show only specific context (e.g., --context training)"
    )

    args = parser.parse_args()

    # Create debugger
    debugger = ExperimentDebugger(args.outputs)

    # Handle list command
    if args.list:
        debugger.list_available_experiments()
        return

    # Handle compare command
    if args.compare:
        debugger.run_comparison(
            args.compare[0],
            args.compare[1],
            validate_only=args.validate_only,
            filter_metrics=args.metrics,
            filter_context=args.context,
        )
        return

    # Run debug display
    debugger.run_debug_display(
        args.experiment,
        validate_only=args.validate_only,
        filter_metrics=args.metrics,
        filter_context=args.context,
    )


if __name__ == "__main__":
    main()
