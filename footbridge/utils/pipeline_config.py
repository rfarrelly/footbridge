# utils/pipeline_config.py
"""
Pipeline configuration and validation utilities.
Centralized configuration management for the betting pipeline.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class PipelineConfig:
    """Configuration manager for the betting pipeline."""

    def __init__(self, base_config=None):
        """Initialize pipeline configuration."""
        from config import DEFAULT_CONFIG

        self.base_config = base_config or DEFAULT_CONFIG

        # Pipeline-specific settings
        self.min_edge_threshold = 0.05  # Minimum edge for betting recommendations
        self.max_recommendations = 20  # Maximum number of recommendations to display
        self.parameter_optimization_days = 60  # Days before re-optimization needed

        # File paths
        self.output_files = {
            "historical_ppi": "historical_ppi.csv",
            "historical_merged": "historical_ppi_and_odds.csv",
            "latest_ppi": "latest_ppi.csv",
            "fixtures_merged": "fixtures_ppi_and_odds.csv",
            "zsd_enhanced": "latest_zsd_enhanced.csv",
            "zsd_candidates": "zsd_betting_candidates.csv",
            "team_dictionary": "team_name_dictionary.csv",
        }

        # ZSD configuration directory
        self.zsd_config_dir = Path("zsd_configs")

    def get_required_files(self, mode: str) -> List[str]:
        """Get list of required files for a specific pipeline mode."""
        requirements = {
            "historical_ppi": [],
            "latest_ppi": [],
            "merge_historical": ["historical_ppi.csv"],
            "merge_future": ["latest_ppi.csv"],
            "zsd_predict": ["fixtures_ppi_and_odds.csv"],
            "full_pipeline": ["team_name_dictionary.csv"],
        }

        return requirements.get(mode, [])

    def validate_data_files(
        self, required_files: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate that required data files exist and are readable."""
        if required_files is None:
            required_files = ["team_name_dictionary.csv"]

        missing_files = []

        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
            else:
                try:
                    # Try to read the file to ensure it's not corrupted
                    pd.read_csv(file_path, nrows=1)
                except Exception:
                    missing_files.append(f"{file_path} (corrupted or unreadable)")

        return len(missing_files) == 0, missing_files

    def validate_data_directories(self) -> Tuple[bool, List[str]]:
        """Validate that data directories exist and contain files."""
        issues = []

        # Check fbref data directory
        if not self.base_config.fbref_data_dir.exists():
            issues.append(
                f"fbref data directory does not exist: {self.base_config.fbref_data_dir}"
            )
        elif not list(self.base_config.fbref_data_dir.rglob("*.csv")):
            issues.append(
                f"No CSV files found in fbref directory: {self.base_config.fbref_data_dir}"
            )

        # Check fbduk data directory
        if not self.base_config.fbduk_data_dir.exists():
            issues.append(
                f"fbduk data directory does not exist: {self.base_config.fbduk_data_dir}"
            )
        elif not list(self.base_config.fbduk_data_dir.rglob("*.csv")):
            issues.append(
                f"No CSV files found in fbduk directory: {self.base_config.fbduk_data_dir}"
            )

        return len(issues) == 0, issues

    def check_zsd_optimization_needed(self) -> bool:
        """Check if ZSD parameter optimization is needed."""
        # If no configs exist, optimization is needed
        if (
            not self.zsd_config_dir.exists()
            or len(list(self.zsd_config_dir.glob("*.json"))) == 0
        ):
            return True

        # Check if any config is older than threshold
        cutoff_date = datetime.now() - timedelta(days=self.parameter_optimization_days)

        for config_file in self.zsd_config_dir.glob("*.json"):
            if config_file.stat().st_mtime < cutoff_date.timestamp():
                return True

        return False

    def get_pipeline_status(self) -> Dict[str, any]:
        """Get comprehensive pipeline status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "data_directories": {},
            "output_files": {},
            "zsd_optimization": {},
        }

        # Check data directories
        dirs_valid, dir_issues = self.validate_data_directories()
        status["data_directories"] = {
            "valid": dirs_valid,
            "issues": dir_issues,
        }

        # Check output files
        for name, path in self.output_files.items():
            file_path = Path(path)
            if file_path.exists():
                stat = file_path.stat()
                status["output_files"][name] = {
                    "exists": True,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            else:
                status["output_files"][name] = {"exists": False}

        # Check ZSD optimization status
        status["zsd_optimization"] = {
            "needed": self.check_zsd_optimization_needed(),
            "config_dir_exists": self.zsd_config_dir.exists(),
            "num_configs": (
                len(list(self.zsd_config_dir.glob("*.json")))
                if self.zsd_config_dir.exists()
                else 0
            ),
        }

        return status

    def print_status_report(self):
        """Print a comprehensive status report."""
        status = self.get_pipeline_status()

        print(f"{'=' * 60}\r\nBETTING PIPELINE STATUS REPORT\r\n{'=' * 60}\r\n")
        print(f"Generated: {status['timestamp']}")

        # Data directories
        print(f"\nData Directories:")
        if status["data_directories"]["valid"]:
            print("  ✓ All data directories are valid")
        else:
            print("  ✗ Data directory issues:")
            for issue in status["data_directories"]["issues"]:
                print(f"    - {issue}")

        # Output files
        print(f"\nOutput Files:")
        for name, info in status["output_files"].items():
            if info["exists"]:
                print(
                    f"  ✓ {name}: {info['size_mb']:.1f}MB (modified: {info['modified']})"
                )
            else:
                print(f"  ✗ {name}: Missing")

        # ZSD optimization
        print(f"\nZSD Configuration:")
        zsd_status = status["zsd_optimization"]
        print(
            f"  Config directory exists: {'✓' if zsd_status['config_dir_exists'] else '✗'}"
        )
        print(f"  Number of configs: {zsd_status['num_configs']}")
        print(f"  Optimization needed: {'✓' if zsd_status['needed'] else '✗'}")

        print("=" * 60)


class PipelineModeValidator:
    """Validator for different pipeline execution modes."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def validate_mode(self, mode: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a pipeline mode.

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        if mode == "get_data":
            # Check if data directories exist
            dirs_valid, dir_issues = self.config.validate_data_directories()
            if not dirs_valid:
                warnings.extend(dir_issues)

        elif mode == "update_teams":
            # Check if data files exist for team name extraction
            dirs_valid, dir_issues = self.config.validate_data_directories()
            if not dirs_valid:
                errors.append("Cannot update teams without data files")
                errors.extend(dir_issues)

        elif mode == "historical_ppi":
            # Check if team dictionary exists
            files_valid, missing = self.config.validate_data_files(
                ["team_name_dictionary.csv"]
            )
            if not files_valid:
                warnings.append(
                    "Team dictionary missing - team name mapping may not work correctly"
                )

        elif mode == "latest_ppi":
            # Check if team dictionary exists
            files_valid, missing = self.config.validate_data_files(
                ["team_name_dictionary.csv"]
            )
            if not files_valid:
                warnings.append(
                    "Team dictionary missing - team name mapping may not work correctly"
                )

        elif mode == "predict":
            # Check if required files exist
            required = self.config.get_required_files("zsd_predict")
            files_valid, missing = self.config.validate_data_files(required)
            if not files_valid:
                errors.append(f"Missing required files for ZSD predictions: {missing}")

        elif mode == "full":
            # Validate all components
            dirs_valid, dir_issues = self.config.validate_data_directories()
            if not dirs_valid:
                warnings.extend(dir_issues)

            files_valid, missing = self.config.validate_data_files(
                ["team_name_dictionary.csv"]
            )
            if not files_valid:
                warnings.append(
                    "Team dictionary missing - run 'update_teams' mode first"
                )

        return len(errors) == 0, errors, warnings

    def get_mode_description(self, mode: str) -> str:
        """Get description of what a mode does."""
        descriptions = {
            "get_data": "Download and update data files for a specific season",
            "update_teams": "Build/update team name dictionary for data source mapping",
            "historical_ppi": "Generate historical PPI data and merge with odds",
            "latest_ppi": "Generate latest PPI predictions and merge with current odds",
            "optimize": "Run ZSD parameter optimization (computationally intensive)",
            "predict": "Generate ZSD predictions using existing parameters",
            "compare": "Compare results from different prediction methods",
            "full": "Run complete pipeline with all components",
        }

        return descriptions.get(mode, "Unknown mode")


def print_pipeline_help():
    """Print comprehensive help for pipeline usage."""
    validator = PipelineModeValidator(PipelineConfig())

    print(f"{'=' * 60}\r\nBETTING PIPELINE USAGE GUIDE\r\n{'=' * 60}\r\n")

    print("Available modes:")
    modes = [
        "get_data",
        "update_teams",
        "historical_ppi",
        "latest_ppi",
        "optimize",
        "predict",
        "compare",
        "full",
    ]

    for mode in modes:
        print(f"\n  {mode}")
        print(f"    {validator.get_mode_description(mode)}")

        if mode == "get_data":
            print("    Usage: python main.py get_data <season>")

    print(f"\nRecommended workflow:")
    print("  1. python main.py get_data <season>  # Download data for analysis")
    print("  2. python main.py update_teams       # Build team name mappings")
    print("  3. python main.py historical_ppi     # Generate historical data")
    print("  4. python main.py optimize           # Optimize ZSD parameters (once)")
    print("  5. python main.py full               # Run daily predictions")

    print(f"\nFor daily use:")
    print("  python main.py predict               # Quick predictions only")
    print("  python main.py full                  # Full pipeline with PPI + ZSD")

    print("=" * 60)
