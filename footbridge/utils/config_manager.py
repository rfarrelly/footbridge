"""
Configuration management for league-specific model parameters.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.core import ModelConfig


class ConfigManager:
    """Manages league-specific configurations."""

    def __init__(self, config_dir: Path = Path("zsd_configs")):
        self.config_dir = config_dir
        self.config_dir.mkdir(exist_ok=True)

        self.default_config = ModelConfig(
            decay_rate=0.001,
            max_goals=15,
            l1_reg=0.0,
            l2_reg=0.01,
            team_reg=0.005,
            auto_tune_regularization=False,
            min_matches_per_team=5,
        )

    def save_league_config(self, league: str, config: ModelConfig) -> None:
        """Save league-specific configuration."""
        config_file = self._get_config_file(league)
        config_dict = self._config_to_dict(config)

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    def load_league_config(self, league: str) -> ModelConfig:
        """Load league-specific configuration."""
        config_file = self._get_config_file(league)

        if not config_file.exists():
            print(f"No saved config for {league}, using default")
            return self.default_config

        try:
            with open(config_file, "r") as f:
                config_dict = json.load(f)

            config_dict.pop("optimized_date", None)  # Remove metadata
            return ModelConfig(**config_dict)

        except Exception as e:
            print(f"Error loading config for {league}: {e}, using default")
            return self.default_config

    def should_reoptimize(self, league: str, days_threshold: int = 90) -> bool:
        """Check if parameters should be re-optimized."""
        config_file = self._get_config_file(league)

        if not config_file.exists():
            return True

        try:
            with open(config_file, "r") as f:
                config_dict = json.load(f)

            if "optimized_date" not in config_dict:
                return True

            opt_date = datetime.fromisoformat(config_dict["optimized_date"])
            return (datetime.now() - opt_date).days > days_threshold

        except Exception:
            return True

    def _get_config_file(self, league: str) -> Path:
        """Get configuration file path for league."""
        safe_name = league.replace(" ", "_").replace("-", "_")
        return self.config_dir / f"{safe_name}_config.json"

    def _config_to_dict(self, config: ModelConfig) -> dict:
        """Convert ModelConfig to dictionary with metadata."""
        config_dict = {
            k: v for k, v in config.__dict__.items() if not k.startswith("_")
        }
        config_dict["optimized_date"] = datetime.now().isoformat()
        return config_dict
