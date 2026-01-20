"""
Core model classes and configurations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for the soccer prediction model."""

    # Core parameters
    decay_rate: float = 0.001
    max_goals: int = 15
    eps: float = 1e-6
    lambda_bounds: Tuple[float, float] = (0.01, 10.0)
    param_bounds: Tuple[float, float] = (-6.0, 6.0)
    min_matches_per_team: int = 5

    # Regularization parameters
    l1_reg: float = 0.0
    l2_reg: float = 0.01
    team_reg: float = 0.005
    shrink_to_mean: bool = True

    # Optimization
    auto_tune_regularization: bool = False
    cv_folds: int = 5
    n_optimization_starts: int = 3
    max_iter: int = 2000


@dataclass
class MatchPrediction:
    """Structured output for match predictions."""

    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    mov_prediction: float
    mov_std_error: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    model_type: str

    @property
    def probabilities(self) -> np.ndarray:
        """Return probabilities as numpy array."""
        return np.array([self.prob_home_win, self.prob_draw, self.prob_away_win])


@dataclass
class BettingMetrics:
    """Structured betting analysis results."""

    bet_type: str
    edge: float
    model_prob: float
    market_prob: float
    sharp_odds: float
    soft_odds: float
    fair_odds: float
    expected_values: List[float]
    kelly_edges: List[float]
    probability_edges: List[float]
