"""
Prediction strategy implementations.
"""

import os
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from .core import ModelConfig


class BasePredictor(ABC):
    """Base class for prediction strategies."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float, **kwargs
    ) -> Tuple[float, float, float]:
        """Predict match outcome probabilities."""
        pass


class BasePoissonCalculator:
    """Shared Poisson calculation utilities."""

    def __init__(self, max_goals: int = 15):
        self.max_goals = max_goals

    def compute_prob_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Compute probability matrix for all score combinations."""
        home_goals = np.arange(0, self.max_goals + 1)
        away_goals = np.arange(0, self.max_goals + 1)
        home_probs = poisson.pmf(home_goals, lambda_home)
        away_probs = poisson.pmf(away_goals, lambda_away)
        return np.outer(home_probs, away_probs)

    def matrix_to_outcomes(self, prob_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Convert probability matrix to match outcome probabilities."""
        prob_matrix /= prob_matrix.sum()
        home_win = np.tril(prob_matrix, -1).sum()
        draw = np.trace(prob_matrix)
        away_win = np.triu(prob_matrix, 1).sum()
        return home_win, draw, away_win


class PoissonPredictor(BasePredictor, BasePoissonCalculator):
    """Standard Poisson prediction."""

    def __init__(self, config: ModelConfig):
        BasePredictor.__init__(self, config)
        BasePoissonCalculator.__init__(self, config.max_goals)

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float, **kwargs
    ) -> Tuple[float, float, float]:
        prob_matrix = self.compute_prob_matrix(lambda_home, lambda_away)
        return self.matrix_to_outcomes(prob_matrix)


class ZIPPredictor(BasePredictor, BasePoissonCalculator):
    """Zero-inflated Poisson with dynamic adjustment."""

    def __init__(self, config: ModelConfig, historical_data: pd.DataFrame):
        BasePredictor.__init__(self, config)
        BasePoissonCalculator.__init__(self, config.max_goals)
        self.adjustment_matrix = self._compute_adjustment_matrix(historical_data)

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float, **kwargs
    ) -> Tuple[float, float, float]:
        base_matrix = self.compute_prob_matrix(lambda_home, lambda_away)
        adjusted_matrix = base_matrix * self.adjustment_matrix
        return self.matrix_to_outcomes(adjusted_matrix)

    def _compute_adjustment_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Compute adjustment matrix from recent historical data."""
        recent_data = data.tail(min(200, len(data)))

        # Compute observed distribution
        home_dist = recent_data["FTHG"].value_counts(normalize=True).sort_index()
        away_dist = recent_data["FTAG"].value_counts(normalize=True).sort_index()

        idx = np.arange(0, self.max_goals + 1)
        observed = np.outer(
            home_dist.reindex(idx, fill_value=0), away_dist.reindex(idx, fill_value=0)
        )

        # Compute expected distribution
        avg_home = recent_data["FTHG"].mean()
        avg_away = recent_data["FTAG"].mean()
        expected = self.compute_prob_matrix(avg_home, avg_away)

        # Calculate adjustment with bounds
        adjustment = np.where(expected > 1e-6, observed / expected, 1.0)
        return np.clip(adjustment, 0.1, 5.0)


class MOVPredictor(BasePredictor):
    """Margin of Victory based predictor."""

    def __init__(self, config: ModelConfig, mov_model):
        super().__init__(config)
        self.mov_model = mov_model

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float, **kwargs
    ) -> Tuple[float, float, float]:
        """Predict match outcomes based on MOV model."""
        mov_prediction = kwargs.get("mov_prediction")
        mov_std_error = kwargs.get("mov_std_error")

        if mov_prediction is None:
            raw_mov = lambda_home - lambda_away
            mov_prediction = (
                self.mov_model.params[0] + self.mov_model.params[1] * raw_mov
            )

        if mov_std_error is None:
            mov_std_error = np.sqrt(self.mov_model.mse_resid)

        # Use normal distribution for outcomes
        prob_home = 1 - norm.cdf(0.5, mov_prediction, mov_std_error)
        prob_away = norm.cdf(-0.5, mov_prediction, mov_std_error)
        prob_draw = max(0.05, 1 - prob_home - prob_away)  # Minimum draw probability

        total = prob_home + prob_draw + prob_away
        return (prob_home / total, prob_draw / total, prob_away / total)


class PredictorFactory:
    """Factory for creating predictor instances."""

    @staticmethod
    def create_predictors(
        config: ModelConfig, historical_data: pd.DataFrame, mov_model
    ) -> dict:
        """Create all available predictors."""
        predictors = {
            "poisson": PoissonPredictor(config),
            "zip": ZIPPredictor(config, historical_data),
            "mov": MOVPredictor(config, mov_model),
        }

        # Add ML predictor if available
        try:
            import os

            from .predictors_ml import MLPredictor

            ml_model_path = "ml_models/ml_predictor.pkl"
            if os.path.exists(ml_model_path):
                predictors["ml"] = MLPredictor(config, model_path=ml_model_path)
                print("  ✓ ML predictor loaded")
        except (ImportError, FileNotFoundError):
            pass  # ML predictor not available

        return predictors
