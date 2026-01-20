"""
Team ratings optimization logic.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .core import ModelConfig


class TeamRatingsOptimizer:
    """Handles the optimization of team ratings."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.regularization_path = []

    def optimize(
        self, data: pd.DataFrame, team_index: Dict[str, int], n_teams: int
    ) -> Dict:
        """Optimize team ratings with multiple starts."""
        best_result = None
        best_loss = float("inf")

        for _ in range(self.config.n_optimization_starts):
            initial_params = self._initialize_parameters(n_teams)

            try:
                result = minimize(
                    lambda p: self._objective_function(p, data, team_index, n_teams),
                    initial_params,
                    method="L-BFGS-B",
                    bounds=self._get_bounds(n_teams),
                    options={"maxiter": self.config.max_iter, "ftol": 1e-9},
                )

                if result.fun < best_loss and result.success:
                    best_result = result
                    best_loss = result.fun

            except Exception:
                continue

        if best_result is None:
            raise RuntimeError("All optimization attempts failed")

        return self._unpack_parameters(best_result.x, n_teams)

    def _initialize_parameters(self, n_teams: int) -> np.ndarray:
        """Initialize parameters with appropriate starting values."""
        n_params = 2 * n_teams + 2
        std_dev = 0.05 if self.config.shrink_to_mean else 0.1
        params = np.random.normal(0, std_dev, n_params)

        # Set global parameters
        params[-2] = 0.2  # home advantage
        params[-1] = -0.1  # away adjustment
        return params

    def _get_bounds(self, n_teams: int) -> List[Tuple[float, float]]:
        """Get parameter bounds."""
        team_bounds = [self.config.param_bounds] * (2 * n_teams)
        global_bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        return team_bounds + global_bounds

    def _objective_function(
        self,
        params: np.ndarray,
        data: pd.DataFrame,
        team_index: Dict[str, int],
        n_teams: int,
    ) -> float:
        """Calculate regularized objective function."""
        base_loss = self._calculate_base_loss(params, data, team_index, n_teams)
        regularization = self._calculate_regularization(params, n_teams)

        total_loss = base_loss + regularization
        self._log_regularization_path(base_loss, regularization, total_loss)

        return total_loss

    def _calculate_base_loss(
        self,
        params: np.ndarray,
        data: pd.DataFrame,
        team_index: Dict[str, int],
        n_teams: int,
    ) -> float:
        """Calculate base prediction loss."""
        attack_ratings, defense_ratings, home_adv, away_adj = self._split_params(
            params, n_teams
        )

        # Get team indices
        home_indices = data["Home"].map(team_index).values
        away_indices = data["Away"].map(team_index).values

        # Compute strengths and predictions
        home_strength = (
            home_adv + attack_ratings[home_indices] - defense_ratings[away_indices]
        )
        away_strength = (
            away_adj + attack_ratings[away_indices] - defense_ratings[home_indices]
        )

        home_goals_pred = self._strength_to_goals(home_strength, data["FTHG"])
        away_goals_pred = self._strength_to_goals(away_strength, data["FTAG"])

        # Weighted squared error
        home_error = (home_goals_pred - data["FTHG"].values) ** 2
        away_error = (away_goals_pred - data["FTAG"].values) ** 2
        weights = data["Weight"].values

        return np.sum(weights * (home_error + away_error))

    def _calculate_regularization(self, params: np.ndarray, n_teams: int) -> float:
        """Calculate all regularization terms."""
        attack_ratings, defense_ratings, home_adv, away_adj = self._split_params(
            params, n_teams
        )

        # L1 and L2 penalties
        l1_penalty = self.config.l1_reg * (
            np.sum(np.abs(attack_ratings)) + np.sum(np.abs(defense_ratings))
        )

        l2_penalty = self.config.l2_reg * (
            np.sum(attack_ratings**2)
            + np.sum(defense_ratings**2)
            + home_adv**2
            + away_adj**2
        )

        # Team imbalance penalty
        team_penalty = 0.0
        if self.config.team_reg > 0:
            imbalances = (attack_ratings - defense_ratings) ** 2
            team_penalty = self.config.team_reg * np.sum(imbalances)

        # Shrinkage penalty
        shrinkage_penalty = 0.0
        if self.config.shrink_to_mean:
            mean_attack = np.mean(attack_ratings)
            mean_defense = np.mean(defense_ratings)
            shrinkage_penalty = 0.001 * (
                np.sum((attack_ratings - mean_attack) ** 2)
                + np.sum((defense_ratings - mean_defense) ** 2)
            )

        return l1_penalty + l2_penalty + team_penalty + shrinkage_penalty

    def _split_params(
        self, params: np.ndarray, n_teams: int
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Split parameter array into components."""
        attack_ratings = params[:n_teams]
        defense_ratings = params[n_teams : 2 * n_teams]
        home_adv, away_adj = params[-2:]
        return attack_ratings, defense_ratings, home_adv, away_adj

    def _strength_to_goals(
        self, strength: np.ndarray, actual_goals: pd.Series
    ) -> np.ndarray:
        """Convert team strength to expected goals."""
        baseline = actual_goals.mean()
        std_dev = actual_goals.std()

        # Sigmoid transformation
        prob = 1 / (1 + np.exp(-np.clip(strength, *self.config.param_bounds)))
        prob = np.clip(prob, self.config.eps, 1 - self.config.eps)
        z_score = norm.ppf(prob)

        return baseline + z_score * std_dev

    def _log_regularization_path(
        self, base_loss: float, regularization: float, total_loss: float
    ):
        """Log regularization path for diagnostics."""
        self.regularization_path.append(
            {
                "base_loss": base_loss,
                "regularization": regularization,
                "total_loss": total_loss,
            }
        )

    def _unpack_parameters(self, params: np.ndarray, n_teams: int) -> Dict:
        """Unpack optimized parameters."""
        attack_ratings, defense_ratings, home_adv, away_adj = self._split_params(
            params, n_teams
        )
        return {
            "attack_ratings": attack_ratings,
            "defense_ratings": defense_ratings,
            "home_advantage": home_adv,
            "away_adjustment": away_adj,
        }
