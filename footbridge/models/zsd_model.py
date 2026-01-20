"""
Main ZSD Poisson model implementation.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .core import MatchPrediction, ModelConfig
from .optimizer import TeamRatingsOptimizer
from .predictors import PredictorFactory


class ZSDPoissonModel:
    """Enhanced soccer prediction model with regularization."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.optimizer = TeamRatingsOptimizer(self.config)

        # Model components
        self.teams = []
        self.team_index = {}
        self.attack_ratings = None
        self.defense_ratings = None
        self.home_advantage = None
        self.away_adjustment = None
        self.mov_model = None
        self.predictors = {}

        # Training diagnostics
        self.convergence_info = {}
        self.data = None
        self.n_teams = 0

    def fit(self, matches_df: pd.DataFrame) -> None:
        """Fit the model to historical match data."""
        self.data = self._prepare_data(matches_df)
        self._validate_data()
        self._initialize_teams()

        # Optimize ratings
        ratings = self.optimizer.optimize(self.data, self.team_index, self.n_teams)
        self._assign_ratings(ratings)

        # Fit MOV model and initialize predictors
        self._fit_mov_model()
        self._initialize_predictors()

        self.convergence_info = {"success": True}

    def predict_match(
        self, home_team: str, away_team: str, method: str = "zip", **kwargs
    ) -> MatchPrediction:
        """Predict a single match outcome."""
        self._validate_teams(home_team, away_team)
        self._validate_method(method)

        # Calculate expected goals
        home_idx = self.team_index[home_team]
        away_idx = self.team_index[away_team]

        lambda_home = self._calculate_expected_goals(home_idx, away_idx, is_home=True)
        lambda_away = self._calculate_expected_goals(away_idx, home_idx, is_home=False)

        # Predict MOV
        predicted_mov, mov_std_error = self._predict_mov(lambda_home, lambda_away)

        # Get outcome probabilities
        kwargs = {"mov_prediction": predicted_mov, "mov_std_error": mov_std_error}
        prob_home, prob_draw, prob_away = self.predictors[
            method
        ].predict_outcome_probabilities(lambda_home, lambda_away, **kwargs)

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            mov_prediction=predicted_mov,
            mov_std_error=mov_std_error,
            prob_home_win=prob_home,
            prob_draw=prob_draw,
            prob_away_win=prob_away,
            model_type=method,
        )

    def get_team_ratings(self) -> pd.DataFrame:
        """Return current team ratings."""
        return pd.DataFrame(
            {
                "Team": self.teams,
                "Attack_Rating": self.attack_ratings,
                "Defense_Rating": self.defense_ratings,
                "Net_Rating": self.attack_ratings - self.defense_ratings,
            }
        ).sort_values("Net_Rating", ascending=False)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the match data."""
        df = df.copy()

        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            df["Date"] = pd.to_datetime(df["Date"])

        df["Home_MOV"] = df["FTHG"] - df["FTAG"]
        df["Total_Goals"] = df["FTHG"] + df["FTAG"]
        df = df.sort_values("Date").reset_index(drop=True)

        # Add time weights
        match_index = np.arange(len(df))[::-1]
        df["Weight"] = np.exp(-self.config.decay_rate * match_index)
        df["Weight"] *= len(df) / df["Weight"].sum()

        return df

    def _validate_data(self) -> None:
        """Validate data quality."""
        required_cols = ["Home", "Away", "FTHG", "FTAG", "Date"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _initialize_teams(self) -> None:
        """Initialize team indexing."""
        self.teams = sorted(list(set(self.data["Home"]).union(set(self.data["Away"]))))
        self.team_index = {team: i for i, team in enumerate(self.teams)}
        self.n_teams = len(self.teams)

    def _assign_ratings(self, ratings: Dict) -> None:
        """Assign optimized ratings to model attributes."""
        self.attack_ratings = ratings["attack_ratings"]
        self.defense_ratings = ratings["defense_ratings"]
        self.home_advantage = ratings["home_advantage"]
        self.away_adjustment = ratings["away_adjustment"]

    def _calculate_expected_goals(
        self, attacking_idx: int, defending_idx: int, is_home: bool
    ) -> float:
        """Calculate expected goals for a team."""
        if is_home:
            strength = (
                self.home_advantage
                + self.attack_ratings[attacking_idx]
                - self.defense_ratings[defending_idx]
            )
            baseline = self.data["FTHG"].mean()
        else:
            strength = (
                self.away_adjustment
                + self.attack_ratings[attacking_idx]
                - self.defense_ratings[defending_idx]
            )
            baseline = self.data["FTAG"].mean()

        return max(self.config.lambda_bounds[0], baseline + 0.5 * strength)

    def _predict_mov(self, lambda_home: float, lambda_away: float) -> tuple:
        """Predict margin of victory."""
        raw_mov = lambda_home - lambda_away
        predicted_mov = self.mov_model.params[0] + self.mov_model.params[1] * raw_mov
        mov_std_error = np.sqrt(self.mov_model.mse_resid)
        return predicted_mov, mov_std_error

    def _fit_mov_model(self) -> None:
        """Fit margin of victory regression model."""
        home_indices = self.data["Home"].map(self.team_index).values
        away_indices = self.data["Away"].map(self.team_index).values

        home_goals = np.array(
            [
                self._calculate_expected_goals(h_idx, a_idx, is_home=True)
                for h_idx, a_idx in zip(home_indices, away_indices)
            ]
        )

        away_goals = np.array(
            [
                self._calculate_expected_goals(a_idx, h_idx, is_home=False)
                for a_idx, h_idx in zip(away_indices, home_indices)
            ]
        )

        raw_mov = home_goals - away_goals
        actual_mov = self.data["Home_MOV"].values

        # Fit weighted regression
        X = sm.add_constant(raw_mov)
        weights = self.data["Weight"].values
        self.mov_model = sm.WLS(actual_mov, X, weights=weights).fit()

    def _initialize_predictors(self) -> None:
        """Initialize prediction methods."""
        self.predictors = PredictorFactory.create_predictors(
            self.config, self.data, self.mov_model
        )

    def _validate_teams(self, home_team: str, away_team: str) -> None:
        """Validate that teams are known."""
        if home_team not in self.team_index or away_team not in self.team_index:
            raise ValueError(f"Unknown team(s): {home_team}, {away_team}")

    def _validate_method(self, method: str) -> None:
        """Validate prediction method."""
        if method not in self.predictors:
            raise ValueError(f"Unknown prediction method: {method}")
