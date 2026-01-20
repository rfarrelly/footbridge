import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from analysis.betting import BettingCalculator, BettingFilter
from models.core import ModelConfig
from models.zsd_model import ZSDPoissonModel
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from utils.odds_helpers import get_no_vig_odds_multiway

warnings.filterwarnings("ignore")


@dataclass
class BackTestConfig:
    """Configuration for backtesting."""

    min_training_weeks: int = 10
    calibration_method: str = "logistic"  # 'logistic', 'isotonic', 'none'
    betting_threshold: float = 0.02
    stake_size: float = 1.0
    max_stake_fraction: float = 0.05  # Max fraction of bankroll per bet
    ppi_filter_top_n: int = 4  # Number of top matches to filter by PPI_Diff
    min_edge: float = 0.02  # Minimum edge for betting
    min_prob: float = 0.1  # Minimum probability for betting
    max_odds: float = 10.0  # Maximum odds for betting


@dataclass
class BettingResult:
    """Result of a single betting decision."""

    match_id: str
    date: pd.Timestamp
    home_team: str
    away_team: str
    bet_type: str  # 'Home', 'Draw', 'Away'
    stake: float
    odds: float
    profit: float
    model_prob: float
    market_prob: float
    edge: float
    expected_value: float


@dataclass
class BackTestResults:
    """Comprehensive backtest results."""

    predictions_df: pd.DataFrame
    betting_results: List[BettingResult]
    metrics: Dict[str, float]
    weekly_performance: pd.DataFrame


@dataclass
class MatchPrediction:
    """Simple prediction result structure to match expected interface."""

    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    lambda_home: float
    lambda_away: float
    mov_prediction: float
    mov_std_error: float


class ModelEvaluator:
    """Handles model evaluation metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, y_prob: np.ndarray, fair_probs: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        # Ensure probabilities sum to 1
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

        # Basic metrics
        y_pred = np.argmax(y_prob, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        logloss = log_loss(y_true, y_prob, labels=[0, 1, 2])

        # Brier score (lower is better)
        y_true_onehot = np.eye(3)[y_true]
        brier = np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))

        metrics = {
            "accuracy": accuracy,
            "log_loss": logloss,
            "brier_score": brier,
            "n_predictions": len(y_true),
        }

        # Add calibration metrics if fair probabilities available
        if fair_probs is not None:
            metrics.update(
                ModelEvaluator._calculate_calibration_metrics(y_prob, fair_probs)
            )

        return metrics

    @staticmethod
    def _calculate_calibration_metrics(
        model_probs: np.ndarray, fair_probs: np.ndarray
    ) -> Dict[str, float]:
        """Calculate calibration-specific metrics."""
        # KL divergence between model and fair probabilities
        kl_div = np.mean(
            np.sum(
                model_probs * np.log(model_probs / (fair_probs + 1e-8) + 1e-8), axis=1
            )
        )

        # Mean absolute difference
        mean_abs_diff = np.mean(np.abs(model_probs - fair_probs))

        return {"kl_divergence": kl_div, "mean_abs_diff": mean_abs_diff}


class BettingSimulator:
    """Simulates betting strategy and tracks performance using BettingCalculator."""

    def __init__(self, config: BackTestConfig):
        self.config = config
        self.bankroll = 100.0  # Starting bankroll
        self.betting_history = []
        self.betting_calculator = BettingCalculator()
        self.betting_filter = BettingFilter(
            min_edge=config.min_edge, min_prob=config.min_prob, max_odds=config.max_odds
        )

    def evaluate_bet(
        self,
        prediction_row: pd.Series,
        actual_outcome: int,
    ) -> Optional[BettingResult]:
        """Evaluate whether to place a bet using BettingCalculator."""
        try:
            # Use the BettingCalculator to analyze the prediction
            betting_metrics = self.betting_calculator.analyze_prediction(prediction_row)

            if betting_metrics is None:
                return None

            # Check if it meets our betting criteria
            if betting_metrics.edge < self.config.betting_threshold:
                return None

            # Kelly criterion for stake sizing (simplified)
            kelly_fraction = betting_metrics.edge / (betting_metrics.soft_odds - 1)
            stake = min(
                self.config.stake_size,
                self.bankroll * min(kelly_fraction, self.config.max_stake_fraction),
            )

            # Calculate profit based on actual outcome
            bet_outcome_map = {"Home": 0, "Draw": 1, "Away": 2}
            bet_outcome_idx = bet_outcome_map[betting_metrics.bet_type]

            if bet_outcome_idx == actual_outcome:
                profit = stake * (betting_metrics.soft_odds - 1)
            else:
                profit = -stake

            self.bankroll += profit

            result = BettingResult(
                match_id=f"{prediction_row['Home']}_{prediction_row['Away']}_{prediction_row['Date']}",
                date=prediction_row["Date"],
                home_team=prediction_row["Home"],
                away_team=prediction_row["Away"],
                bet_type=betting_metrics.bet_type,
                stake=stake,
                odds=betting_metrics.soft_odds,
                profit=profit,
                model_prob=betting_metrics.model_prob,
                market_prob=betting_metrics.market_prob,
                edge=betting_metrics.edge,
                expected_value=betting_metrics.edge * stake,
            )

            self.betting_history.append(result)
            return result

        except Exception as e:
            print(f"Error evaluating bet: {e}")
            return None


class ProbabilityCalibrator:
    """Handles probability calibration using various methods."""

    def __init__(self, method: str = "logistic"):
        self.method = method
        self.calibrator = None
        self.scaler = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        external_features: Optional[np.ndarray] = None,
    ):
        """Fit calibration model."""
        if external_features is not None:
            X = np.column_stack([X, external_features])

        if self.method == "logistic":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.calibrator = LogisticRegression(
                multi_class="multinomial", max_iter=1000, random_state=42
            )
            self.calibrator.fit(X_scaled, y)

        self.is_fitted = True

    def calibrate(
        self, X: np.ndarray, external_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply calibration to probabilities."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")

        if self.method == "none":
            return X

        if external_features is not None:
            X = np.column_stack([X, external_features])

        if self.method == "logistic":
            X_scaled = self.scaler.transform(X)
            return self.calibrator.predict_proba(X_scaled)

        return X


class Backtester:
    """Improved backtesting framework with BettingCalculator integration."""

    def __init__(self, config: Optional[BackTestConfig] = None):
        self.config = config or BackTestConfig()
        self.evaluator = ModelEvaluator()
        self.betting_simulator = BettingSimulator(self.config)

    def backtest_cross_season(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        train_season: str,
        test_season: str,
        league: str,
        model_params: Optional[Dict] = None,
    ) -> BackTestResults:
        """Run cross-season backtest with enhanced features and BettingCalculator."""

        # Filter and prepare data
        train_data = (
            data[(data["Season"] == train_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Date"])
        )

        test_data = (
            data[(data["Season"] == test_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Wk", "Date"])
        )

        # Remove promoted and relegated teams
        teams_train_data = set(pd.concat([train_data["Home"], train_data["Away"]]))
        teams_test_data = set(pd.concat([test_data["Home"], test_data["Away"]]))
        common_teams = teams_train_data & teams_test_data

        train_data = train_data[
            train_data["Home"].isin(common_teams)
            & train_data["Away"].isin(common_teams)
        ]

        test_data = test_data[
            test_data["Home"].isin(common_teams) & test_data["Away"].isin(common_teams)
        ]

        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError(
                f"Insufficient data for {league} {train_season}/{test_season}"
            )

        all_predictions = []
        weekly_metrics = []

        print(f"Backtesting {league}: {train_season} -> {test_season}")
        print(f"Training matches: {len(train_data)}, Test matches: {len(test_data)}")

        # Get unique weeks for testing
        test_weeks = sorted(test_data["Wk"].unique())

        for week_num, week in enumerate(test_weeks, 1):
            if week_num < self.config.min_training_weeks:
                continue

            try:
                # Prepare training data (previous season + previous weeks in test season)
                past_test_data = test_data[test_data["Wk"] < week]
                combined_training = pd.concat(
                    [train_data, past_test_data], ignore_index=True
                )

                # Fit model
                model_params = model_params or {}

                if isinstance(model_class, type):
                    model = model_class(model_params.get("config", ModelConfig()))
                else:
                    model = model_class(**model_params)

                model.fit(combined_training)

                # Get current week matches
                week_matches = test_data[test_data["Wk"] == week].copy()
                week_predictions = []
                week_betting_results = []

                # Predict each match in the week
                for _, match in week_matches.iterrows():
                    prediction_result = self._predict_single_match(model, match)
                    if prediction_result:
                        week_predictions.append(prediction_result)

                        # Evaluate betting opportunity using BettingCalculator
                        betting_result = self._evaluate_betting_opportunity(
                            pd.Series(prediction_result),
                            prediction_result["Actual_Outcome"],
                        )
                        if betting_result:
                            week_betting_results.append(betting_result)

                all_predictions.extend(week_predictions)

                # Calculate weekly metrics
                if week_predictions:
                    week_df = pd.DataFrame(week_predictions)
                    week_metrics = self._calculate_weekly_metrics(week_df, week)
                    weekly_metrics.append(week_metrics)

                print(
                    f"Week {week}: {len(week_predictions)} predictions, {len(week_betting_results)} bets"
                )

            except Exception as e:
                print(f"Error in week {week}: {str(e)}")
                continue

        # Compile results
        predictions_df = pd.DataFrame(all_predictions)
        weekly_performance_df = pd.DataFrame(weekly_metrics)

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(predictions_df)

        return BackTestResults(
            predictions_df=predictions_df,
            betting_results=self.betting_simulator.betting_history,
            metrics=overall_metrics,
            weekly_performance=weekly_performance_df,
        )

    def backtest_ppi_filtered_strategy(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        train_season: str,
        test_season: str,
        league: str,
        model_params: Optional[Dict] = None,
        top_n_ppi_matches: Optional[int] = None,
    ) -> BackTestResults:
        """
        Runs a backtest strategy that filters matches based on PPI_Diff being close to zero,
        then bets on the top N matches with the lowest absolute PPI_Diff.
        """
        top_n_ppi_matches = top_n_ppi_matches or self.config.ppi_filter_top_n

        # Filter and prepare data
        train_data = (
            data[(data["Season"] == train_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Date"])
        )

        test_data = (
            data[(data["Season"] == test_season) & (data["League"] == league)]
            .copy()
            .sort_values(["Wk", "Date"])
        )

        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError(
                f"Insufficient data for {league} {train_season}/{test_season}"
            )

        all_predictions = []
        weekly_metrics = []

        print(
            f"Backtesting PPI-filtered strategy for {league}: {train_season} -> {test_season}"
        )
        print(f"Training matches: {len(train_data)}, Test matches: {len(test_data)}")
        print(
            f"Filtering for top {top_n_ppi_matches} matches by lowest absolute PPI_Diff per week."
        )

        # Get unique weeks for testing
        test_weeks = sorted(test_data["Wk"].unique())

        # Reset betting simulator for this new strategy
        self.betting_simulator = BettingSimulator(self.config)

        for week_num, week in enumerate(test_weeks, 1):
            if week_num < self.config.min_training_weeks:
                continue

            try:
                # Prepare training data (previous season + previous weeks in test season)
                past_test_data = test_data[test_data["Wk"] < week]
                combined_training = pd.concat(
                    [train_data, past_test_data], ignore_index=True
                )

                # Fit model
                model_params = model_params or {}

                if isinstance(model_class, type):
                    model = model_class(model_params.get("config", ModelConfig()))
                else:
                    model = model_class(**model_params)

                model.fit(combined_training)

                # Get current week matches
                week_matches = test_data[test_data["Wk"] == week].copy()

                # List to hold all predictions for the week before filtering
                current_week_all_predictions = []

                # Predict for all matches in the current week
                for _, match in week_matches.iterrows():
                    prediction_result = self._predict_single_match(model, match)
                    if prediction_result:
                        current_week_all_predictions.append(prediction_result)

                if not current_week_all_predictions:
                    print(f"Week {week}: No predictions generated.")
                    continue

                week_predictions_df = pd.DataFrame(current_week_all_predictions)

                # Filter by PPI_Diff: take top N matches with lowest absolute PPI_Diff
                if (
                    "PPI_Diff" in week_predictions_df.columns
                    and not week_predictions_df["PPI_Diff"].isnull().all()
                ):
                    week_predictions_df["abs_ppi_diff"] = week_predictions_df[
                        "PPI_Diff"
                    ].abs()
                    filtered_matches = week_predictions_df.sort_values(
                        "abs_ppi_diff"
                    ).head(top_n_ppi_matches)
                else:
                    print(
                        f"Warning: 'PPI_Diff' not found or all NaN for week {week}. Skipping PPI filter."
                    )
                    filtered_matches = week_predictions_df.copy()

                week_betting_results = []
                for _, prediction_row in filtered_matches.iterrows():
                    betting_result = self._evaluate_betting_opportunity(
                        prediction_row, prediction_row["Actual_Outcome"]
                    )
                    if betting_result:
                        week_betting_results.append(betting_result)

                all_predictions.extend(filtered_matches.to_dict(orient="records"))

                # Calculate weekly metrics for the FILTERED matches
                if not filtered_matches.empty:
                    week_metrics = self._calculate_weekly_metrics(
                        filtered_matches, week
                    )
                    weekly_metrics.append(week_metrics)

                print(
                    f"Week {week}: {len(filtered_matches)} predictions considered, {len(week_betting_results)} bets placed"
                )

            except Exception as e:
                print(f"Error in week {week}: {str(e)}")
                continue

        # Compile results
        predictions_df = pd.DataFrame(all_predictions)
        weekly_performance_df = pd.DataFrame(weekly_metrics)

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(predictions_df)

        return BackTestResults(
            predictions_df=predictions_df,
            betting_results=self.betting_simulator.betting_history,
            metrics=overall_metrics,
            weekly_performance=weekly_performance_df,
        )

    def _predict_single_match(self, model, match_row: pd.Series) -> Optional[Dict]:
        """Generate prediction for a single match with proper odds handling."""
        try:
            home_team = match_row["Home"]
            away_team = match_row["Away"]

            # Get predictions for all methods that BettingCalculator expects
            predictions = self._get_all_method_predictions(model, home_team, away_team)

            # Extract both sharp and soft odds
            sharp_odds = [
                match_row.get("PSH", np.nan),
                match_row.get("PSD", np.nan),
                match_row.get("PSA", np.nan),
            ]

            soft_odds = [
                match_row.get("B365H", np.nan),
                match_row.get("B365D", np.nan),
                match_row.get("B365A", np.nan),
            ]

            # Calculate fair probabilities from sharp odds
            fair_probs = self._calculate_fair_probabilities(sharp_odds)

            # Determine actual outcome
            actual_outcome = self._get_actual_outcome(match_row)

            # Extract PPI features if they exist in the match_row
            ppi_features = {}
            for col in ["PPI_A", "PPI_Diff", "hPPI", "aPPI"]:
                if col in match_row.index:
                    ppi_features[col] = match_row[col]
                else:
                    ppi_features[col] = np.nan

            return {
                "Date": match_row["Date"],
                "Season": match_row["Season"],
                "Week": match_row["Wk"],
                "Home": home_team,
                "Away": away_team,
                "FTHG": match_row["FTHG"],
                "FTAG": match_row["FTAG"],
                "Actual_Outcome": actual_outcome,
                # Primary model probabilities (for backward compatibility)
                "Model_Prob_H": predictions["zip"]["prob_home_win"],
                "Model_Prob_D": predictions["zip"]["prob_draw"],
                "Model_Prob_A": predictions["zip"]["prob_away_win"],
                # All method probabilities (expected by BettingCalculator)
                "Poisson_Prob_H": predictions["poisson"]["prob_home_win"],
                "Poisson_Prob_D": predictions["poisson"]["prob_draw"],
                "Poisson_Prob_A": predictions["poisson"]["prob_away_win"],
                "ZIP_Prob_H": predictions["zip"]["prob_home_win"],
                "ZIP_Prob_D": predictions["zip"]["prob_draw"],
                "ZIP_Prob_A": predictions["zip"]["prob_away_win"],
                "MOV_Prob_H": predictions["mov"]["prob_home_win"],
                "MOV_Prob_D": predictions["mov"]["prob_draw"],
                "MOV_Prob_A": predictions["mov"]["prob_away_win"],
                # Fair probabilities from market
                "Fair_Prob_H": fair_probs[0],
                "Fair_Prob_D": fair_probs[1],
                "Fair_Prob_A": fair_probs[2],
                # Model outputs
                "Lambda_Home": predictions["zip"]["lambda_home"],
                "Lambda_Away": predictions["zip"]["lambda_away"],
                "MOV_Prediction": predictions["zip"]["mov_prediction"],
                "MOV_StdError": predictions["zip"]["mov_std_error"],
                # Sharp odds (for fair probabilities)
                "PSH": sharp_odds[0],
                "PSD": sharp_odds[1],
                "PSA": sharp_odds[2],
                # Soft odds (for betting)
                "B365H": soft_odds[0],
                "B365D": soft_odds[1],
                "B365A": soft_odds[2],
                **ppi_features,
            }

        except Exception as e:
            print(
                f"Error predicting match {match_row.get('Home', 'Unknown')} vs {match_row.get('Away', 'Unknown')}: {e}"
            )
            return None

    def _get_all_method_predictions(
        self, model, home_team: str, away_team: str
    ) -> Dict[str, Dict]:
        """Get predictions for all methods that BettingCalculator expects."""
        try:
            # Check if teams exist in model
            if home_team not in model.team_index or away_team not in model.team_index:
                # Return default predictions for unknown teams
                default_prediction = {
                    "prob_home_win": 0.33,
                    "prob_draw": 0.33,
                    "prob_away_win": 0.34,
                    "lambda_home": 1.5,
                    "lambda_away": 1.2,
                    "mov_prediction": 0.0,
                    "mov_std_error": 1.0,
                }
                return {
                    "poisson": default_prediction.copy(),
                    "zip": default_prediction.copy(),
                    "mov": default_prediction.copy(),
                }

            # If model has predict_match method with different prediction types
            if hasattr(model, "predict_match"):
                predictions = {}
                for method in ["poisson", "zip", "mov"]:
                    try:
                        pred = model.predict_match(home_team, away_team, method=method)
                        predictions[method] = {
                            "prob_home_win": pred.prob_home_win,
                            "prob_draw": pred.prob_draw,
                            "prob_away_win": pred.prob_away_win,
                            "lambda_home": pred.lambda_home,
                            "lambda_away": pred.lambda_away,
                            "mov_prediction": pred.mov_prediction,
                            "mov_std_error": pred.mov_std_error,
                        }
                    except:
                        # Fall back to basic calculation if method not available
                        predictions[method] = self._calculate_basic_prediction(
                            model, home_team, away_team
                        )
                return predictions
            else:
                # Fall back to manual calculation for all methods
                basic_pred = self._calculate_basic_prediction(
                    model, home_team, away_team
                )
                return {
                    "poisson": basic_pred.copy(),
                    "zip": basic_pred.copy(),
                    "mov": basic_pred.copy(),
                }

        except Exception as e:
            print(f"Error in _get_all_method_predictions: {e}")
            # Return default predictions
            default_prediction = {
                "prob_home_win": 0.33,
                "prob_draw": 0.33,
                "prob_away_win": 0.34,
                "lambda_home": 1.5,
                "lambda_away": 1.2,
                "mov_prediction": 0.0,
                "mov_std_error": 1.0,
            }
            return {
                "poisson": default_prediction.copy(),
                "zip": default_prediction.copy(),
                "mov": default_prediction.copy(),
            }

    def _calculate_basic_prediction(
        self, model, home_team: str, away_team: str
    ) -> Dict:
        """Calculate basic prediction when advanced methods aren't available."""
        try:
            # Get team indices
            home_idx = model.team_index[home_team]
            away_idx = model.team_index[away_team]

            # Calculate expected goals using model parameters
            home_strength = (
                model.home_advantage
                + model.attack_ratings[home_idx]
                - model.defense_ratings[away_idx]
            )

            away_strength = (
                model.away_adjustment
                + model.attack_ratings[away_idx]
                - model.defense_ratings[home_idx]
            )

            # Convert strength to expected goals (simplified approach)
            lambda_home = max(0.1, 1.5 + 0.5 * home_strength)
            lambda_away = max(0.1, 1.2 + 0.5 * away_strength)

            # Calculate match outcome probabilities using Poisson distribution
            max_goals = getattr(model.config, "max_goals", 15)
            prob_matrix = np.zeros((max_goals + 1, max_goals + 1))

            for h_goals in range(max_goals + 1):
                for a_goals in range(max_goals + 1):
                    prob_matrix[h_goals, a_goals] = poisson.pmf(
                        h_goals, lambda_home
                    ) * poisson.pmf(a_goals, lambda_away)

            # Calculate outcome probabilities
            prob_home_win = np.sum(
                [
                    prob_matrix[h, a]
                    for h in range(max_goals + 1)
                    for a in range(max_goals + 1)
                    if h > a
                ]
            )
            prob_draw = np.sum([prob_matrix[h, h] for h in range(max_goals + 1)])
            prob_away_win = np.sum(
                [
                    prob_matrix[h, a]
                    for h in range(max_goals + 1)
                    for a in range(max_goals + 1)
                    if h < a
                ]
            )

            # Normalize probabilities
            total_prob = prob_home_win + prob_draw + prob_away_win
            if total_prob > 0:
                prob_home_win /= total_prob
                prob_draw /= total_prob
                prob_away_win /= total_prob

            # Calculate margin of victory prediction and standard error
            mov_prediction = lambda_home - lambda_away
            mov_std_error = np.sqrt(lambda_home + lambda_away)

            return {
                "prob_home_win": prob_home_win,
                "prob_draw": prob_draw,
                "prob_away_win": prob_away_win,
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "mov_prediction": mov_prediction,
                "mov_std_error": mov_std_error,
            }

        except Exception as e:
            print(f"Error in _calculate_basic_prediction: {e}")
            # Return default prediction
            return {
                "prob_home_win": 0.33,
                "prob_draw": 0.33,
                "prob_away_win": 0.34,
                "lambda_home": 1.5,
                "lambda_away": 1.2,
                "mov_prediction": 0.0,
                "mov_std_error": 1.0,
            }

    def _calculate_fair_probabilities(self, odds: List[float]) -> List[float]:
        """Calculate fair (no-vig) probabilities from bookmaker odds."""
        try:
            if any(np.isnan(odds)) or any(o <= 1.0 for o in odds):
                return [np.nan, np.nan, np.nan]

            fair_odds_tuple = get_no_vig_odds_multiway(odds)
            fair_probs = [1 / o for o in fair_odds_tuple]
            return fair_probs

        except Exception as e:
            print(f"Error calculating fair probabilities: {e}")
            return [np.nan, np.nan, np.nan]

    def _get_actual_outcome(self, match_row: pd.Series) -> int:
        """Get actual match outcome (0=Home, 1=Draw, 2=Away)."""
        home_goals = match_row["FTHG"]
        away_goals = match_row["FTAG"]

        if home_goals > away_goals:
            return 0  # Home win
        elif home_goals == away_goals:
            return 1  # Draw
        else:
            return 2  # Away win

    def _evaluate_betting_opportunity(
        self, prediction_row: pd.Series, actual_outcome: int
    ) -> Optional[BettingResult]:
        """Evaluate betting opportunity using BettingCalculator."""
        return self.betting_simulator.evaluate_bet(prediction_row, actual_outcome)

    def _calculate_weekly_metrics(self, week_df: pd.DataFrame, week: int) -> Dict:
        """Calculate metrics for a single week."""
        y_true = week_df["Actual_Outcome"].values
        y_prob = week_df[["Model_Prob_H", "Model_Prob_D", "Model_Prob_A"]].values

        # Handle missing fair probabilities gracefully
        if all(
            col in week_df.columns
            for col in ["Fair_Prob_H", "Fair_Prob_D", "Fair_Prob_A"]
        ):
            fair_probs = week_df[["Fair_Prob_H", "Fair_Prob_D", "Fair_Prob_A"]].values
        else:
            fair_probs = None

        metrics = self.evaluator.calculate_metrics(y_true, y_prob, fair_probs)
        metrics["Week"] = week
        metrics["n_matches"] = len(week_df)

        return metrics

    def _calculate_overall_metrics(
        self, predictions_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate overall backtest metrics."""
        if len(predictions_df) == 0:
            return {}

        # Model performance metrics
        y_true = predictions_df["Actual_Outcome"].values
        y_prob = predictions_df[["Model_Prob_H", "Model_Prob_D", "Model_Prob_A"]].values

        # Handle missing fair probabilities gracefully
        if all(
            col in predictions_df.columns
            for col in ["Fair_Prob_H", "Fair_Prob_D", "Fair_Prob_A"]
        ):
            fair_probs = predictions_df[
                ["Fair_Prob_H", "Fair_Prob_D", "Fair_Prob_A"]
            ].values
        else:
            fair_probs = None

        model_metrics = self.evaluator.calculate_metrics(y_true, y_prob, fair_probs)

        # Betting performance metrics
        betting_metrics = self._calculate_betting_metrics()

        # Combine all metrics
        overall_metrics = {**model_metrics, **betting_metrics}

        return overall_metrics

    def _calculate_betting_metrics(self) -> Dict[str, float]:
        """Calculate betting-specific performance metrics."""
        betting_history = self.betting_simulator.betting_history

        if not betting_history:
            return {
                "total_bets": 0,
                "total_profit": 0.0,
                "roi_percent": 0.0,
                "win_rate": 0.0,
                "avg_edge": 0.0,
                "final_bankroll": self.betting_simulator.bankroll,
            }

        profits = [bet.profit for bet in betting_history]
        stakes = [bet.stake for bet in betting_history]
        edges = [bet.edge for bet in betting_history]
        expected_values = [bet.expected_value for bet in betting_history]
        wins = [bet.profit > 0 for bet in betting_history]

        return {
            "total_bets": len(betting_history),
            "total_profit": sum(profits),
            "total_staked": sum(stakes),
            "roi_percent": 100 * sum(profits) / sum(stakes) if sum(stakes) > 0 else 0.0,
            "win_rate": 100 * sum(wins) / len(wins) if wins else 0.0,
            "avg_edge": np.mean(edges),
            "avg_expected_value": np.mean(expected_values),
            "profit_per_bet": np.mean(profits),
            "final_bankroll": self.betting_simulator.bankroll,
            "max_profit": max(profits) if profits else 0.0,
            "max_loss": min(profits) if profits else 0.0,
        }

    def backtest_with_calibration(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        train_season: str,
        test_season: str,
        league: str,
        calibration_features: Optional[List[str]] = None,
        test_split: float = 0.2,
    ) -> Dict[str, BackTestResults]:
        """Run backtest with and without calibration for comparison."""

        # Run base backtest
        base_results = self.backtest_cross_season(
            data, model_class, train_season, test_season, league
        )

        results = {"base": base_results}

        # If we have calibration features, run calibrated version
        if calibration_features and len(base_results.predictions_df) > 0:
            calibrated_results = self._run_calibrated_backtest(
                base_results.predictions_df, calibration_features, test_split
            )
            results["calibrated"] = calibrated_results

        return results

    def _run_calibrated_backtest(
        self,
        predictions_df: pd.DataFrame,
        calibration_features: List[str],
        test_split: float,
    ) -> BackTestResults:
        """Run backtest with calibrated probabilities."""

        # Filter calibration features to only include those present in predictions_df
        available_calibration_features = [
            f for f in calibration_features if f in predictions_df.columns
        ]
        if not available_calibration_features:
            print(
                "Warning: No specified calibration features found in predictions_df. Running without external features for calibration."
            )

        # Split data for calibration
        n_calibration = int(len(predictions_df) * (1 - test_split))
        cal_df = predictions_df.iloc[:n_calibration].copy()
        test_df = predictions_df.iloc[n_calibration:].copy()

        # Prepare calibration data
        X_cal = cal_df[["Model_Prob_H", "Model_Prob_D", "Model_Prob_A"]].values
        y_cal = cal_df["Actual_Outcome"].values

        external_cal = (
            cal_df[available_calibration_features].values
            if available_calibration_features
            else None
        )

        # Fit calibrator
        calibrator = ProbabilityCalibrator(method=self.config.calibration_method)
        calibrator.fit(X_cal, y_cal, external_cal)

        # Apply calibration to test set
        X_test = test_df[["Model_Prob_H", "Model_Prob_D", "Model_Prob_A"]].values
        external_test = (
            test_df[available_calibration_features].values
            if available_calibration_features
            else None
        )
        calibrated_probs = calibrator.calibrate(X_test, external_test)

        # Update test dataframe with calibrated probabilities
        test_df = test_df.copy()
        test_df["Model_Prob_H"] = calibrated_probs[:, 0]
        test_df["Model_Prob_D"] = calibrated_probs[:, 1]
        test_df["Model_Prob_A"] = calibrated_probs[:, 2]

        # Recalculate metrics with calibrated probabilities
        y_true = test_df["Actual_Outcome"].values
        y_prob_cal = calibrated_probs
        fair_probs = test_df[["Fair_Prob_H", "Fair_Prob_D", "Fair_Prob_A"]].values

        cal_metrics = self.evaluator.calculate_metrics(y_true, y_prob_cal, fair_probs)

        # Simulate betting with calibrated probabilities using BettingCalculator
        cal_betting_simulator = BettingSimulator(self.config)
        cal_betting_results = []

        for _, row in test_df.iterrows():
            betting_result = cal_betting_simulator.evaluate_bet(
                row, row["Actual_Outcome"]
            )
            if betting_result:
                cal_betting_results.append(betting_result)

        # Calculate calibrated betting metrics
        cal_betting_metrics = self._calculate_betting_metrics_from_simulator(
            cal_betting_simulator
        )
        cal_metrics.update(cal_betting_metrics)

        return BackTestResults(
            predictions_df=test_df,
            betting_results=cal_betting_simulator.betting_history,
            metrics=cal_metrics,
            weekly_performance=pd.DataFrame(),  # Could implement if needed
        )

    def _calculate_betting_metrics_from_simulator(
        self, simulator: BettingSimulator
    ) -> Dict[str, float]:
        """Calculate betting metrics from a specific simulator instance."""
        betting_history = simulator.betting_history

        if not betting_history:
            return {
                "total_bets": 0,
                "total_profit": 0.0,
                "roi_percent": 0.0,
                "win_rate": 0.0,
                "avg_edge": 0.0,
                "final_bankroll": simulator.bankroll,
            }

        profits = [bet.profit for bet in betting_history]
        stakes = [bet.stake for bet in betting_history]
        edges = [bet.edge for bet in betting_history]
        expected_values = [bet.expected_value for bet in betting_history]
        wins = [bet.profit > 0 for bet in betting_history]

        return {
            "total_bets": len(betting_history),
            "total_profit": sum(profits),
            "total_staked": sum(stakes),
            "roi_percent": 100 * sum(profits) / sum(stakes) if sum(stakes) > 0 else 0.0,
            "win_rate": 100 * sum(wins) / len(wins) if wins else 0.0,
            "avg_edge": np.mean(edges),
            "avg_expected_value": np.mean(expected_values),
            "profit_per_bet": np.mean(profits),
            "final_bankroll": simulator.bankroll,
            "max_profit": max(profits) if profits else 0.0,
            "max_loss": min(profits) if profits else 0.0,
        }

    def print_results_summary(self, results: Dict[str, BackTestResults]):
        """Print a formatted summary of backtest results."""
        print(f"{'=' * 60}\r\nBACKTEST RESULTS SUMMARY\r\n{'=' * 60}\r\n")

        for method, result in results.items():
            print(f"\n{method.upper()} MODEL RESULTS:")
            print("-" * 30)

            metrics = result.metrics
            print(f"Prediction Performance:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"  Log Loss: {metrics.get('log_loss', 0):.4f}")
            print(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
            print(f"  Total Predictions: {metrics.get('n_predictions', 0)}")

            print(f"\nBetting Performance:")
            print(f"  Total Bets: {metrics.get('total_bets', 0)}")
            print(f"  Total Profit: ${metrics.get('total_profit', 0):.2f}")
            print(f"  ROI: {metrics.get('roi_percent', 0):.2f}%")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"  Average Edge: {metrics.get('avg_edge', 0):.3f}")
            print(
                f"  Average Expected Value: ${metrics.get('avg_expected_value', 0):.2f}"
            )
            print(f"  Final Bankroll: ${metrics.get('final_bankroll', 100):.2f}")

    def save_betting_results_to_csv(
        self, results: BackTestResults, filename: str = "betting_results.csv"
    ):
        """Saves the detailed betting results to a CSV file."""
        if not results.betting_results:
            print(f"No betting results to save for {filename}.")
            return

        betting_df = pd.DataFrame([bet.__dict__ for bet in results.betting_results])
        betting_df.to_csv(filename, index=False)
        print(f"Betting results saved to {filename}")

    def save_predictions_to_csv(
        self, results: BackTestResults, filename: str = "predictions.csv"
    ):
        """Saves the detailed predictions dataframe to a CSV file."""
        if results.predictions_df.empty:
            print(f"No predictions to save for {filename}.")
            return
        results.predictions_df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")


# Updated example usage functions to work with the new model structure
def run_backtest_example():
    """Example of how to run the improved backtesting with your data."""

    # Load your data
    matches = pd.read_csv("historical_ppi_and_odds.csv", dtype={"Wk": int})

    # Configure the model
    model_config = ModelConfig(
        decay_rate=0.001,
        max_goals=15,
        min_matches_per_team=5,
        l1_reg=0.0,
        l2_reg=0.01,
        team_reg=0.005,
        auto_tune_regularization=False,
    )

    # Configure the backtesting with BettingCalculator integration
    backtest_config = BackTestConfig(
        min_training_weeks=10,
        calibration_method="logistic",
        betting_threshold=0.025,
        stake_size=1.0,
        ppi_filter_top_n=4,
        min_edge=0.02,
        min_prob=0.1,
        max_odds=10.0,
    )

    # Create backtester
    backtester = Backtester(backtest_config)

    # Updated model factory function
    def ZSDPoissonModelWithConfig(**kwargs):
        config = kwargs.get("config", model_config)
        return ZSDPoissonModel(config)

    # Run base backtest
    print(
        f"{'=' * 60}\r\nRunning Enhanced Backtest with BettingCalculator...\r\n{'=' * 60}\r\n"
    )
    try:
        base_results = backtester.backtest_cross_season(
            data=matches,
            model_class=ZSDPoissonModelWithConfig,
            train_season="2022-2023",
            test_season="2023-2024",
            league="2-Bundesliga",
            model_params={"config": model_config},
        )

        print(f"{'=' * 60}\r\nENHANCED BACKTEST RESULTS\r\n{'=' * 60}\r\n")

        metrics = base_results.metrics
        print(f"\nModel Performance:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"  Log Loss: {metrics.get('log_loss', 0):.4f}")
        print(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
        print(f"  Total Predictions: {metrics.get('n_predictions', 0)}")

        print(f"\nBetting Performance (with BettingCalculator):")
        print(f"  Total Bets: {metrics.get('total_bets', 0)}")
        print(f"  Total Profit: ${metrics.get('total_profit', 0):.2f}")
        print(f"  ROI: {metrics.get('roi_percent', 0):.2f}%")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Average Edge: {metrics.get('avg_edge', 0):.3f}")
        print(f"  Average Expected Value: ${metrics.get('avg_expected_value', 0):.2f}")
        print(f"  Final Bankroll: ${metrics.get('final_bankroll', 100):.2f}")

        # Show some sample predictions
        if not base_results.predictions_df.empty:
            print(f"\nSample Predictions (Enhanced Model):")
            display_cols = [
                "Date",
                "Home",
                "Away",
                "Model_Prob_H",
                "Model_Prob_D",
                "Model_Prob_A",
                "Fair_Prob_H",
                "Fair_Prob_D",
                "Fair_Prob_A",
                "Actual_Outcome",
            ]
            print(base_results.predictions_df[display_cols].head(10))

        # Show betting results if any
        if base_results.betting_results:
            betting_df = pd.DataFrame(
                [
                    {
                        "Date": bet.date,
                        "Match": f"{bet.home_team} vs {bet.away_team}",
                        "Bet": bet.bet_type,
                        "Stake": bet.stake,
                        "Odds": bet.odds,
                        "Profit": bet.profit,
                        "Edge": bet.edge,
                        "Expected_Value": bet.expected_value,
                    }
                    for bet in base_results.betting_results[:10]
                ]
            )

            print(f"\nSample Bets (Enhanced Model with BettingCalculator):")
            print(betting_df)

        # Save results to CSV
        backtester.save_betting_results_to_csv(
            base_results, "enhanced_betting_results.csv"
        )
        backtester.save_predictions_to_csv(base_results, "enhanced_predictions.csv")

        return base_results

    except Exception as e:
        print(f"Error running enhanced backtest: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_calibration_example():
    """Example of running backtest with calibration using PPI features."""

    matches = pd.read_csv("historical_ppi_and_odds.csv", dtype={"Wk": int})

    # Check if PPI columns exist in the original data
    ppi_cols = ["PPI_A", "PPI_Diff", "hPPI", "aPPI"]
    available_ppi_cols = [col for col in ppi_cols if col in matches.columns]

    if not available_ppi_cols:
        print("No PPI columns found for calibration in the original data.")
        return

    model_config = ModelConfig(
        decay_rate=0.001,
        l1_reg=0.0,
        l2_reg=0.01,
        team_reg=0.005,
        auto_tune_regularization=False,
    )

    backtest_config = BackTestConfig(
        min_training_weeks=10,
        calibration_method="logistic",
        betting_threshold=0.02,
        min_edge=0.02,
        min_prob=0.1,
        max_odds=10.0,
    )

    backtester = Backtester(backtest_config)

    def ZSDPoissonModelWithConfig(**kwargs):
        config = kwargs.get("config", model_config)
        return ZSDPoissonModel(config)

    print(
        f"{'=' * 60}\r\nRunning Calibration Example with BettingCalculator...\r\n{'=' * 60}\r\n"
    )

    try:
        # Run backtest with calibration
        results_calibrated = backtester.backtest_with_calibration(
            data=matches,
            model_class=ZSDPoissonModelWithConfig,
            train_season="2023-2024",
            test_season="2024-2025",
            league="Premier-League",
            calibration_features=available_ppi_cols,
            test_split=0.2,
        )

        # Print comparison
        backtester.print_results_summary(results_calibrated)

        # Save results to CSV
        if "calibrated" in results_calibrated:
            backtester.save_betting_results_to_csv(
                results_calibrated["calibrated"], "calibrated_betting_results.csv"
            )
            backtester.save_predictions_to_csv(
                results_calibrated["calibrated"], "calibrated_predictions.csv"
            )

        return results_calibrated

    except Exception as e:
        print(f"Error running calibration example: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_ppi_filtered_example():
    """Example: running backtest with PPI-filtered strategy and BettingCalculator."""

    matches = pd.read_csv("historical_ppi_and_odds.csv", dtype={"Wk": int})

    # Ensure 'PPI_Diff' column exists in the data
    if "PPI_Diff" not in matches.columns:
        print(
            "Error: 'PPI_Diff' column not found in the historical data. Cannot run PPI-filtered strategy."
        )
        return

    model_config = ModelConfig(
        decay_rate=0.001,
        l1_reg=0.0,
        l2_reg=0.01,
        team_reg=0.005,
        auto_tune_regularization=False,
    )

    backtest_config = BackTestConfig(
        min_training_weeks=10,
        calibration_method="logistic",
        betting_threshold=0.02,
        stake_size=1.0,
        ppi_filter_top_n=4,
        min_edge=0.02,
        min_prob=0.1,
        max_odds=10.0,
    )

    backtester = Backtester(backtest_config)

    def ZSDPoissonModelWithConfig(**kwargs):
        config = kwargs.get("config", model_config)
        return ZSDPoissonModel(config)

    print(
        f"{'=' * 60}\r\nRunning PPI-Filtered Strategy with BettingCalculator...\r\n{'=' * 60}\r\n"
    )

    try:
        ppi_results = backtester.backtest_ppi_filtered_strategy(
            data=matches,
            model_class=ZSDPoissonModelWithConfig,
            train_season="2023-2024",
            test_season="2024-2025",
            league="Premier-League",
            model_params={"config": model_config},
            top_n_ppi_matches=4,
        )

        print(f"{'=' * 60}\r\nPPI-FILTERED STRATEGY RESULTS\r\n{'=' * 60}\r\n")
        metrics = ppi_results.metrics
        print(f"\nModel Performance (for filtered matches):")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"  Log Loss: {metrics.get('log_loss', 0):.4f}")
        print(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
        print(f"  Total Predictions: {metrics.get('n_predictions', 0)}")

        print(f"\nBetting Performance (PPI-Filtered with BettingCalculator):")
        print(f"  Total Bets: {metrics.get('total_bets', 0)}")
        print(f"  Total Profit: ${metrics.get('total_profit', 0):.2f}")
        print(f"  ROI: {metrics.get('roi_percent', 0):.2f}%")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Average Edge: {metrics.get('avg_edge', 0):.3f}")
        print(f"  Average Expected Value: ${metrics.get('avg_expected_value', 0):.2f}")
        print(f"  Final Bankroll: ${metrics.get('final_bankroll', 100):.2f}")

        # Save results to CSV
        backtester.save_betting_results_to_csv(
            ppi_results, "ppi_filtered_betting_results.csv"
        )
        backtester.save_predictions_to_csv(ppi_results, "ppi_filtered_predictions.csv")

        return ppi_results

    except Exception as e:
        print(f"Error running PPI-filtered example: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Running enhanced backtest example with BettingCalculator integration...")
    run_backtest_example()

    print(f"{'=' * 60}\r\nRunning calibration example...\r\n{'=' * 60}\r\n")
    run_calibration_example()

    print(f"{'=' * 60}\r\nRunning PPI-filtered strategy example...\r\n{'=' * 60}\r\n")
    run_ppi_filtered_example()
