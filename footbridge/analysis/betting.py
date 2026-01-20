"""
Betting analysis and edge calculation utilities.
"""

from typing import Dict, List, Optional

import pandas as pd
from models.core import BettingMetrics
from utils.odds_helpers import get_no_vig_odds_multiway


class BettingCalculator:
    """Handles betting analysis and edge calculations."""

    def __init__(self):
        self.bet_types = ["Home", "Draw", "Away"]

    def analyze_prediction(self, prediction_row: pd.Series) -> Optional[BettingMetrics]:
        """Analyze a single prediction for betting opportunities."""
        odds_cols = ["PSH", "PSD", "PSA"]
        if any(pd.isna(prediction_row.get(col)) for col in odds_cols):
            return None

        try:
            return self._calculate_betting_metrics(prediction_row)
        except Exception as e:
            print(f"Error in betting analysis: {e}")
            return None

    def find_betting_candidates(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Find betting candidates from predictions dataframe."""
        if len(predictions_df) == 0:
            return pd.DataFrame()

        candidates = []
        for _, pred in predictions_df.iterrows():
            metrics = self.analyze_prediction(pred)
            if metrics:
                candidate_dict = pred.to_dict()
                candidate_dict.update(self._metrics_to_dict(metrics))
                candidates.append(candidate_dict)

        return pd.DataFrame(candidates)

    def _calculate_betting_metrics(self, pred: pd.Series) -> BettingMetrics:
        """Calculate comprehensive betting metrics."""
        # Extract odds
        sharp_odds = [pred["PSH"], pred["PSD"], pred["PSA"]]
        soft_odds = [pred["B365H"], pred["B365D"], pred["B365A"]]

        # Get model probabilities
        prob_sets = self._extract_model_probabilities(pred)

        # Calculate no-vig and combined probabilities
        no_vig_sharp_odds = get_no_vig_odds_multiway(sharp_odds)
        no_vig_sharp_probs = [1 / odd for odd in no_vig_sharp_odds]
        model_avg_probs = self._average_model_probabilities(prob_sets)
        weighted_probs = self._combine_probabilities(
            model_avg_probs, no_vig_sharp_probs
        )

        # Calculate edges
        edges = self._calculate_all_edges(weighted_probs, soft_odds)
        best_bet_idx = self._find_best_bet(edges["expected_values"])

        return BettingMetrics(
            bet_type=self.bet_types[best_bet_idx],
            edge=edges["expected_values"][best_bet_idx],
            model_prob=weighted_probs[best_bet_idx],
            market_prob=no_vig_sharp_probs[best_bet_idx],
            sharp_odds=sharp_odds[best_bet_idx],
            soft_odds=soft_odds[best_bet_idx],
            fair_odds=1 / weighted_probs[best_bet_idx],
            expected_values=edges["expected_values"],
            kelly_edges=edges["kelly_edges"],
            probability_edges=edges["probability_edges"],
        )

    def _extract_model_probabilities(self, pred: pd.Series) -> Dict[str, List[float]]:
        """Extract model probabilities from prediction row."""
        prob_sets = {
            "poisson": [
                pred["Poisson_Prob_H"],
                pred["Poisson_Prob_D"],
                pred["Poisson_Prob_A"],
            ],
            "zip": [pred["ZIP_Prob_H"], pred["ZIP_Prob_D"], pred["ZIP_Prob_A"]],
            "mov": [pred["MOV_Prob_H"], pred["MOV_Prob_D"], pred["MOV_Prob_A"]],
        }

        # Add ML if available
        if all(col in pred.index for col in ["ML_Prob_H", "ML_Prob_D", "ML_Prob_A"]):
            prob_sets["ml"] = [pred["ML_Prob_H"], pred["ML_Prob_D"], pred["ML_Prob_A"]]

        return prob_sets

    def _average_model_probabilities(
        self, prob_sets: Dict[str, List[float]]
    ) -> List[float]:
        """Calculate average probabilities across models."""
        n_models = len(prob_sets)
        return [
            sum(prob_sets[model][i] for model in prob_sets) / n_models for i in range(3)
        ]

    def _combine_probabilities(
        self,
        model_probs: List[float],
        market_probs: List[float],
        model_weight: float = 0.1,
    ) -> List[float]:
        """Combine model and market probabilities."""
        market_weight = 1 - model_weight
        return [
            model_probs[i] * model_weight + market_probs[i] * market_weight
            for i in range(3)
        ]

    def _calculate_all_edges(
        self, weighted_probs: List[float], soft_odds: List[float]
    ) -> Dict[str, List[float]]:
        """Calculate all edge types."""
        probability_edges = [weighted_probs[i] - 1 / soft_odds[i] for i in range(3)]
        expected_values = [(weighted_probs[i] * soft_odds[i]) - 1 for i in range(3)]
        kelly_edges = [
            (
                (weighted_probs[i] * soft_odds[i] - 1) / (soft_odds[i] - 1)
                if soft_odds[i] > 1
                else 0
            )
            for i in range(3)
        ]

        return {
            "probability_edges": probability_edges,
            "expected_values": expected_values,
            "kelly_edges": kelly_edges,
        }

    def _find_best_bet(self, expected_values: List[float]) -> int:
        """Find index of best betting opportunity."""
        return expected_values.index(max(expected_values))

    def _metrics_to_dict(self, metrics: BettingMetrics) -> Dict:
        """Convert BettingMetrics to dictionary for DataFrame construction."""
        base_dict = {
            "Bet_Type": metrics.bet_type,
            "Edge": metrics.edge,
            "Model_Prob": metrics.model_prob,
            "Market_Prob": metrics.market_prob,
            "Sharp_Odds": metrics.sharp_odds,
            "Soft_Odds": metrics.soft_odds,
            "Fair_Odds_Selected": metrics.fair_odds,
        }

        # Add detailed edge calculations
        for i, bet_type in enumerate(self.bet_types):
            suffix = bet_type[0]  # H, D, A
            base_dict.update(
                {
                    f"EV_{suffix}": metrics.expected_values[i],
                    f"Prob_Edge_{suffix}": metrics.probability_edges[i],
                    f"Kelly_{suffix}": metrics.kelly_edges[i],
                }
            )

        return base_dict


class BettingFilter:
    """Filters betting candidates based on various criteria."""

    def __init__(
        self, min_edge: float = 0.02, min_prob: float = 0.1, max_odds: float = 10.0
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds

    def filter_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to betting candidates."""
        if len(candidates_df) == 0:
            return candidates_df

        filtered = candidates_df[
            (candidates_df["Edge"] >= self.min_edge)
            & (candidates_df["Model_Prob"] >= self.min_prob)
            & (candidates_df["Soft_Odds"] <= self.max_odds)
        ].copy()

        return filtered.sort_values("Edge", ascending=False)
