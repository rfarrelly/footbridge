"""
Model manager for handling multiple league models.
"""

import itertools
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from analysis.betting import BettingCalculator
from backtesting import BackTestConfig, Backtester
from models.core import ModelConfig
from models.zsd_model import ZSDPoissonModel
from utils.config_manager import ConfigManager


class ModelManager:
    """Manages ZSD Poisson models for multiple leagues."""

    def __init__(self, global_config, config_dir: Path = Path("zsd_configs")):
        self.global_config = global_config
        self.config_manager = ConfigManager(config_dir)
        self.betting_calculator = BettingCalculator()
        self.models = {}

    def optimize_league_parameters(
        self, historical_data: pd.DataFrame, league: str, save_config: bool = True
    ) -> ModelConfig:
        """Optimize regularization parameters for a specific league."""
        print(f"Optimizing parameters for {league}...")

        league_data = historical_data[historical_data["League"] == league].copy()
        if len(league_data) < 100:
            print(f"Insufficient data for {league}, using default config")
            return self.config_manager.default_config

        param_combinations = self._generate_param_combinations()
        best_config = self._find_best_config(league_data, param_combinations, league)

        if save_config:
            self.config_manager.save_league_config(league, best_config)

        return best_config

    def fit_league_model(
        self,
        historical_data: pd.DataFrame,
        league: str,
        min_training_matches: int = 180,
    ) -> Optional[ZSDPoissonModel]:
        """Fit a ZSD Poisson model for a specific league."""
        print(f"Fitting ZSD model for {league}...")

        league_data = historical_data[
            (historical_data["League"] == league)
            & historical_data["Season"].isin(
                [
                    self.global_config.current_season,
                    self.global_config.previous_season,
                ]
            )
        ].copy()
        league_data = league_data.sort_values("Date").reset_index(drop=True)

        if len(league_data) < min_training_matches:
            print(f"Insufficient data for {league}: {len(league_data)} matches")
            return None

        config = self.config_manager.load_league_config(league)

        try:
            model = ZSDPoissonModel(config)
            model.fit(league_data)
            self.models[league] = model

            print(f"  Successfully fitted model for {league}")
            print(
                f"  Training data: {len(league_data)} matches, Teams: {len(model.teams)}"
            )
            print(f"  Last match in data:\r\n{league_data.tail(1)}")
            return model

        except Exception as e:
            print(f"Error fitting model for {league}: {e}")
            return None

    def predict_matches(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for upcoming matches."""
        predictions = []
        previous_season = self.global_config.previous_season

        for _, match in fixtures_df.iterrows():
            league = match.get("League")

            if self._is_relegated_or_promoted(
                teams=set(match[["Home", "Away"]]),
                league=league,
                previous_season=previous_season,
            ):
                print(
                    f"Rejected fixture: {match['Home']} v {match['Away']} - one or more teams not in previous season"
                )
                continue

            model = self.models.get(league)

            if model is None:
                print(f"No model available for {league}")
                continue

            try:
                pred_dict = self._generate_match_predictions(match, model)
                predictions.append(pred_dict)
            except Exception as e:
                print(f"Error predicting {match['Home']} vs {match['Away']}: {e}")

        return pd.DataFrame(predictions)

    def _is_relegated_or_promoted(
        self, teams: set[str], league: str, previous_season: str
    ) -> bool:
        """Checks if teams have been relegated or promoted to/from previous season"""
        previous_season_file_path = f"{self.global_config.fbref_data_dir}/{league}/{league}_{previous_season}.csv"
        if not teams.issubset(pd.read_csv(previous_season_file_path)["Home"]):
            return True
        return False

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate parameter combinations for optimization."""
        param_grid = {
            "l1_reg": [0.0, 0.001],
            "l2_reg": [0.005, 0.01, 0.05],
            "team_reg": [0.001, 0.005, 0.01],
            "decay_rate": [0.0005, 0.001],
        }

        all_combos = list(itertools.product(*param_grid.values()))
        selected_combos = random.sample(all_combos, k=min(5, len(all_combos)))

        return [dict(zip(param_grid.keys(), combo)) for combo in selected_combos]

    def _find_best_config(
        self, league_data: pd.DataFrame, param_combinations: List[Dict], league: str
    ) -> ModelConfig:
        """Find the best configuration through backtesting."""
        best_config, best_score = None, float("inf")

        for i, params in enumerate(param_combinations):
            print(f"  Testing combination {i + 1}/{len(param_combinations)}: {params}")

            # Create fresh backtester for each test - CRITICAL for independent results
            backtest_config = BackTestConfig(
                min_training_weeks=8, betting_threshold=0.02, stake_size=1.0
            )
            backtester = Backtester(backtest_config)

            try:
                # FIXED: Properly merge default config with parameter overrides
                base_config_dict = {
                    "decay_rate": self.config_manager.default_config.decay_rate,
                    "max_goals": self.config_manager.default_config.max_goals,
                    "eps": self.config_manager.default_config.eps,
                    "lambda_bounds": self.config_manager.default_config.lambda_bounds,
                    "param_bounds": self.config_manager.default_config.param_bounds,
                    "min_matches_per_team": self.config_manager.default_config.min_matches_per_team,
                    "l1_reg": self.config_manager.default_config.l1_reg,
                    "l2_reg": self.config_manager.default_config.l2_reg,
                    "team_reg": self.config_manager.default_config.team_reg,
                    "shrink_to_mean": self.config_manager.default_config.shrink_to_mean,
                    "auto_tune_regularization": self.config_manager.default_config.auto_tune_regularization,
                    "cv_folds": self.config_manager.default_config.cv_folds,
                    "n_optimization_starts": self.config_manager.default_config.n_optimization_starts,
                    "max_iter": self.config_manager.default_config.max_iter,
                }

                # Override with test parameters
                base_config_dict.update(params)
                test_config = ModelConfig(**base_config_dict)

                # DEBUG: Print the actual config being tested
                print(
                    f"    Testing config: l1={test_config.l1_reg}, l2={test_config.l2_reg}, team={test_config.team_reg}, decay={test_config.decay_rate}"
                )

                results = backtester.backtest_cross_season(
                    data=league_data,
                    model_class=lambda **kwargs: ZSDPoissonModel(test_config),
                    train_season=self.global_config.previous_season,
                    test_season=self.global_config.current_season,
                    league=league,
                )

                # Additional diagnostics
                self._diagnose_backtesting_results(results, params, i)

                score = self._calculate_optimization_score(results)

                if score < best_score:
                    best_score = score
                    best_config = test_config
                    self._save_best_results(results, league)
                    print(f"    ✅ New best config found!")
                else:
                    print(f"    Score: {score:.4f} (not best)")

            except Exception as e:
                print(f"    Error testing parameters: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"\n🏆 Best configuration for {league}:")
        if best_config:
            print(f"   l1_reg: {best_config.l1_reg}")
            print(f"   l2_reg: {best_config.l2_reg}")
            print(f"   team_reg: {best_config.team_reg}")
            print(f"   decay_rate: {best_config.decay_rate}")
            print(f"   Best score: {best_score:.4f}")

        return best_config or self.config_manager.default_config

    def _diagnose_backtesting_results(self, results, params, run_index):
        """Diagnostic checks for backtesting logic."""
        print(f"    📊 Diagnostics for run {run_index + 1}:")

        if len(results.predictions_df) > 0:
            # Check prediction variability
            avg_probs = [
                results.predictions_df["Model_Prob_H"].mean(),
                results.predictions_df["Model_Prob_D"].mean(),
                results.predictions_df["Model_Prob_A"].mean(),
            ]
            print(
                f"      Avg model probs: H={avg_probs[0]:.3f}, D={avg_probs[1]:.3f}, A={avg_probs[2]:.3f}"
            )

            # Check odds availability
            b365_available = (
                (~results.predictions_df[["B365H", "B365D", "B365A"]].isnull())
                .all(axis=1)
                .sum()
            )
            ps_available = (
                (~results.predictions_df[["PSH", "PSD", "PSA"]].isnull())
                .all(axis=1)
                .sum()
            )
            print(
                f"      Complete odds: B365={b365_available}/{len(results.predictions_df)}, PS={ps_available}/{len(results.predictions_df)}"
            )

            # Check betting details
            n_bets = len(results.betting_results)
            if n_bets > 0:
                edges = [bet.edge for bet in results.betting_results]
                odds_used = [bet.odds for bet in results.betting_results]
                print(
                    f"      Betting: {n_bets} bets, edge range [{min(edges):.4f}, {max(edges):.4f}], odds range [{min(odds_used):.2f}, {max(odds_used):.2f}]"
                )
            else:
                print(f"      Betting: No bets placed (threshold={0.02})")
        else:
            print(f"      ⚠️  No predictions generated!")

    def _calculate_optimization_score(self, results) -> float:
        """Calculate optimization score from backtest results."""
        log_loss = results.metrics.get("log_loss", 10.0)
        roi = results.metrics.get("roi_percent", -100.0)
        n_bets = results.metrics.get("total_bets", 0)

        bet_penalty = max(0, (50 - n_bets) * 0.01) if n_bets < 50 else 0
        score = log_loss - (roi / 100.0) + bet_penalty

        print(
            f"    Score: {score:.4f} (LogLoss: {log_loss:.4f}, ROI: {roi:.1f}%, Bets: {n_bets})"
        )
        return score

    def _save_best_results(self, results, league: str):
        """Save best optimization results."""
        # Create directories if they don't exist
        Path("optimisation_validation/betting_results").mkdir(
            parents=True, exist_ok=True
        )
        Path("optimisation_validation/prediction_results").mkdir(
            parents=True, exist_ok=True
        )

        # Create a fresh backtester just for saving (to avoid any state issues)
        save_backtester = Backtester(BackTestConfig())

        save_backtester.save_betting_results_to_csv(
            results,
            f"optimisation_validation/betting_results/{league}_best_betting_results.csv",
        )
        save_backtester.save_predictions_to_csv(
            results,
            f"optimisation_validation/prediction_results/{league}_best_predictions.csv",
        )

    def _generate_match_predictions(self, match, model) -> Dict:
        """Generate all prediction types for a single match."""
        home_team, away_team = match["Home"], match["Away"]

        # Get predictions for all methods (now includes 'ml' if available)
        methods = list(model.predictors.keys())  # ['poisson', 'zip', 'mov', 'ml']

        predictions = {}
        for method in methods:
            try:
                # Pass PPI and odds to ML predictor via kwargs
                predictions[method] = model.predict_match(
                    home_team,
                    away_team,
                    method=method,
                    hPPI=match.get("hPPI", 0),
                    aPPI=match.get("aPPI", 0),
                    PPIDiff=match.get("PPIDiff", 0),
                    PSH=match.get("PSH", 2.0),
                    PSD=match.get("PSD", 3.5),
                    PSA=match.get("PSA", 4.0),
                    B365H=match.get("B365H", 2.0),
                    B365D=match.get("B365D", 3.5),
                    B365A=match.get("B365A", 4.0),
                )
            except Exception as e:
                print(f"Error with {method} predictor: {e}")
                continue

        # Build prediction dictionary
        pred_dict = {
            "Date": match["Date"],
            "League": match.get("League"),
            "Home": home_team,
            "Away": away_team,
        }

        # Add predictions for each method
        method_mapping = {
            "poisson": "Poisson",
            "zip": "ZIP",
            "mov": "MOV",
            "ml": "ML",  # Add ML
        }

        for method, name in method_mapping.items():
            if method in predictions:
                pred = predictions[method]
                pred_dict.update(
                    {
                        f"{name}_Prob_H": pred.prob_home_win,
                        f"{name}_Prob_D": pred.prob_draw,
                        f"{name}_Prob_A": pred.prob_away_win,
                    }
                )

        # Add backward compatibility fields (using ZIP as primary)
        zip_pred = predictions["zip"]
        pred_dict.update(
            {
                "ZSD_Prob_H": zip_pred.prob_home_win,
                "ZSD_Prob_D": zip_pred.prob_draw,
                "ZSD_Prob_A": zip_pred.prob_away_win,
                "ZSD_Lambda_H": zip_pred.lambda_home,
                "ZSD_Lambda_A": zip_pred.lambda_away,
                "ZSD_MOV": zip_pred.mov_prediction,
                "ZSD_MOV_SE": zip_pred.mov_std_error,
                "Model_Type": zip_pred.model_type,
            }
        )

        # Add fixture columns if present
        fixture_cols = [
            "Wk",
            "PSH",
            "PSD",
            "PSA",
            "PSCH",
            "PSCD",
            "PSCA",
            "B365H",
            "B365D",
            "B365A",
            "B365CH",
            "B365CD",
            "B365CA",
            "PPI_Diff",
            "hPPI",
            "aPPI",
        ]

        for col in fixture_cols:
            if col in match.index:
                pred_dict[col] = match[col]

        # NEW: Add betting analysis if odds are available
        pred_series = pd.Series(pred_dict)
        betting_metrics = self.betting_calculator.analyze_prediction(pred_series)

        if betting_metrics:
            # Add betting metrics to prediction
            betting_dict = self.betting_calculator._metrics_to_dict(betting_metrics)
            pred_dict.update(betting_dict)
        else:
            # Add default values if no betting analysis possible
            pred_dict.update(
                {
                    "Bet_Type": None,
                    "Edge": 0.0,
                    "Model_Prob": 0.0,
                    "Market_Prob": 0.0,
                    "Sharp_Odds": 0.0,
                    "Soft_Odds": 0.0,
                    "Fair_Odds_Selected": 0.0,
                    "EV_H": 0.0,
                    "EV_D": 0.0,
                    "EV_A": 0.0,
                    "Prob_Edge_H": 0.0,
                    "Prob_Edge_D": 0.0,
                    "Prob_Edge_A": 0.0,
                    "Kelly_H": 0.0,
                    "Kelly_D": 0.0,
                    "Kelly_A": 0.0,
                }
            )

        return pred_dict
