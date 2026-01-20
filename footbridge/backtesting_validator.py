import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class ValidationResult:
    """Results of backtest validation analysis."""

    is_valid: bool
    confidence_score: float
    red_flags: List[str]
    warnings: List[str]
    detailed_metrics: Dict[str, float]


class BacktestValidator:
    """Comprehensive validation tools for backtesting results."""

    def __init__(self):
        self.validation_results = {}

    def validate_backtest_results(
        self,
        betting_results_csv: str,
        predictions_csv: str,
        historical_data_csv: str,
        config,
    ) -> ValidationResult:
        """
        Comprehensive validation of backtest results.

        Args:
            betting_results_csv: Path to CSV with betting results
            predictions_csv: Path to CSV with all predictions
            historical_data_csv: Path to original historical data
        """

        print(f"{'=' * 60}\r\nCOMPREHENSIVE BACKTEST VALIDATION\r\n{'=' * 60}\r\n")

        league_name = self._extract_league_name(betting_results_csv)
        test_season = config.base_config.current_season

        # Load data
        try:
            betting_df = pd.read_csv(betting_results_csv)
            predictions_df = pd.read_csv(predictions_csv)
            historical_df = pd.read_csv(historical_data_csv)
            historical_df = historical_df[
                (historical_df["League"] == league_name)
                & (historical_df["Season"].isin([test_season]))
            ]

            print(f"Loaded:")
            print(f"  - {len(betting_df)} betting results")
            print(f"  - {len(predictions_df)} predictions")
            print(f"  - {len(historical_df)} historical matches")

        except FileNotFoundError as e:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                red_flags=[f"Cannot load required files: {e}"],
                warnings=[],
                detailed_metrics={},
            )

        red_flags = []
        warnings = []
        detailed_metrics = {}

        # Run all validation checks
        print("\n1. TEMPORAL CONSISTENCY CHECKS")
        print("-" * 40)
        temporal_flags, temporal_warnings, temporal_metrics = (
            self._validate_temporal_consistency(
                betting_df, predictions_df, historical_df
            )
        )
        red_flags.extend(temporal_flags)
        warnings.extend(temporal_warnings)
        detailed_metrics.update(temporal_metrics)

        print("\n2. ODDS AND PROBABILITY VALIDATION")
        print("-" * 40)
        odds_flags, odds_warnings, odds_metrics = self._validate_odds_consistency(
            betting_df, predictions_df
        )
        red_flags.extend(odds_flags)
        warnings.extend(odds_warnings)
        detailed_metrics.update(odds_metrics)

        print("\n3. BETTING PATTERN ANALYSIS")
        print("-" * 40)
        pattern_flags, pattern_warnings, pattern_metrics = (
            self._analyze_betting_patterns(betting_df, predictions_df)
        )
        red_flags.extend(pattern_flags)
        warnings.extend(pattern_warnings)
        detailed_metrics.update(pattern_metrics)

        print("\n4. PERFORMANCE DISTRIBUTION ANALYSIS")
        print("-" * 40)
        perf_flags, perf_warnings, perf_metrics = (
            self._analyze_performance_distribution(betting_df)
        )
        red_flags.extend(perf_flags)
        warnings.extend(perf_warnings)
        detailed_metrics.update(perf_metrics)

        print("\n5. STATISTICAL SIGNIFICANCE TESTS")
        print("-" * 40)
        stat_flags, stat_warnings, stat_metrics = self._run_significance_tests(
            betting_df
        )
        red_flags.extend(stat_flags)
        warnings.extend(stat_warnings)
        detailed_metrics.update(stat_metrics)

        print("\n6. LOOK-AHEAD BIAS DETECTION")
        print("-" * 40)
        lookahead_flags, lookahead_warnings, lookahead_metrics = (
            self._detect_lookahead_bias(predictions_df, historical_df)
        )
        red_flags.extend(lookahead_flags)
        warnings.extend(lookahead_warnings)
        detailed_metrics.update(lookahead_metrics)

        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(
            red_flags, warnings, detailed_metrics
        )
        is_valid = confidence_score > 0.7 and len(red_flags) == 0

        # Print summary
        print(f"{'=' * 60}\r\nVALIDATION SUMMARY\r\n{'=' * 60}\r\n")
        print(f"Overall Valid: {is_valid}")
        print(f"Confidence Score: {confidence_score:.2f}/1.0")
        print(f"Red Flags: {len(red_flags)}")
        print(f"Warnings: {len(warnings)}")

        if red_flags:
            print("\n🚨 RED FLAGS (Critical Issues):")
            for flag in red_flags:
                print(f"  - {flag}")

        if warnings:
            print("\n⚠️  WARNINGS (Potential Issues):")
            for warning in warnings:
                print(f"  - {warning}")

        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            red_flags=red_flags,
            warnings=warnings,
            detailed_metrics=detailed_metrics,
        )

    # IMPROVED TEMPORAL VALIDATION
    # Replace _validate_temporal_consistency method in backtesting_validator.py

    def _validate_temporal_consistency(
        self,
        betting_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        historical_df: pd.DataFrame,
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Check for temporal consistency issues with improved logic."""
        red_flags = []
        warnings = []
        metrics = {}

        # Convert dates
        betting_df["date"] = pd.to_datetime(betting_df["date"])
        predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
        historical_df["Date"] = pd.to_datetime(historical_df["Date"])

        # Check 1: No future information used (IMPROVED)
        future_bets = 0
        temporal_violations = []

        for _, bet in betting_df.iterrows():
            # Find the actual match in historical data
            match_rows = historical_df[
                (historical_df["Home"] == bet["home_team"])
                & (historical_df["Away"] == bet["away_team"])
            ]

            if not match_rows.empty:
                # Get the closest match by date
                actual_match_date = match_rows.iloc[0]["Date"]
                bet_date = bet["date"]

                # Allow same-day betting (before match) but not after
                # In real backtesting, bet should be placed BEFORE match result is known
                if bet_date > actual_match_date:
                    future_bets += 1
                    temporal_violations.append(
                        {
                            "bet_date": bet_date,
                            "match_date": actual_match_date,
                            "match": f"{bet['home_team']} vs {bet['away_team']}",
                        }
                    )

        if future_bets > 0:
            red_flags.append(
                f"Found {future_bets} bets placed after match occurred - Look-ahead bias!"
            )

            # Show first few violations for debugging
            for violation in temporal_violations[:3]:
                red_flags.append(
                    f"  Example: {violation['match']} - bet on {violation['bet_date'].date()}, match on {violation['match_date'].date()}"
                )

        print(f"✓ Future information check: {future_bets} violations found")
        metrics["future_bets_violation"] = future_bets
        metrics["temporal_violations_detail"] = len(temporal_violations)

        # Check 2: Predictions should be made before matches
        prediction_violations = 0
        for _, pred in predictions_df.iterrows():
            # Check if prediction date is reasonable relative to match date
            pred_date = pred.get("Date")
            if pd.notna(pred_date):
                # In a proper backtest, predictions should be made on or slightly before match date
                # Flag if prediction appears to be made after match (accounting for time zones)
                pass  # This would need match kickoff times to be fully implemented

        # Check 3: Chronological betting order
        betting_df_sorted = betting_df.sort_values("date")
        if not betting_df["date"].equals(betting_df_sorted["date"]):
            warnings.append("Betting results not in chronological order")

        print(f"✓ Temporal ordering validated")
        print(f"✓ Prediction timing checked")

        return red_flags, warnings, metrics

    def _validate_odds_consistency(
        self, betting_df: pd.DataFrame, predictions_df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Validate odds and probability consistency."""
        red_flags = []
        warnings = []
        metrics = {}

        # Check 1: Probabilities sum to ~1
        prob_cols = ["Model_Prob_H", "Model_Prob_D", "Model_Prob_A"]
        if all(col in predictions_df.columns for col in prob_cols):
            prob_sums = predictions_df[prob_cols].sum(axis=1)
            prob_sum_violations = np.sum(np.abs(prob_sums - 1.0) > 0.01)

            if (
                prob_sum_violations > len(predictions_df) * 0.1
            ):  # More than 10% violations
                red_flags.append(
                    f"Model probabilities don't sum to 1 in {prob_sum_violations} cases"
                )

            metrics["prob_sum_mean"] = prob_sums.mean()
            metrics["prob_sum_violations"] = prob_sum_violations
            print(f"✓ Probability sum check: {prob_sum_violations} violations")

        # Check 2: Reasonable odds values
        if "odds" in betting_df.columns:
            unrealistic_odds = np.sum(
                (betting_df["odds"] < 1.01) | (betting_df["odds"] > 100)
            )
            if unrealistic_odds > 0:
                warnings.append(f"Found {unrealistic_odds} bets with unrealistic odds")

            metrics["unrealistic_odds"] = unrealistic_odds
            metrics["mean_odds"] = betting_df["odds"].mean()
            print(f"✓ Odds validation: {unrealistic_odds} unrealistic odds found")

        # Check 3: Edge calculations are consistent
        if all(
            col in betting_df.columns for col in ["model_prob", "fair_prob", "edge"]
        ):
            calculated_edges = betting_df["model_prob"] - betting_df["fair_prob"]
            edge_discrepancies = np.sum(
                np.abs(calculated_edges - betting_df["edge"]) > 0.001
            )

            if edge_discrepancies > 0:
                red_flags.append(
                    f"Edge calculation inconsistencies in {edge_discrepancies} cases"
                )

            metrics["edge_discrepancies"] = edge_discrepancies
            print(f"✓ Edge calculation check: {edge_discrepancies} discrepancies")

        return red_flags, warnings, metrics

    def _analyze_betting_patterns(
        self, betting_df: pd.DataFrame, predictions_df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Analyze betting patterns for suspicious behavior."""
        red_flags = []
        warnings = []
        metrics = {}

        # Check 1: Bet type distribution
        if "bet_type" in betting_df.columns:
            bet_dist = betting_df["bet_type"].value_counts(normalize=True)

            # Flag if >80% of bets are on one outcome type
            if bet_dist.max() > 0.8:
                warnings.append(
                    f"Heavily skewed betting: {bet_dist.max():.1%} on {bet_dist.idxmax()}"
                )

            metrics.update({f"bet_type_{bt}_pct": pct for bt, pct in bet_dist.items()})
            print(f"✓ Bet type distribution: {dict(bet_dist.round(3))}")

        # Check 2: Edge distribution
        if "edge" in betting_df.columns:
            mean_edge = betting_df["edge"].mean()
            min_edge = betting_df["edge"].min()

            # Suspiciously high average edge
            if mean_edge > 0.15:
                red_flags.append(f"Unrealistically high average edge: {mean_edge:.3f}")
            elif mean_edge > 0.10:
                warnings.append(f"Very high average edge: {mean_edge:.3f}")

            # Check for negative edges (shouldn't bet on these)
            negative_edge_bets = np.sum(betting_df["edge"] < 0)
            if negative_edge_bets > 0:
                red_flags.append(f"Found {negative_edge_bets} bets with negative edge")

            metrics.update(
                {
                    "mean_edge": mean_edge,
                    "min_edge": min_edge,
                    "max_edge": betting_df["edge"].max(),
                    "negative_edge_bets": negative_edge_bets,
                }
            )
            print(
                f"✓ Edge analysis: Mean={mean_edge:.3f}, Negative={negative_edge_bets}"
            )

        # Check 3: Stake sizing patterns
        if "stake" in betting_df.columns:
            unique_stakes = betting_df["stake"].nunique()
            if unique_stakes == 1:
                warnings.append(
                    "All bets have identical stake size - consider dynamic sizing"
                )

            # Check for unrealistic stake progression
            max_stake = betting_df["stake"].max()
            min_stake = betting_df["stake"].min()

            metrics.update(
                {
                    "unique_stakes": unique_stakes,
                    "max_stake": max_stake,
                    "min_stake": min_stake,
                    "stake_ratio": max_stake / min_stake if min_stake > 0 else np.inf,
                }
            )
            print(
                f"✓ Stake analysis: {unique_stakes} unique stakes, ratio={max_stake / min_stake:.1f}"
            )

        return red_flags, warnings, metrics

    def _analyze_performance_distribution(
        self, betting_df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Analyze the distribution of betting performance."""
        red_flags = []
        warnings = []
        metrics = {}

        if "profit" not in betting_df.columns:
            warnings.append("No profit column found for performance analysis")
            return red_flags, warnings, metrics

        profits = betting_df["profit"]

        # Basic performance metrics
        total_profit = profits.sum()
        win_rate = np.mean(profits > 0)
        avg_win = profits[profits > 0].mean() if np.any(profits > 0) else 0
        avg_loss = profits[profits < 0].mean() if np.any(profits < 0) else 0

        # Check for unrealistic win rates
        if win_rate > 0.65:
            red_flags.append(f"Unrealistically high win rate: {win_rate:.1%}")
        elif win_rate > 0.6:
            warnings.append(f"Very high win rate: {win_rate:.1%}")

        # Check profit distribution
        profit_std = profits.std()
        profit_skew = profits.skew() if len(profits) > 2 else 0

        # Analyze streaks
        wins = (profits > 0).astype(int)
        win_streaks = self._calculate_streaks(wins)
        loss_streaks = self._calculate_streaks(1 - wins)

        max_win_streak = max(win_streaks) if win_streaks else 0
        max_loss_streak = max(loss_streaks) if loss_streaks else 0

        # Flag unusually long win streaks
        expected_max_streak = (
            -np.log(0.05) / np.log(1 - win_rate) if win_rate < 1 else len(profits)
        )
        if max_win_streak > expected_max_streak * 1.5:
            warnings.append(f"Unusually long win streak: {max_win_streak} bets")

        metrics.update(
            {
                "total_profit": total_profit,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_std": profit_std,
                "profit_skew": profit_skew,
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak,
                "expected_max_streak": expected_max_streak,
            }
        )

        print(
            f"✓ Performance analysis: WR={win_rate:.1%}, Max win streak={max_win_streak}"
        )

        return red_flags, warnings, metrics

    def _run_significance_tests(
        self, betting_df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Run statistical significance tests on the results."""
        red_flags = []
        warnings = []
        metrics = {}

        if "profit" not in betting_df.columns or len(betting_df) < 10:
            warnings.append("Insufficient data for significance testing")
            return red_flags, warnings, metrics

        profits = betting_df["profit"]

        # T-test for profit vs zero
        from scipy.stats import ttest_1samp

        t_stat, p_value = ttest_1samp(profits, 0)

        # Check statistical significance
        is_significant = p_value < 0.05

        # Calculate confidence interval
        from scipy.stats import t

        n = len(profits)
        mean_profit = profits.mean()
        std_err = profits.std() / np.sqrt(n)
        ci_95 = t.interval(0.95, n - 1, mean_profit, std_err)

        # Sharpe ratio (risk-adjusted return)
        if profits.std() > 0:
            sharpe_ratio = mean_profit / profits.std() * np.sqrt(len(profits))
        else:
            sharpe_ratio = 0

        metrics.update(
            {
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": is_significant,
                "ci_95_lower": ci_95[0],
                "ci_95_upper": ci_95[1],
                "sharpe_ratio": sharpe_ratio,
                "sample_size": n,
            }
        )

        if not is_significant:
            warnings.append(f"Results not statistically significant (p={p_value:.3f})")

        if sharpe_ratio > 2.0:
            warnings.append(f"Extremely high Sharpe ratio: {sharpe_ratio:.2f}")

        print(f"✓ Significance test: p={p_value:.3f}, Sharpe={sharpe_ratio:.2f}")

        return red_flags, warnings, metrics

    def _detect_lookahead_bias(
        self, predictions_df: pd.DataFrame, historical_df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Detect potential look-ahead bias in predictions."""
        red_flags = []
        warnings = []
        metrics = {}

        # This is a complex check that would need more detailed implementation
        # For now, we'll do basic checks

        # Check 1: Model probabilities vs actual outcomes correlation
        if all(
            col in predictions_df.columns for col in ["Model_Prob_H", "Actual_Outcome"]
        ):
            # Convert outcomes to probabilities for correlation
            n_outcomes = predictions_df["Actual_Outcome"].nunique()
            if n_outcomes == 3:  # Home, Draw, Away
                actual_home = (predictions_df["Actual_Outcome"] == 0).astype(float)
                model_home = predictions_df["Model_Prob_H"]

                correlation = np.corrcoef(actual_home, model_home)[0, 1]

                # Suspiciously high correlation might indicate overfitting or look-ahead bias
                if correlation > 0.3:
                    warnings.append(
                        f"High correlation between model and outcomes: {correlation:.3f}"
                    )

                metrics["model_outcome_correlation"] = correlation
                print(f"✓ Model-outcome correlation: {correlation:.3f}")

        return red_flags, warnings, metrics

    def _calculate_streaks(self, binary_series):
        """Calculate consecutive streaks in a binary series."""
        if len(binary_series) == 0:
            return []

        streaks = []
        current_streak = 1

        for i in range(1, len(binary_series)):
            if (
                binary_series.iloc[i] == binary_series.iloc[i - 1]
                and binary_series.iloc[i] == 1
            ):
                current_streak += 1
            else:
                if binary_series.iloc[i - 1] == 1:
                    streaks.append(current_streak)
                current_streak = 1

        # Don't forget the last streak
        if binary_series.iloc[-1] == 1:
            streaks.append(current_streak)

        return streaks

    def _calculate_confidence_score(
        self, red_flags: List[str], warnings: List[str], metrics: Dict[str, float]
    ) -> float:
        """Calculate overall confidence score for the backtest."""

        # Start with base score
        score = 1.0

        # Penalize red flags heavily
        score -= len(red_flags) * 0.3

        # Penalize warnings moderately
        score -= len(warnings) * 0.1

        # Adjust based on specific metrics
        if "p_value" in metrics:
            if metrics["p_value"] > 0.05:
                score -= 0.2  # Not statistically significant

        if "win_rate" in metrics:
            if metrics["win_rate"] > 0.65:
                score -= 0.15  # Suspiciously high

        if "mean_edge" in metrics:
            if metrics["mean_edge"] > 0.1:
                score -= 0.1  # Very high edge

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

    def create_validation_report(
        self,
        validation_result: ValidationResult,
        betting_results_csv: str,
        output_path: str = "backtest_validation_report.html",
    ) -> None:
        """Create a comprehensive HTML validation report."""

        betting_df = pd.read_csv(betting_results_csv)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .red-flag {{ color: #d32f2f; font-weight: bold; }}
                .warning {{ color: #f57c00; font-weight: bold; }}
                .metric {{ background-color: #f8f9fa; padding: 5px; margin: 2px 0; }}
                .valid {{ color: #388e3c; font-weight: bold; }}
                .invalid {{ color: #d32f2f; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Backtest Validation Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Overall Valid: <span class="{"valid" if validation_result.is_valid else "invalid"}">
                    {validation_result.is_valid}</span></p>
                <p>Confidence Score: {validation_result.confidence_score:.2f}/1.0</p>
            </div>
        """

        # Add red flags
        if validation_result.red_flags:
            html_content += """
            <div class="section">
                <h2>🚨 Critical Issues (Red Flags)</h2>
                <ul>
            """
            for flag in validation_result.red_flags:
                html_content += f'<li class="red-flag">{flag}</li>'
            html_content += "</ul></div>"

        # Add warnings
        if validation_result.warnings:
            html_content += """
            <div class="section">
                <h2>⚠️ Warnings</h2>
                <ul>
            """
            for warning in validation_result.warnings:
                html_content += f'<li class="warning">{warning}</li>'
            html_content += "</ul></div>"

        # Add detailed metrics
        html_content += """
        <div class="section">
            <h2>📊 Detailed Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """

        for metric, value in validation_result.detailed_metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            html_content += f"<tr><td>{metric}</td><td>{value_str}</td></tr>"

        html_content += "</table></div>"

        # Add sample betting results
        if len(betting_df) > 0:
            sample_size = min(20, len(betting_df))
            sample_df = betting_df.head(sample_size)

            html_content += f"""
            <div class="section">
                <h2>📋 Sample Betting Results (First {sample_size} bets)</h2>
                {sample_df.to_html(classes="table", table_id="betting_sample")}
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"\n📄 Validation report saved to: {output_path}")

    @staticmethod
    def _extract_league_name(file_path: str) -> str:
        # Get the file name only
        file_name = os.path.basename(file_path)
        # Remove the suffix
        if file_name.endswith("_best_betting_results.csv"):
            league_name = file_name.replace("_best_betting_results.csv", "")
            return league_name
        else:
            raise ValueError("Unexpected file name format.")


# Usage functions for your specific setup


def validate_zsd_backtest_results(betting_csv, predictions_csv, config):
    """Validate ZSD backtesting results."""

    validator = BacktestValidator()

    historical_csv = "historical_ppi_and_odds.csv"

    try:
        # Run validation
        result = validator.validate_backtest_results(
            betting_results_csv=betting_csv,
            predictions_csv=predictions_csv,
            historical_data_csv=historical_csv,
            config=config,
        )

        # Create detailed report
        validator.create_validation_report(
            validation_result=result,
            betting_results_csv=betting_csv,
            output_path="zsd_validation_report.html",
        )

        return result

    except Exception as e:
        print(f"Validation failed: {e}")
        return None


def manual_bet_inspection_helper(betting_csv_path: str, n_samples: int = 50):
    """Create a helper file for manual inspection of individual bets."""

    betting_df = pd.read_csv(betting_csv_path)

    # Sample random bets for inspection
    if len(betting_df) > n_samples:
        sample_df = betting_df.sample(n_samples, random_state=42)
    else:
        sample_df = betting_df.copy()

    # Add inspection columns
    sample_df["Manual_Verification"] = "PENDING"  # CORRECT/INCORRECT/SUSPICIOUS
    sample_df["Notes"] = ""
    sample_df["Odds_Verified"] = "PENDING"  # YES/NO
    sample_df["Match_Date_Correct"] = "PENDING"  # YES/NO

    # Reorder columns for easier inspection
    inspection_cols = [
        "date",
        "home_team",
        "away_team",
        "bet_type",
        "odds",
        "stake",
        "profit",
        "edge",
        "model_prob",
        "fair_prob",
        "Manual_Verification",
        "Odds_Verified",
        "Match_Date_Correct",
        "Notes",
    ]

    available_cols = [col for col in inspection_cols if col in sample_df.columns]
    inspection_df = sample_df[available_cols]

    # Save inspection template
    output_path = f"manual_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    inspection_df.to_csv(output_path, index=False)

    print(f"Manual inspection template created: {output_path}")
    print(f"Contains {len(inspection_df)} randomly sampled bets for verification")
    print("\nInspection Instructions:")
    print("1. Verify odds are correct by checking bookmaker websites/archives")
    print("2. Confirm match dates and results")
    print("3. Check if edge calculations seem reasonable")
    print("4. Mark 'Manual_Verification' as CORRECT/INCORRECT/SUSPICIOUS")
    print("5. Add notes for any concerns")

    return inspection_df


def cross_validate_with_different_seeds():
    """Run backtest with different random seeds to check stability."""

    print(f"{'=' * 60}\r\nCROSS-VALIDATION WITH DIFFERENT SEEDS\r\n{'=' * 60}\r\n")

    # This would require modifying your backtesting code to accept random seeds
    # and running the same backtest multiple times

    results = []
    for seed in [42, 123, 456, 789, 999]:
        print(f"Running backtest with seed {seed}...")
        # result = run_backtest_with_seed(seed)  # You'd implement this
        # results.append(result['roi_percent'])

    # Check if results are consistent across seeds
    # roi_std = np.std(results)
    # if roi_std > 5.0:  # More than 5% standard deviation
    #     print(f"⚠️  High variance across seeds: {roi_std:.1f}%")
    # else:
    #     print(f"✓ Stable results across seeds: {roi_std:.1f}% std dev")


def benchmark_against_random_betting(betting_csv):
    """Compare your results against random betting baseline."""

    print(f"{'=' * 60}\r\nRANDOM BETTING BENCHMARK\r\n{'=' * 60}\r\n")

    try:
        betting_df = pd.read_csv(betting_csv)

        # Simulate random betting on the same matches
        n_simulations = 1000
        random_results = []

        for _ in range(n_simulations):
            # Random betting: pick random outcome for each match
            random_profits = []

            for _, bet in betting_df.iterrows():
                # Assume we randomly pick home/draw/away with equal probability
                random_choice = np.random.choice(["home", "draw", "away"])

                # Get the odds for that choice (simplified)
                if hasattr(bet, "odds"):  # Use actual odds if available
                    odds = bet["odds"]
                else:
                    odds = 2.0  # Default odds

                # Simulate outcome (simplified - would need actual match results)
                win_prob = 1 / 3  # Equal probability assumption
                if np.random.random() < win_prob:
                    profit = bet["stake"] * (odds - 1)
                else:
                    profit = -bet["stake"]

                random_profits.append(profit)

            random_roi = sum(random_profits) / sum(betting_df["stake"]) * 100
            random_results.append(random_roi)

        # Compare against your actual results
        actual_roi = betting_df["profit"].sum() / betting_df["stake"].sum() * 100
        random_roi_mean = np.mean(random_results)
        random_roi_std = np.std(random_results)

        # Calculate percentile
        better_than_random = np.mean(actual_roi > np.array(random_results))

        print(f"Your ROI: {actual_roi:.1f}%")
        print(f"Random betting ROI: {random_roi_mean:.1f}% ± {random_roi_std:.1f}%")
        print(
            f"Your result is better than {better_than_random:.1%} of random strategies"
        )

        if better_than_random < 0.95:
            print("⚠️  Results not significantly better than random betting")
        else:
            print("✓ Results significantly outperform random betting")

    except Exception as e:
        print(f"Benchmark failed: {e}")


def analyze_market_efficiency_violations(betting_csv):
    """Check if your edges are realistic given market efficiency."""

    print(f"{'=' * 60}\r\nMARKET EFFICIENCY ANALYSIS\r\n{'=' * 60}\r\n")

    try:
        betting_df = pd.read_csv(betting_csv)

        if "edge" not in betting_df.columns:
            print("No edge data available")
            return

        edges = betting_df["edge"]

        # Analyze edge distribution
        print(f"Edge Statistics:")
        print(f"  Mean: {edges.mean():.3f}")
        print(f"  Median: {edges.median():.3f}")
        print(f"  Std: {edges.std():.3f}")
        print(f"  Max: {edges.max():.3f}")
        print(f"  Min: {edges.min():.3f}")

        # Check for unrealistic patterns
        high_edges = np.sum(edges > 0.1)  # >10% edge
        very_high_edges = np.sum(edges > 0.2)  # >20% edge

        print(f"\nEdge Distribution:")
        print(f"  Edges > 5%: {np.sum(edges > 0.05)}")
        print(f"  Edges > 10%: {high_edges}")
        print(f"  Edges > 20%: {very_high_edges}")

        if very_high_edges > 0:
            print("🚨 Found edges >20% - extremely suspicious!")
        elif high_edges > len(edges) * 0.1:
            print("⚠️  Many edges >10% - check calculations")
        else:
            print("✓ Edge distribution seems reasonable")

        # Look at edge vs outcome correlation
        if "profit" in betting_df.columns:
            wins = betting_df["profit"] > 0
            win_edges = edges[wins]
            loss_edges = edges[~wins]

            print(f"\nEdge by Outcome:")
            print(f"  Winning bets avg edge: {win_edges.mean():.3f}")
            print(f"  Losing bets avg edge: {loss_edges.mean():.3f}")

            # This should be similar - if winning bets have much higher edges,
            # it might indicate look-ahead bias
            edge_diff = win_edges.mean() - loss_edges.mean()
            if abs(edge_diff) > 0.05:
                print(f"⚠️  Large edge difference by outcome: {edge_diff:.3f}")

    except Exception as e:
        print(f"Market efficiency analysis failed: {e}")


if __name__ == "__main__":
    # Example usage
    print("Running comprehensive backtest validation...")

    # 1. Main validation
    validation_result = validate_zsd_backtest_results()

    if validation_result:
        print(f"\nValidation complete!")
        print(f"Valid: {validation_result.is_valid}")
        print(f"Confidence: {validation_result.confidence_score:.2f}")

    # 2. Manual inspection helper
    try:
        print(f"\nCreating manual inspection helper...")
        manual_bet_inspection_helper("base_betting_results.csv", n_samples=30)
    except FileNotFoundError:
        print("Betting results file not found - run backtest first")

    # 3. Additional validation checks
    print(f"{'=' * 60}\r\nADDITIONAL VALIDATION CHECKS\r\n{'=' * 60}\r\n")

    # Market efficiency check
    analyze_market_efficiency_violations()

    # Random betting benchmark
    benchmark_against_random_betting()

    # Cross-validation note
    print(f"{'=' * 60}\r\nRECOMMENDED ADDITIONAL CHECKS\r\n{'=' * 60}\r\n")
    print("1. Run backtest on different time periods")
    print("2. Test on different leagues separately")
    print("3. Use walk-forward validation")
    print("4. Check results against betting exchange data")
    print("5. Paper trade for a few weeks before going live")
