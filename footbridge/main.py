import asyncio
import sys
from pathlib import Path

import pandas as pd
from backtesting_validator import (
    analyze_market_efficiency_violations,
    benchmark_against_random_betting,
    manual_bet_inspection_helper,
    validate_zsd_backtest_results,
)
from config import DEFAULT_CONFIG, AppConfig, Leagues
from processing import LeagueProcessor, get_historical_ppi
from pydoll.browser import Chrome
from pydoll.browser.options import ChromiumOptions
from utils.pipeline_config import (
    PipelineConfig,
    PipelineModeValidator,
    print_pipeline_help,
)
from utils.team_name_dict_builder import build_team_name_dictionary
from zsd_integration import (
    daily_model_fitting,
    generate_zsd_predictions,
    periodic_parameter_optimization,
    setup_zsd_integration,
)


class BettingPipeline:
    """Main betting pipeline orchestrator."""

    def __init__(self, config=None):
        self.pipeline_config = PipelineConfig(config or DEFAULT_CONFIG)
        self.validator = PipelineModeValidator(self.pipeline_config)
        self.zsd_processor = None

    def _initialize_zsd(self):
        """Initialize ZSD processor if not already done."""
        if self.zsd_processor is None:
            print("Initializing ZSD processor...")
            self.zsd_processor = setup_zsd_integration(self.pipeline_config.base_config)
        return self.zsd_processor

    def run_zsd_predictions(self) -> bool:
        """Generate ZSD predictions with enhanced features."""
        if not self._validate_mode("predict"):
            return False
        print(f"{'=' * 60}\r\nRUNNING ENHANCED ZSD PREDICTIONS\r\n{'=' * 60}\r\n")

        try:
            # Load fixtures data
            fixtures_file = "fixtures_ppi_and_odds.csv"
            if not Path(fixtures_file).exists():
                print(
                    f"Error: {fixtures_file} not found. Please run 'latest_ppi' first."
                )
                return False

            fixtures_with_odds = pd.read_csv(fixtures_file)
            print(f"Loaded {len(fixtures_with_odds)} fixtures")

            # Initialize ZSD processor and fit models
            zsd_processor = self._initialize_zsd()

            print("Fitting ZSD models with latest data...")
            daily_model_fitting(zsd_processor)

            print("Generating enhanced ZSD predictions...")
            zsd_predictions = generate_zsd_predictions(
                zsd_processor, fixtures_with_odds
            )

            if not zsd_predictions:
                print("No ZSD predictions generated")
                return False

            # Process and save predictions
            return self._process_zsd_results(zsd_predictions)

        except FileNotFoundError as e:
            print(f"Required file not found: {e}")
            return False
        except Exception as e:
            print(f"Error in ZSD predictions: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _process_zsd_results(self, zsd_predictions) -> bool:
        """Process and save ZSD prediction results."""
        zsd_df = pd.DataFrame(zsd_predictions)

        # Save all predictions
        output_file = "latest_zsd_enhanced.csv"
        zsd_df.to_csv(output_file, index=False)
        print(f"Saved {len(zsd_df)} predictions to {output_file}")

        # Extract betting candidates
        betting_candidates = zsd_df[zsd_df["Is_Betting_Candidate"] == True]

        # Always display and save candidates (even if empty)
        self._display_betting_candidates(betting_candidates)

        candidates_file = "zsd_betting_candidates.csv"
        betting_candidates.to_csv(candidates_file, index=False)
        print(
            f"Saved {len(betting_candidates)} betting candidates to {candidates_file}"
        )

        # Display summary
        self._display_prediction_summary(zsd_df, betting_candidates)
        return True

    def _validate_mode(self, mode: str) -> bool:
        """Validate mode and show any issues."""
        is_valid, errors, warnings = self.validator.validate_mode(mode)

        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            print()

        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
            print("Cannot proceed with current mode. Please fix the issues above.")
            return False

        return True

    def run_parameter_optimization(self) -> bool:
        """Run ZSD parameter optimization for all leagues."""
        if not self._validate_mode("optimize"):
            return False

        print(f"{'=' * 60}\r\nRUNNING ZSD PARAMETER OPTIMIZATION\r\n{'=' * 60}\r\n")

        try:
            # Check if historical data exists
            historical_file = "historical_ppi_and_odds.csv"
            if not Path(historical_file).exists():
                print(
                    f"Error: {historical_file} not found. Please run 'historical_ppi' first."
                )
                return False

            zsd_processor = self._initialize_zsd()
            periodic_parameter_optimization(zsd_processor)
            print("Parameter optimization completed successfully")
            return True

        except FileNotFoundError as e:
            print(f"Required file not found: {e}")
            print("Please run 'historical_ppi' mode first to generate historical data.")
            return False
        except Exception as e:
            print(f"Error in parameter optimization: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _get_chrome_options(self) -> ChromiumOptions:
        options = ChromiumOptions()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )
        return options

    async def run_get_data(self, season: str, league: str | None = None) -> bool:
        """Download data for all leagues for a specific season."""
        print(f"Downloading data for season: {season}")

        if not season or len(season.split("-")) != 2:
            print("Error: Please provide a valid season format (e.g., '2023-24')")
            return False

        config = AppConfig(season)
        failed_leagues = []
        successful_leagues = []

        if league:
            league = Leagues[league]
            processor = LeagueProcessor(league, config)
            try:
                print(f"Processing {league.name}...")
                async with Chrome(options=self._get_chrome_options()) as browser:
                    await processor.get_fbref_data(browser)
                processor.get_fbduk_data()
                successful_leagues.append(league.name)
            except Exception as e:
                print(f"Error getting data for {league.name} {season}: {e}\r\n")
                failed_leagues.append(league.name)
        else:
            for league in Leagues:
                processor = LeagueProcessor(league, config)
                try:
                    print(f"Processing {league.name}...")
                    async with Chrome(options=self._get_chrome_options()) as browser:
                        await processor.get_fbref_data(browser)
                    processor.get_fbduk_data()
                    successful_leagues.append(league.name)
                except Exception as e:
                    print(f"Error getting data for {league.name} {season}: {e}\r\n")
                    failed_leagues.append(league.name)
                    continue

        # Summary
        print(f"\nData download summary:")
        print(f"  Successful: {len(successful_leagues)} leagues")
        print(f"  Failed: {len(failed_leagues)} leagues")

        if failed_leagues:
            print(f"  Failed leagues: {', '.join(failed_leagues)}")

        return len(failed_leagues) == 0

    def run_latest_ppi(self) -> bool:
        print(f"{'=' * 60}\r\nGENERATING LATEST PPI PREDICTIONS\r\n{'=' * 60}\r\n")

        ppi_main_leagues = []
        ppi_extra_leagues = []
        failed_leagues = []

        for league in Leagues:
            print(f"Processing {league.name} ({league.value['fbref_name']})")

            processor = LeagueProcessor(league, self.pipeline_config.base_config)

            try:
                ppi = processor.get_points_performance_index()

                if not isinstance(ppi, pd.DataFrame):
                    print(f"  No PPI data for {league.name}")
                    continue

                if not ppi.empty:
                    if league.is_extra:
                        ppi_extra_leagues.append(ppi)
                    else:
                        ppi_main_leagues.append(ppi)
                    print(f"  Generated {len(ppi)} PPI records for {league.name}")
                else:
                    print(f"  No PPI data for {league.name}")
            except Exception as e:
                print(f"Error getting latest PPI for {processor.league_name}: {e}")
                failed_leagues.append(league.name)
                continue

        from config import END_DATE, TODAY

        if ppi_extra_leagues:
            print(
                f"Getting EXTRA PPI betting candidates for the period {TODAY} to {END_DATE}"
            )
            ppi_latest_extra = pd.concat(ppi_extra_leagues).sort_values(by="PPIDiff")
            ppi_latest_extra.to_csv(f"{TODAY}_latest_ppi_extra.csv", index=False)
            print(f"Saved {len(ppi_latest_extra)} PPI records to latest_ppi_extra.csv")
        else:
            print("No PPI data generated for extra leagues")

        if ppi_main_leagues:
            print(
                f"Getting MAIN PPI betting candidates for the period {TODAY} to {END_DATE}"
            )
            ppi_latest_main = pd.concat(ppi_main_leagues).sort_values(by="PPIDiff")
            ppi_latest_main.to_csv(f"{TODAY}_latest_ppi_main.csv", index=False)
            print(f"Saved {len(ppi_latest_main)} PPI records to latest_ppi_main.csv")
        else:
            print("No PPI data generated for extra leagues")

        if failed_leagues:
            print(f"Note: Failed leagues: {', '.join(failed_leagues)}")
        return True

    def run_historical_ppi(self) -> bool:
        print(f"{'=' * 60}\r\nGENERATING HISTORICAL PPI DATA\r\n{'=' * 60}\r\n")
        historical_ppi = get_historical_ppi(self.pipeline_config.base_config)
        historical_ppi.to_csv("historical_metrics.csv", index=False)
        print(f"Saved {len(historical_ppi)} historical PPI records")

    def _display_betting_candidates(self, betting_candidates):
        """Display betting candidates in a formatted way with enhanced information."""
        if len(betting_candidates) == 0:
            print("No ZSD betting candidates found with current thresholds")
            return

        print(f"\nFound {len(betting_candidates)} enhanced ZSD betting candidates:")
        print("-" * 100)

        # Sort by edge (highest first)
        betting_candidates = betting_candidates.sort_values("Edge", ascending=False)

        max_display = min(
            len(betting_candidates), self.pipeline_config.max_recommendations
        )

        for idx, candidate in betting_candidates.head(max_display).iterrows():
            print(f"{candidate['Home']} vs {candidate['Away']} ({candidate['League']})")
            print(f"  Date: {candidate['Date']}")

            bet_type = candidate.get("Bet_Type", "N/A")
            edge = candidate.get("Edge", 0)
            soft_odds = candidate.get("Soft_Odds", 0)

            print(
                f"  Recommended Bet: {bet_type} at {soft_odds:.2f} (Edge: {edge:.3f})"
            )

            # Display all prediction methods
            print(f"  Model Predictions:")
            print(
                f"    Poisson:  H={candidate.get('Poisson_Prob_H', 0):.3f}, "
                f"D={candidate.get('Poisson_Prob_D', 0):.3f}, "
                f"A={candidate.get('Poisson_Prob_A', 0):.3f}"
            )
            print(
                f"    ZIP:      H={candidate.get('ZIP_Prob_H', 0):.3f}, "
                f"D={candidate.get('ZIP_Prob_D', 0):.3f}, "
                f"A={candidate.get('ZIP_Prob_A', 0):.3f}"
            )
            print(
                f"    MOV:      H={candidate.get('MOV_Prob_H', 0):.3f}, "
                f"D={candidate.get('MOV_Prob_D', 0):.3f}, "
                f"A={candidate.get('MOV_Prob_A', 0):.3f}"
            )

            # Display market analysis
            print(f"  Market Analysis:")
            print(
                f"    Model Prob: {candidate.get('Model_Prob', 0):.3f}, "
                f"Market Prob: {candidate.get('Market_Prob', 0):.3f}"
            )
            print(
                f"    Fair Odds: {candidate.get('Fair_Odds_Selected', 0):.2f}, "
                f"Sharp Odds: {candidate.get('Sharp_Odds', 0):.2f}"
            )

            if "PPI_Diff" in candidate:
                print(f"  PPI_Diff: {candidate['PPI_Diff']:.3f}")
            print()

        if len(betting_candidates) > max_display:
            print(f"... and {len(betting_candidates) - max_display} more candidates")

    def _display_prediction_summary(self, zsd_df, betting_candidates):
        """Display prediction summary statistics."""
        print(f"\nZSD PREDICTION SUMMARY:")
        print(f"Total predictions: {len(zsd_df)}")
        print(f"Betting candidates: {len(betting_candidates)}")

        if len(betting_candidates) > 0:
            avg_edge = betting_candidates["Edge"].mean()
            max_edge = betting_candidates["Edge"].max()
            print(f"Average edge: {avg_edge:.3f}")
            print(f"Maximum edge: {max_edge:.3f}")

            # Show distribution of bet types
            bet_type_dist = betting_candidates["Bet_Type"].value_counts()
            print(f"Bet type distribution: {bet_type_dist.to_dict()}")

        # Show distribution by league
        if len(zsd_df) > 0:
            league_summary = (
                zsd_df.groupby("League")
                .agg({"Is_Betting_Candidate": "sum", "ZIP_Prob_H": "count"})
                .rename(
                    columns={
                        "ZIP_Prob_H": "Total_Matches",
                        "Is_Betting_Candidate": "Betting_Candidates",
                    }
                )
            )

            print(f"\nBy League:")
            print(league_summary)

    def run_backtest_validation(self, betting_csv, predictions_csv):
        """Run comprehensive backtest validation."""
        print("Running comprehensive backtest validation...")

        # Check if files exist
        if not Path(betting_csv).exists():
            print(f"Error: Betting results file not found: {betting_csv}")
            return False

        if not Path(predictions_csv).exists():
            print(f"Error: Predictions file not found: {predictions_csv}")
            return False

        try:
            # 1. Main validation
            validation_result = validate_zsd_backtest_results(
                betting_csv, predictions_csv, self.pipeline_config
            )

            if validation_result:
                print(f"\nValidation complete!")
                print(f"Valid: {validation_result.is_valid}")
                print(f"Confidence: {validation_result.confidence_score:.2f}")

            # 2. Manual inspection helper
            print(f"\nCreating manual inspection helper...")
            manual_bet_inspection_helper(betting_csv, n_samples=50)

            # 3. Additional validation checks
            print(f"{'=' * 60}\r\nADDITIONAL VALIDATION CHECKS\r\n{'=' * 60}\r\n")

            # Market efficiency check
            analyze_market_efficiency_violations(betting_csv)

            # Random betting benchmark
            benchmark_against_random_betting(betting_csv)

            # Cross-validation note
            print(f"{'=' * 60}\r\nRECOMMENDED ADDITIONAL CHECKS\r\n{'=' * 60}\r\n")
            print("1. Run backtest on different time periods")
            print("2. Test on different leagues separately")
            print("3. Use walk-forward validation")
            print("4. Check results against betting exchange data")
            print("5. Paper trade for a few weeks before going live")

            return True

        except Exception as e:
            print(f"Error in backtest validation: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main entry point for the betting pipeline."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_pipeline_help()
        return

    pipeline = BettingPipeline()

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "status":
            pipeline.pipeline_config.print_status_report()

        # Mostly for testing
        elif mode == "get_data":
            if len(sys.argv) > 2:
                season = sys.argv[2]
                league = sys.argv[3]
                asyncio.run(pipeline.run_get_data(season, league))
            else:
                print("Usage: uv run app/main.py get_data <season> <league>")

        elif mode == "get_all_data":
            season = sys.argv[2]
            asyncio.run(pipeline.run_get_data(season))

        elif mode == "latest_ppi":
            pipeline.run_latest_ppi()

        elif mode == "historical_ppi":
            pipeline.run_historical_ppi()

        elif mode == "update_teams":
            build_team_name_dictionary()

        elif mode == "optimize":
            pipeline.run_parameter_optimization()

        elif mode == "validate":
            betting_results_directory = "optimisation_validation/betting_results"
            prediction_results_directory = "optimisation_validation/prediction_results"
            if len(sys.argv) > 3:
                betting_filename = f"{betting_results_directory}/{sys.argv[2]}"
                prediction_filename = f"{prediction_results_directory}/{sys.argv[3]}"
                pipeline.run_backtest_validation(betting_filename, prediction_filename)
            else:
                print(
                    "Usage: uv run app/main.py validate <betting_file> <predictions_file>"
                )

        elif mode == "predict":
            pipeline.run_zsd_predictions()

        elif mode == "train_ml":
            # Train ML predictor
            from train_ml_predictor import main as train_ml

            train_ml()

        else:
            print(f"Unknown mode: {mode}")
            print_pipeline_help()


if __name__ == "__main__":
    main()
