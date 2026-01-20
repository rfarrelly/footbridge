# utils/data_merging.py
"""
Data merging utilities for combining different data sources.
Moved from main.py for better organization.
"""

import pandas as pd
from config import DEFAULT_CONFIG, Leagues
from utils.datetime_helpers import format_date


def load_team_name_mapping():
    """Load team name mapping dictionary."""
    try:
        team_name_dict = pd.read_csv("team_name_dictionary.csv")
        return dict(zip(team_name_dict["fbduk"], team_name_dict["fbref"]))
    except FileNotFoundError:
        print(
            "Warning: team_name_dictionary.csv not found. Run 'update_teams' mode first."
        )
        return {}


def map_team_names(df, mapping_dict, home_col="Home", away_col="Away"):
    """Apply team name mapping to a dataframe."""

    def map_team_name(team_name):
        return mapping_dict.get(team_name, team_name)

    df = df.copy()
    df[home_col] = df[home_col].apply(map_team_name)
    df[away_col] = df[away_col].apply(map_team_name)
    return df


def load_fbduk_odds_data():
    """Load and combine all fbduk odds data files."""

    exclude_leagues = [league.fbref_name for league in Leagues if league.is_extra]

    # odds_files = list(DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv"))
    odds_files = [
        str(file)
        for file in DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv")
        if file.is_file()
        if not any(exclude in str(file) for exclude in exclude_leagues)
    ]
    if not odds_files:
        raise FileNotFoundError(
            f"No CSV files found in {DEFAULT_CONFIG.fbduk_data_dir}"
        )

    fbduk_odds_data = pd.concat([pd.read_csv(file) for file in odds_files])

    return fbduk_odds_data


def merge_historical_odds_data():
    """Merge historical fbduk odds data with fbref PPI data."""
    print("Merging historical odds data...")

    try:
        # Load data
        fbduk_odds_data = load_fbduk_odds_data()

        try:
            fbref_historical_rpi_data = pd.read_csv("historical_ppi.csv")
        except FileNotFoundError:
            print("historical_ppi.csv not found. Run historical_ppi mode first.")
            return

        print(f"fbduk input matches (odds): {fbduk_odds_data.shape[0]}")
        print(f"fbref input matches: {fbref_historical_rpi_data.shape[0]}")

        # Load team name mapping
        fbduk_to_fbref = load_team_name_mapping()

        if not fbduk_to_fbref:
            print("Warning: No team name mapping found. Proceeding without mapping.")
        else:
            # Apply team name mapping
            fbduk_odds_data = map_team_names(fbduk_odds_data, fbduk_to_fbref)
            print(f"fbduk after mapping: {fbduk_odds_data.shape[0]}")

        # Identify non-matching rows
        extra_fbduk_rows = pd.merge(
            fbduk_odds_data,
            fbref_historical_rpi_data,
            on=["Date", "Home", "Away"],
            how="left",
            indicator=True,
        )

        fbduk_not_in_fbref = extra_fbduk_rows[extra_fbduk_rows["_merge"] == "left_only"]

        if len(fbduk_not_in_fbref) > 0:
            print(
                f"{fbduk_not_in_fbref.shape[0]} fbduk rows not found in fbref (potentially non-reg season):"
            )
            print(
                fbduk_not_in_fbref[["Date", "Home", "Away"]].drop_duplicates().head(10)
            )

        # Perform the merge
        merged_df = (
            pd.merge(
                fbref_historical_rpi_data,
                fbduk_odds_data,
                on=["Date", "Home", "Away"],
                how="left",
            )
            .sort_values("Date")
            .rename({"Season_x": "Season", "Wk_x": "Wk"}, axis="columns")
            .drop_duplicates(subset=["Season", "Home", "Away"], keep="first")
        )

        # Identify unmerged rows
        unmerged_rows = merged_df[merged_df["PSH"].isna()]

        if len(unmerged_rows) > 0:
            print(f"{unmerged_rows.shape[0]} unmerged historical odds rows:")
            print(unmerged_rows[["Date", "Home", "Away"]].head(10))

        print(f"Merged historical odds size: {merged_df.shape[0]}")
        print(f"{merged_df[merged_df['PSH'].isna()].shape[0]} unmerged historical odds")

        # Save merged data
        merged_df.to_csv("historical_ppi_and_odds.csv", index=False)
        print("Saved historical_ppi_and_odds.csv")

    except Exception as e:
        print(f"Error merging historical odds data: {e}")
        raise


def load_future_odds_data():
    """Load future odds data from fixtures files."""
    try:
        # Main fixtures
        fbduk_main_odds = pd.read_csv("fixtures.csv").rename(
            {"HomeTeam": "Home", "AwayTeam": "Away"}, axis="columns"
        )[
            [
                "Date",
                "Home",
                "Away",
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
            ]
        ]

        # Optionally load additional fixtures
        additional_fixtures = []
        try:
            fbduk_extra_odds = pd.read_csv("new_league_fixtures.csv").rename(
                {"HomeTeam": "Home", "AwayTeam": "Away"}, axis="columns"
            )[["Date", "Home", "Away", "PSH", "PSD", "PSA"]]
            additional_fixtures.append(fbduk_extra_odds)
        except FileNotFoundError:
            pass  # Additional fixtures are optional

        # Combine all fixtures
        if additional_fixtures:
            fbduk_odds_data = pd.concat([fbduk_main_odds] + additional_fixtures)
        else:
            fbduk_odds_data = fbduk_main_odds

        return format_date(fbduk_odds_data)

    except FileNotFoundError as e:
        print(f"Error loading future odds data: {e}")
        raise


def merge_future_odds_data():
    """Merge future odds data with latest PPI data."""
    print("Merging future odds data...")

    try:
        # Load data
        try:
            latest_ppi = pd.read_csv("latest_ppi.csv")
        except FileNotFoundError:
            print("latest_ppi.csv not found. Run latest_ppi mode first.")
            return

        fbduk_odds_data = load_future_odds_data()

        print(f"Latest PPI records: {len(latest_ppi)}")
        print(f"Future odds records: {len(fbduk_odds_data)}")

        # Load team name mapping
        fbduk_to_fbref = load_team_name_mapping()

        if fbduk_to_fbref:
            # Apply team name mapping
            fbduk_odds_data = map_team_names(fbduk_odds_data, fbduk_to_fbref)

        # Merge data
        merged_df = pd.merge(
            latest_ppi,
            fbduk_odds_data,
            on=["Date", "Home", "Away"],
            how="left",
        ).sort_values("Date")

        # Select and order columns
        columns = [
            "Wk",
            "Date",
            "League",
            "Home",
            "Away",
            "aOppPPG",
            "hOppPPG",
            "aPPG",
            "hPPG",
            "hPPI",
            "aPPI",
            "PPI_Diff",
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
        ]

        # Only keep columns that exist
        available_columns = [col for col in columns if col in merged_df.columns]
        merged_df = merged_df[available_columns].sort_values(by="PPI_Diff")

        print(f"Merged future odds size: {merged_df.shape[0]}")

        # Check for unmerged records
        unmerged = merged_df[merged_df["PSH"].isna()]
        if len(unmerged) > 0:
            print(f"Warning: {len(unmerged)} records without odds data")

        # Save merged data
        merged_df.to_csv("fixtures_ppi_and_odds.csv", index=False)
        print("Saved fixtures_ppi_and_odds.csv")

    except Exception as e:
        print(f"Error merging future odds data: {e}")
        raise


def validate_merged_data(filename):
    """Validate merged data file for common issues."""
    try:
        df = pd.read_csv(filename)

        print(f"\nValidating {filename}:")
        print(f"  Total records: {len(df)}")

        # Check for missing critical columns
        required_cols = ["Date", "Home", "Away"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  WARNING: Missing required columns: {missing_cols}")

        # Check for duplicates
        duplicates = df.duplicated(subset=["Date", "Home", "Away"]).sum()
        if duplicates > 0:
            print(f"  WARNING: {duplicates} duplicate matches found")

        # Check data quality
        if "PSH" in df.columns:
            missing_odds = df["PSH"].isna().sum()
            print(f"  Records without odds: {missing_odds}")

        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

        return True

    except Exception as e:
        print(f"Error validating {filename}: {e}")
        return False


# Convenience function to run all merging operations
def merge_all_data():
    """Run all data merging operations in the correct order."""
    print("Running all data merging operations...")

    try:
        merge_historical_odds_data()
        validate_merged_data("historical_ppi_and_odds.csv")

        merge_future_odds_data()
        validate_merged_data("fixtures_ppi_and_odds.csv")

        print("All data merging completed successfully!")

    except Exception as e:
        print(f"Error in data merging: {e}")
        raise
