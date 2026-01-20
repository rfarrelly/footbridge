import csv
import difflib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG


class TeamNameMapper:
    """
    A class to manage team name mappings across different data sources.
    """

    def __init__(self, csv_path: str, data_sources: List[str]):
        """
        Initialize the TeamNameMapper with a CSV file path and data sources.

        Args:
            csv_path: Path to the CSV file that stores team name mappings
            data_sources: List of data source names (will be used as column headers)
        """
        self.csv_path = csv_path
        self.data_sources = data_sources
        self.team_dict: Dict[str, Dict[str, str]] = {}

        # Load existing dictionary if file exists
        if os.path.exists(csv_path):
            self.load_dictionary()
        else:
            # Create a new file with headers
            self._create_new_dictionary()

    def _create_new_dictionary(self) -> None:
        """Create a new dictionary CSV file with the appropriate headers."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.data_sources)

    def load_dictionary(self) -> None:
        """Load the team name dictionary from the CSV file."""
        self.team_dict = {}

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Validate headers match our data sources
            headers = reader.fieldnames
            if headers and not set(headers).issubset(set(self.data_sources)):
                missing_sources = set(headers) - set(self.data_sources)
                print(
                    f"Warning: CSV contains sources not in current configuration: {missing_sources}"
                )

            for row in reader:
                # Create an entry for each team in each source
                for source, team_name in row.items():
                    if (
                        team_name and source in self.data_sources
                    ):  # Skip empty cells and unknown sources
                        if source not in self.team_dict:
                            self.team_dict[source] = {}
                        # Map team name to the full row
                        self.team_dict[source][team_name] = {
                            s: v for s, v in row.items() if v and s in self.data_sources
                        }

    def save_dictionary(self) -> None:
        """Save the current dictionary to the CSV file."""
        # First, collect all unique team entries
        all_entries = []
        processed_entries = set()

        for source in self.data_sources:
            if source not in self.team_dict:
                continue

            for team_name, mappings in self.team_dict[source].items():
                # Create a tuple of values to check if we've seen this entry
                entry_key = tuple(mappings.get(s, "") for s in self.data_sources)

                if entry_key not in processed_entries:
                    # Ensure all data sources are present (even if empty)
                    entry = {s: mappings.get(s, "") for s in self.data_sources}
                    all_entries.append(entry)
                    processed_entries.add(entry_key)

        # Write entries to CSV
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.data_sources)
            writer.writeheader()
            writer.writerows(all_entries)

    def find_team_mapping(
        self, team_name: str, source: str
    ) -> Optional[Dict[str, str]]:
        """
        Find a team mapping for a given team name from a specific source.

        Args:
            team_name: The team name to look up
            source: The data source this team name is from

        Returns:
            Dictionary mapping source names to team names, or None if not found
        """
        if source in self.team_dict and team_name in self.team_dict[source]:
            return self.team_dict[source][team_name]
        return None

    def find_partial_matches(
        self, team_name: str, source: str, min_similarity: float = 0.6
    ) -> List[Tuple[str, float, Dict[str, str]]]:
        """
        Find partial matches for a team name.

        Args:
            team_name: The team name to look up
            source: The data source this team name is from
            min_similarity: Minimum similarity ratio required (0-1)

        Returns:
            List of tuples (matching_name, similarity_score, mapping_dict)
        """
        matches = []
        team_name_lower = team_name.lower()

        # Try to find matches in any source
        for src in self.data_sources:
            if src not in self.team_dict:
                continue

            for existing_name, mapping in self.team_dict[src].items():
                existing_lower = existing_name.lower()

                # Check exact match on any word
                team_words = set(team_name_lower.split())
                existing_words = set(existing_lower.split())

                # For team names, check if any full word matches exactly
                word_match = False
                for team_word in team_words:
                    if (
                        len(team_word) > 2
                    ):  # Ignore short words like "FC", "United", etc.
                        for exist_word in existing_words:
                            if len(exist_word) > 2 and team_word == exist_word:
                                word_match = True
                                break

                # If we found an exact word match, give it a high similarity
                if word_match:
                    # Calculate real similarity for sorting
                    similarity = difflib.SequenceMatcher(
                        None, team_name_lower, existing_lower
                    ).ratio()
                    # Boost similarity for word matches
                    adjusted_similarity = max(similarity, 0.7)
                    matches.append((existing_name, adjusted_similarity, mapping))
                else:
                    # Fall back to sequence matching
                    similarity = difflib.SequenceMatcher(
                        None, team_name_lower, existing_lower
                    ).ratio()
                    if similarity >= min_similarity:
                        matches.append((existing_name, similarity, mapping))

        # Sort by similarity score (descending)
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def add_team_mapping(
        self, team_name: str, source: str, mappings: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Add a new team mapping.

        Args:
            team_name: The team name to add
            source: The data source for this team name
            mappings: Optional dictionary of mappings for other sources

        Returns:
            Complete mapping dictionary
        """
        if source not in self.data_sources:
            raise ValueError(f"Unknown data source: {source}")

        # Initialize source dictionary if needed
        if source not in self.team_dict:
            self.team_dict[source] = {}

        # Check if team already exists in this source
        if team_name in self.team_dict[source]:
            return self.team_dict[source][team_name]

        # Create new mapping
        new_mapping = {source: team_name}
        if mappings:
            # Add only mappings for valid sources
            for s, name in mappings.items():
                if s in self.data_sources:
                    new_mapping[s] = name

        # Add to dictionary
        self.team_dict[source][team_name] = new_mapping

        # Also add references from other sources
        for src, name in new_mapping.items():
            if src != source:
                if src not in self.team_dict:
                    self.team_dict[src] = {}
                self.team_dict[src][name] = new_mapping

        return new_mapping

    def suggest_mappings(self, team_name: str, source: str) -> List[Dict[str, str]]:
        """
        Suggest possible mappings for a team name based on partial matches.

        Args:
            team_name: The team name to find mappings for
            source: The data source for this team name

        Returns:
            List of possible mapping dictionaries
        """
        # First check for exact match
        exact_match = self.find_team_mapping(team_name, source)
        if exact_match:
            return [exact_match]

        # Then check for partial matches
        partial_matches = self.find_partial_matches(team_name, source)
        return [match[2] for match in partial_matches]


class TeamNameManagerCLI:
    """Command-line interface for the TeamNameMapper."""

    def __init__(self, csv_path: str, data_sources: List[str]):
        """
        Initialize the CLI interface.

        Args:
            csv_path: Path to the CSV file
            data_sources: List of data source names
        """
        self.mapper = TeamNameMapper(csv_path, data_sources)

    def process_team_entry(
        self,
        team_name: str,
        source: str,
        auto_match: bool = False,
        auto_threshold: float = 0.8,
        interactive: bool = True,
    ) -> None:
        """
        Process a team name entry, checking for existing mappings or suggesting new ones.

        Args:
            team_name: The team name to process
            source: The data source for this team name
            auto_match: Whether to automatically match teams above the threshold
            auto_threshold: Similarity threshold for automatic matching
            interactive: Whether to prompt for user input
        """
        # Skip empty team names
        if not team_name.strip():
            return

        # Check if team already exists
        existing = self.mapper.find_team_mapping(team_name, source)
        if existing:
            if interactive:
                print(f"Team '{team_name}' from {source} is already mapped:")
                for src, name in existing.items():
                    if name:  # Skip empty values
                        print(f"  - {src}: {name}")
            return

        # Check for partial matches
        partial_matches = self.mapper.find_partial_matches(team_name, source)

        if partial_matches:
            if auto_match and partial_matches[0][1] >= auto_threshold:
                # Auto-match with the highest similarity match
                best_match = partial_matches[0]
                mapping = best_match[2].copy()
                mapping[source] = team_name

                # Update all references
                for src, name in mapping.items():
                    if src not in self.mapper.team_dict:
                        self.mapper.team_dict[src] = {}
                    self.mapper.team_dict[src][name] = mapping

                if interactive:
                    print(
                        f"Auto-matched '{team_name}' from {source} with '{best_match[0]}' (similarity: {best_match[1]:.2f})"
                    )
            elif interactive:
                print(f"Found similar teams for '{team_name}' from {source}:")
                for i, (match_name, similarity, mapping) in enumerate(partial_matches):
                    print(f"  {i + 1}. '{match_name}' (similarity: {similarity:.2f})")
                    for src, name in mapping.items():
                        if name:  # Skip empty values
                            print(f"     - {src}: {name}")

                choice = input(
                    "\nSelect a match to use (number), 'n' to create new, or 'q' to quit: "
                )

                if choice.lower() == "q":
                    return
                elif choice.lower() == "n":
                    self._create_new_mapping(team_name, source, interactive)
                else:
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(partial_matches):
                            # Add this team name to the existing mapping
                            mapping = partial_matches[idx][2].copy()
                            mapping[source] = team_name

                            # Update all references
                            for src, name in mapping.items():
                                if src not in self.mapper.team_dict:
                                    self.mapper.team_dict[src] = {}
                                self.mapper.team_dict[src][name] = mapping

                            print("Updated mapping.")
                        else:
                            print("Invalid selection. Creating new mapping.")
                            self._create_new_mapping(team_name, source, interactive)
                    except ValueError:
                        print("Invalid input. Creating new mapping.")
                        self._create_new_mapping(team_name, source, interactive)
        else:
            # No matches found
            if interactive:
                self._create_new_mapping(team_name, source, interactive)
            else:
                # Just add the team name without mapping
                self.mapper.add_team_mapping(team_name, source)

        # Save after each operation
        self.mapper.save_dictionary()

    def _create_new_mapping(
        self, team_name: str, source: str, interactive: bool = True
    ) -> None:
        """Create a new mapping for a team name."""
        if interactive:
            print(f"Creating new mapping for '{team_name}' from {source}.")
            mappings = {source: team_name}

            # Ask for mappings for other sources
            for src in self.mapper.data_sources:
                if src != source:
                    name = input(f"Enter team name for {src} (leave blank to skip): ")
                    if name.strip():
                        mappings[src] = name

            self.mapper.add_team_mapping(team_name, source, mappings)
            print("New mapping created successfully.")
        else:
            self.mapper.add_team_mapping(team_name, source)

    def process_batch(
        self,
        team_names: List[str],
        source: str,
        auto_match: bool = False,
        auto_threshold: float = 0.8,
        interactive: bool = True,
    ) -> None:
        """
        Process a batch of team names from a single source.

        Args:
            team_names: List of team names to process
            source: The data source for these team names
            auto_match: Whether to automatically match teams above the threshold
            auto_threshold: Similarity threshold for automatic matching
            interactive: Whether to prompt for user input
        """
        for team_name in team_names:
            if interactive:
                print(f"\nProcessing '{team_name}' from {source}...")
            self.process_team_entry(
                team_name, source, auto_match, auto_threshold, interactive
            )

    def import_team_data(
        self,
        file_path: str,
        source: str,
        team_col: str,
        auto_match: bool = True,
        auto_threshold: float = 0.8,
        interactive: bool = True,
    ) -> None:
        """
        Import team names from a CSV file.

        Args:
            file_path: Path to the CSV file with team names
            source: The data source these names are from
            team_col: The column name containing team names
            auto_match: Whether to automatically match teams above the threshold
            auto_threshold: Similarity threshold for automatic matching
            interactive: Whether to prompt for user input
        """
        teams = set()

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if team_col in row and row[team_col].strip():
                    teams.add(row[team_col])

        if interactive:
            print(f"Found {len(teams)} teams in {file_path}")
        self.process_batch(list(teams), source, auto_match, auto_threshold, interactive)

    def import_team_list(
        self,
        team_list: List[str],
        source: str,
        auto_match: bool = True,
        auto_threshold: float = 0.7,
        interactive: bool = True,
    ) -> None:
        """
        Import team names from a list.

        Args:
            team_list: List of team names
            source: The data source these names are from
            auto_match: Whether to automatically match teams above the threshold
            auto_threshold: Similarity threshold for automatic matching
            interactive: Whether to prompt for user input
        """
        if interactive:
            print(f"Processing {len(team_list)} teams for source '{source}'")
        self.process_batch(team_list, source, auto_match, auto_threshold, interactive)


def build_team_name_dictionary():
    """
    Standalone function to build team name dictionary.
    Moved from main.py for better organization.
    """
    data_sources = ["fbref", "fbduk"]
    csv_path = "team_name_dictionary.csv"

    # Create manager
    manager = TeamNameManagerCLI(csv_path, data_sources)

    # Extract unique team names from fbduk files
    fbduk_files = list(DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv"))

    if fbduk_files:
        fbduk_teams = np.unique(
            pd.concat(
                [
                    pd.read_csv(str(file))[["Home", "Away"]]
                    for file in fbduk_files
                    if file.is_file()
                ]
            )
            .to_numpy()
            .flatten()
        )
    else:
        print("Warning: No fbduk files found")
        fbduk_teams = np.array([])

    # Extract unique team names from fbref files
    fbref_files = list(DEFAULT_CONFIG.fbref_data_dir.rglob("*.csv"))

    if fbref_files:
        fbref_teams = np.unique(
            pd.concat(
                [
                    pd.read_csv(str(file))[["Home", "Away"]]
                    for file in fbref_files
                    if file.is_file()
                ]
            )
            .to_numpy()
            .flatten()
        )
    else:
        print("Warning: No fbref files found")
        fbref_teams = np.array([])

    print(f"Found {len(fbref_teams)} unique fbref teams")
    print(f"Found {len(fbduk_teams)} unique fbduk teams")

    if len(fbref_teams) > 0:
        manager.import_team_list(
            fbref_teams, "fbref", auto_match=True, auto_threshold=0.7, interactive=True
        )

    if len(fbduk_teams) > 0:
        manager.import_team_list(
            fbduk_teams, "fbduk", auto_match=True, auto_threshold=0.7, interactive=True
        )

    print(f"Dictionary saved to {csv_path}")

    # Validate the dictionary
    try:
        dict_df = pd.read_csv(csv_path)
        print(f"Dictionary contains {len(dict_df)} team mappings")

        # Check for unmapped teams
        unmapped_fbref = dict_df[dict_df["fbref"].isna()]
        unmapped_fbduk = dict_df[dict_df["fbduk"].isna()]

        if len(unmapped_fbref) > 0:
            print(f"Warning: {len(unmapped_fbref)} fbref teams are unmapped")

        if len(unmapped_fbduk) > 0:
            print(f"Warning: {len(unmapped_fbduk)} fbduk teams are unmapped")

    except Exception as e:
        print(f"Error validating dictionary: {e}")


def validate_team_mappings():
    """
    Validate existing team name mappings.
    """
    try:
        dict_df = pd.read_csv("team_name_dictionary.csv")

        print("Team Name Dictionary Validation:")
        print(f"  Total mappings: {len(dict_df)}")

        # Check for duplicates
        fbref_duplicates = dict_df["fbref"].duplicated().sum()
        fbduk_duplicates = dict_df["fbduk"].duplicated().sum()

        if fbref_duplicates > 0:
            print(f"  WARNING: {fbref_duplicates} duplicate fbref entries")

        if fbduk_duplicates > 0:
            print(f"  WARNING: {fbduk_duplicates} duplicate fbduk entries")

        # Check for missing mappings
        missing_fbref = dict_df["fbref"].isna().sum()
        missing_fbduk = dict_df["fbduk"].isna().sum()

        if missing_fbref > 0:
            print(f"  WARNING: {missing_fbref} entries missing fbref mapping")

        if missing_fbduk > 0:
            print(f"  WARNING: {missing_fbduk} entries missing fbduk mapping")

        # Show sample mappings
        print("\nSample mappings:")
        sample_mappings = dict_df.dropna().head(5)
        for _, row in sample_mappings.iterrows():
            print(f"  {row['fbduk']} -> {row['fbref']}")

        return len(dict_df), fbref_duplicates == 0 and fbduk_duplicates == 0

    except FileNotFoundError:
        print("team_name_dictionary.csv not found. Run 'update_teams' mode first.")
        return 0, False
    except Exception as e:
        print(f"Error validating team mappings: {e}")
        return 0, False


def get_unmapped_teams():
    """
    Identify teams that exist in data files but are not in the mapping dictionary.
    """
    try:
        # Load existing dictionary
        try:
            dict_df = pd.read_csv("team_name_dictionary.csv")
            mapped_fbref = set(dict_df["fbref"].dropna())
            mapped_fbduk = set(dict_df["fbduk"].dropna())
        except FileNotFoundError:
            print("No existing dictionary found")
            mapped_fbref, mapped_fbduk = set(), set()

        # Get teams from current data files
        fbref_files = list(DEFAULT_CONFIG.fbref_data_dir.rglob("*.csv"))
        fbduk_files = list(DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv"))

        current_fbref = set()
        current_fbduk = set()

        if fbref_files:
            current_fbref = set(
                np.unique(
                    pd.concat(
                        [
                            pd.read_csv(str(file))[["Home", "Away"]]
                            for file in fbref_files
                            if file.is_file()
                        ]
                    )
                    .to_numpy()
                    .flatten()
                )
            )

        if fbduk_files:
            current_fbduk = set(
                np.unique(
                    pd.concat(
                        [
                            pd.read_csv(str(file))[["Home", "Away"]]
                            for file in fbduk_files
                            if file.is_file()
                        ]
                    )
                    .to_numpy()
                    .flatten()
                )
            )

        # Find unmapped teams
        unmapped_fbref = current_fbref - mapped_fbref
        unmapped_fbduk = current_fbduk - mapped_fbduk

        print("Unmapped Teams Analysis:")
        print(
            f"  Unmapped fbref teams ({len(unmapped_fbref)}): {sorted(list(unmapped_fbref))[:10]}"
        )
        print(
            f"  Unmapped fbduk teams ({len(unmapped_fbduk)}): {sorted(list(unmapped_fbduk))[:10]}"
        )

        return unmapped_fbref, unmapped_fbduk

    except Exception as e:
        print(f"Error identifying unmapped teams: {e}")
        return set(), set()


def main():
    pass


if __name__ == "__main__":
    main()
