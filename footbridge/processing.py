import pandas as pd
import rs
from config import AppConfig, Leagues
from ingestion import DataIngestion


class LeagueProcessor:
    def __init__(self, league: Leagues, config: AppConfig):
        self.league = league
        self.config = config
        self.league_name = league.fbref_name
        self.fbref_dir = config.get_fbref_league_dir(
            self.league.fbduk_id + "_" + self.league_name
            if self.league.is_extra
            else self.league_name
        )
        self.ingestion = DataIngestion(config)

    @property
    def played_matches(self):
        file_path = (
            f"{self.fbref_dir}/{self.league_name}_{self.config.current_season}.csv"
        )
        if self.league.is_extra:
            file_path = f"{self.fbref_dir}/{self.league.fbduk_id}_{self.league_name}_{self.config.current_season}.csv"
        return file_path

    @property
    def unplayed_matches(self):
        filen_path = f"{self.fbref_dir}/unplayed_{self.league_name}_{self.config.current_season}.csv"
        if self.league.is_extra:
            filen_path = f"{self.fbref_dir}/unplayed_{self.league.fbduk_id}_{self.league_name}_{self.config.current_season}.csv"
        return filen_path

    async def get_fbref_data(self, browser):
        await self.ingestion.get_fbref_data(
            league=self.league, season=self.config.current_season, browser=browser
        )

    def get_fbduk_data(self):
        self.ingestion.get_fbduk_data(
            league=self.league, season=self.config.current_season
        )

    def get_points_performance_index(self) -> pd.DataFrame:
        return rs.compute_ppi_for_fixtures(self.unplayed_matches, self.played_matches)


def get_historical_ppi(config: AppConfig) -> pd.DataFrame:
    print("Processing historical PPI")

    exclude_leagues = [league.fbref_name for league in Leagues if league.is_extra]

    files = [
        str(file)
        for file in config.fbref_data_dir.rglob("*.csv")
        if file.is_file()
        if "unplayed" not in str(file)
        if not any(exclude in str(file) for exclude in exclude_leagues)
    ]

    historical_metrics = []

    for file_path in files:
        print(f"Processing {file_path}")
        try:
            historical_metrics.append(rs.compute_historical_ppi(file_path))
        except Exception as e:
            print(f"Failed to process {file_path}, continuing ... {e}")
            continue

    historical_metrics = pd.concat(historical_metrics)

    print(f"Historical processor processed: {historical_metrics.shape[0]} records")
    return historical_metrics.sort_values("Date").reset_index(drop=True)
