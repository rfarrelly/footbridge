import asyncio
import time

import pandas as pd
from config import AppConfig, Leagues
from pydoll.browser import Chrome
from utils.data_corrections import fix_scunthorpe_wealdestone_2025_2026
from utils.datetime_helpers import format_date
from utils.url_helpers import (
    fbduk_main_url_builder,
    fbref_url_builder,
)


class DataIngestion:
    def __init__(self, config: AppConfig):
        self.config = config

    def write_files(
        self,
        df: pd.DataFrame,
        dir: str,
        league_name: str,
        season: str,
        prefix: str = None,
    ):
        filename = f"{league_name}_{season}.csv"
        if prefix:
            filename = prefix + filename
        df.to_csv(f"{dir}/{filename}", index=False)
        print(f"File '{filename}' downloaded and saved to '{dir}'\r\n")

    async def get_data_async(self, browser: Chrome, url: str):
        # Step 1: Open a new tab
        tab = await browser.start()
        try:
            # Step 2: Load the page (and possibly bypass CAPTCHA)
            async with tab.expect_and_bypass_cloudflare_captcha(time_before_click=5):
                # Step 3: Allow page scripts / Cloudflare completion
                await asyncio.sleep(2)
                await tab.go_to(url)
                print(f"✅ Loaded: {url}")
                # Step 4: Get page data (HTTP request or HTML)
            data = await tab.request.get(url)
            return data
        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")
            raise
        finally:
            await tab.close()

    async def get_fbref_data(self, league: Leagues, season: str, browser):
        league_name = league.fbref_name
        url = fbref_url_builder(self.config.fbref_base_url, league, season)
        if league.is_extra:
            dir_path = self.config.get_fbref_league_dir(
                league.fbduk_id + "_" + league_name
            )
        else:
            dir_path = self.config.get_fbref_league_dir(league_name)
        try:
            response = await self.get_data_async(browser, url)
        except Exception as e:
            print(e)

        columns = [
            "Wk",
            "League",
            "Season",
            "Day",
            "Date",
            "Home",
            "Score",
            "Away",
        ]

        data_df = pd.read_html(response.content)[0]

        data_df = data_df[
            ~data_df["Notes"].isin(["Match Suspended", "Match Cancelled"])
        ]

        if "Round" in data_df.columns:
            data_df = self._drop_non_regular_matches(data_df)

        data_df["League"] = league_name
        data_df["Season"] = season

        unplayed_fixtures_df = (
            data_df[data_df["Score"].isna()][columns]
            .drop("Score", axis="columns")
            .dropna(how="any", axis="index")
        )

        played_fixtures_df = data_df.dropna(
            how="any", subset=["Wk", "Score"], axis="index"
        )[columns]

        played_fixtures_df[["FTHG", "FTAG"]] = played_fixtures_df["Score"].apply(
            lambda x: pd.Series(self._parse_score(x))
        )
        played_fixtures_df = played_fixtures_df.drop("Score", axis="columns")

        played_fixtures_df = fix_scunthorpe_wealdestone_2025_2026(played_fixtures_df)

        if league.is_extra:
            self.write_files(
                played_fixtures_df,
                dir_path,
                league_name,
                season,
                prefix=league.fbduk_id + "_",
            )
            self.write_files(
                unplayed_fixtures_df,
                dir_path,
                league_name,
                season,
                prefix="unplayed_" + league.fbduk_id + "_",
            )
        else:
            self.write_files(played_fixtures_df, dir_path, league_name, season)
            self.write_files(
                unplayed_fixtures_df, dir_path, league_name, season, prefix="unplayed_"
            )
        time.sleep(3)

    def get_fbduk_data(self, league: Leagues, season: str):
        if league.is_extra:
            return None

        league_name = league.fbref_name
        odds_columns = [
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

        url = fbduk_main_url_builder(self.config.fbduk_base_url_main, league, season)
        columns = ["Date", "HomeTeam", "AwayTeam"] + odds_columns

        dir_path = self.config.get_fbduk_league_dir(league_name)

        data_df = pd.read_csv(url, encoding="latin-1")[columns].rename(
            columns={"HomeTeam": "Home", "AwayTeam": "Away"}
        )

        data_df = self._fill_empty_odds(data_df)

        data_df = format_date(data_df)

        self.write_files(data_df, dir_path, league_name, season)

    @staticmethod
    def _parse_score(score: str) -> tuple:
        if score:
            goals = score.split("–")
            return int(goals[0]), int(goals[1])
        return 0, 0

    @staticmethod
    def _drop_non_regular_matches(df: pd.DataFrame):
        df = df.copy()
        # Identify the first value we want to keep
        first_value = df["Round"].iloc[0]

        # Keep rows until the first non-matching one
        return df[df["Round"] == first_value]

    @staticmethod
    def _fill_empty_odds(df: pd.DataFrame):
        df = df.copy()

        pinny_pairs = [("PSH", "PSCH"), ("PSD", "PSCD"), ("PSA", "PSCA")]
        b365_pairs = [("B365H", "B365CH"), ("B365D", "B365CD"), ("B365A", "B365CA")]

        for col1, col2 in pinny_pairs:
            # Fill col1 from col2 where col1 is NaN
            df[col1] = df[col1].fillna(df[col2])
            # Fill col2 from col1 where col2 is NaN
            df[col2] = df[col2].fillna(df[col1])

        for col1, col2 in b365_pairs:
            # Fill col1 from col2 where col1 is NaN
            df[col1] = df[col1].fillna(df[col2])
            # Fill col2 from col1 where col2 is NaN
            df[col2] = df[col2].fillna(df[col1])

        return df
