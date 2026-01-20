import random
import time
from datetime import datetime, timedelta

import pandas as pd
from curl_cffi import requests
from utils.datetime_helpers import filter_date_range

main_leagues = [
    "england",
    "england2",
    "england3",
    "england4",
    "england5",
    "belgium",
    "germany",
    "germany2",
    "spain",
    "spain2",
    "france",
    "france2",
    "greece",
    "netherlands",
    "netherlands2",
    "italy",
    "italy2",
    "portugal",
    "portugal2",
    "scotland",
    "scotland2",
    "scotland3",
    "scotland4",
    "turkey",
]

extra_leagues = [
    "argentina",
    "austria",
    "australia",
    "brazil",
    "switzerland",
    "czechrepublic",
    "denmark",
    "finland",
    "southkorea",
    "japan",
    "norway",
    "poland",
    "russia",
    "sweden",
    "ukraine",
]

ALT_INDEX_LEAGUES = [
    "greece",
    "scotland",
    "scotland3",
    "scotland4",
    "switzerland",
    "czechrepublic",
    "denmark",
    "finland",
    "poland",
    "ukraine",
]

TODAY = datetime.now().date()
TIME_DELTA = 3
END_DATE = TODAY + timedelta(days=TIME_DELTA)


def get_ppi_tables(league):
    print(f"Getting PPI for {league}")
    url = f"https://www.soccerstats.com/table.asp?league={league}&tid=rp"

    res = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        impersonate="safari_ios",
    )

    ppi_table = (
        pd.read_html(res.content)[11]
        .drop(
            ["Unnamed: 0", "Points Performance Index (Team PPG x Opponents PPG)"],
            axis=1,
        )
        .rename(
            columns={
                "Unnamed: 1": "Team",
                "Team PPG": "TeamPPG",
                "Opponents PPG": "OppsPPG",
                "Points Performance Index (Team PPG x Opponents PPG).1": "PPI",
            }
        )
    )

    ppi_table = ppi_table.head(len(ppi_table) - 1)

    ppi_table["OppsPPG"] = (
        ppi_table["OppsPPG"].str.extract(r"^\s*([0-9]+(?:\.[0-9]+)?)").astype(float)
    )

    ppi_table["TeamPPG"] = ppi_table["TeamPPG"].astype(float)
    ppi_table["PPI"] = ppi_table["PPI"].astype(float)

    if league == "england5":
        ppi_table.to_csv(f"SOCCERSTATS/{TODAY}_{league}.csv", index=False)

    league_average_ppg = ppi_table["TeamPPG"].mean()

    ppi_table["PPINorm"] = round(ppi_table["PPI"] / league_average_ppg**2, 2)

    delay = random.randint(5, 10)
    print(f"Waiting {delay} seconds")
    time.sleep(delay)

    return ppi_table


def get_fixtures(league):
    print(f"Getting fixtures for {league}")
    url = f"https://www.soccerstats.com/results.asp?league={league}"
    res = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        impersonate="safari_ios",
    )

    table_index = 10
    if league in ALT_INDEX_LEAGUES:
        table_index = 9

    df = (
        pd.read_html(res.content)[table_index]
        .dropna(how="all")
        .iloc[1:]
        .rename(columns={0: "Date", 1: "Home", 2: "Time", 3: "Away"})[
            ["Date", "Home", "Time", "Away"]
        ]
    )

    if df.empty:
        return df

    df = df[~df["Time"].str.contains(r"\d+\s*-\s*\d+", regex=True, na=False)]

    current_year = datetime.now().year
    df["Date"] = pd.to_datetime(df["Date"] + f" {current_year}", format="%a %d %b %Y")
    df["League"] = league

    delay = random.randint(5, 10)
    print(f"Waiting {delay} seconds")
    time.sleep(delay)

    print(f"Getting fixtures for dates between {TODAY} to {END_DATE}")

    return filter_date_range(df, TODAY, END_DATE)


def merge_metrics(fixtures, metrics):
    combined = fixtures.merge(metrics, left_on="Home", right_on="Team")
    combined = combined.merge(metrics, left_on="Away", right_on="Team").rename(
        columns={
            "PPI_x": "hPPI",
            "PPI_y": "aPPI",
            "GP_x": "hGP",
            "GP_y": "aGP",
            "PPI_x": "hPPI",
            "PPI_y": "aPPI",
            "PPINorm_x": "hPPINorm",
            "PPINorm_y": "aPPINorm",
        }
    )
    cols = [
        "League",
        "Date",
        "Time",
        "Home",
        "Away",
        "hGP",
        "aGP",
        "hPPI",
        "aPPI",
        "hPPINorm",
        "aPPINorm",
    ]
    combined = combined[cols]
    combined["hPPI"] = combined["hPPI"].astype(float)
    combined["aPPI"] = combined["aPPI"].astype(float)
    combined["PPI_DIFF"] = round(abs(combined["hPPI"] - combined["aPPI"]), 2)
    combined["PPI_DIFF_NORM"] = round(
        abs(combined["hPPINorm"] - combined["aPPINorm"]), 2
    )
    return combined.reset_index(drop=True).sort_values("PPI_DIFF")


# Main leagues
main_fixtures = pd.concat([get_fixtures(league) for league in main_leagues])
main_ppi_tables = pd.concat([get_ppi_tables(league) for league in main_leagues])
merge_metrics(main_fixtures, main_ppi_tables).to_csv(
    f"{TODAY}_PPI_LATEST_MAIN.csv", index=False
)

# Extra Leagues
extra_fixtures = pd.concat([get_fixtures(league) for league in extra_leagues])
extra_ppi_tables = pd.concat([get_ppi_tables(league) for league in extra_leagues])
merge_metrics(extra_fixtures, extra_ppi_tables).to_csv(
    f"{TODAY}_PPI_LATEST_EXTRA.csv", index=False
)
