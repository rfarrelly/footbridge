import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from curl_cffi import requests
from utils.datetime_helpers import filter_date_range

# ==========================================
# LOGGING SETUP
# ==========================================


class Tee:
    """Write output to both the console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


LOG_FILE = Path.cwd() / (f"soccerstats_{datetime.now():%Y%m%d_%H%M%S}.log")

log_handle = open(LOG_FILE, "w", encoding="utf-8", buffering=1)

sys.stdout = Tee(sys.__stdout__, log_handle)
sys.stderr = Tee(sys.__stderr__, log_handle)

print("=" * 60)
print("SOCCERSTATS SCRAPER")
print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"Log: {LOG_FILE.resolve()}")
print("=" * 60)


# ==========================================
# LEAGUE CONFIGURATION
# ==========================================

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

# (country_slug, league_slug)
LEAGUE_URLS = {
    "england": ("england", "premier-league"),
    "england2": ("england", "championship"),
    "england3": ("england", "league-one"),
    "england4": ("england", "league-two"),
    "england5": ("england", "national-league"),
    "belgium": ("belgium", "pro-league"),
    "germany": ("germany", "bundesliga"),
    "germany2": ("germany", "2-bundesliga"),
    "spain": ("spain", "laliga"),
    "spain2": ("spain", "laliga2"),
    "france": ("france", "ligue-1"),
    "france2": ("france", "ligue-2"),
    "greece": ("greece", "super-league"),
    "netherlands": ("netherlands", "eredivisie"),
    "netherlands2": ("netherlands", "eerste-divisie"),
    "italy": ("italy", "serie-a"),
    "italy2": ("italy", "serie-b"),
    "portugal": ("portugal", "liga-portugal"),
    "portugal2": ("portugal", "liga-portugal-2"),
    "scotland": ("scotland", "premiership"),
    "scotland2": ("scotland", "championship"),
    "scotland3": ("scotland", "league-one"),
    "scotland4": ("scotland", "league-two"),
    "turkey": ("turkey", "super-lig"),
}


# ==========================================
# DATE AND REQUEST CONFIGURATION
# ==========================================

TODAY = pd.Timestamp.today().normalize()
TIME_DELTA = 4
END_DATE = TODAY + pd.Timedelta(days=TIME_DELTA)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ==========================================
# GET PPI DATA
# ==========================================


def get_ppi(league):
    if league not in LEAGUE_URLS:
        print(f"[PPI] {league}: skipped (no URL mapping)")
        return pd.DataFrame()

    country_slug, league_slug = LEAGUE_URLS[league]

    if "PLACEHOLDER" in (country_slug, league_slug):
        print(f"[PPI] {league}: skipped (placeholder URL)")
        return pd.DataFrame()

    url = (
        f"https://www.soccerstats.com/en-gb/leagues/"
        f"{country_slug}/{league_slug}/{league}/table/pointsperformance"
    )

    try:
        res = requests.get(
            url,
            headers=HEADERS,
            impersonate="safari_ios",
        )
        res.raise_for_status()

        tables = pd.read_html(res.content)

        ppi_table = None

        for table in tables:
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = [
                    " ".join(
                        str(part)
                        for part in col
                        if str(part) != "nan" and not str(part).startswith("Unnamed")
                    ).strip()
                    for col in table.columns
                ]
            else:
                table.columns = [str(col).strip() for col in table.columns]

            headings = " ".join(table.columns).lower()

            if (
                "team" in headings
                and "opponents" in headings
                and "points performance index" in headings
            ):
                ppi_table = table.copy()
                break

        if ppi_table is None:
            print(f"[PPI] {league}: failed (PPI table not found)")
            return pd.DataFrame()

        def find_column(keyword, exclude=None):
            exclude = exclude or []

            for col in ppi_table.columns:
                name = str(col).lower()

                if keyword in name and not any(x in name for x in exclude):
                    return col

            return None

        team_col = find_column("team")
        gp_col = find_column("gp")
        team_ppg_col = find_column("team ppg")
        opps_ppg_col = find_column("opponents ppg")
        ppi_col = find_column("points performance index")

        required = {
            "Team": team_col,
            "GP": gp_col,
            "TeamPPG": team_ppg_col,
            "OppsPPG": opps_ppg_col,
            "PPI": ppi_col,
        }

        missing = [name for name, col in required.items() if col is None]

        if missing:
            print(f"[PPI] {league}: failed (missing columns: {missing})")
            return pd.DataFrame()

        df = ppi_table[
            [
                team_col,
                gp_col,
                team_ppg_col,
                opps_ppg_col,
                ppi_col,
            ]
        ].copy()

        df.columns = [
            "Team",
            "GP",
            "TeamPPG",
            "OppsPPG",
            "PPI",
        ]

        df = df.dropna(subset=["Team"])

        for col in ["GP", "TeamPPG", "OppsPPG", "PPI"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["PPI"])

        df["League"] = league

        delay = random.randint(5, 10)
        time.sleep(delay)

        print(f"[PPI] {league}: {len(df)} teams collected")

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"[PPI] {league}: error ({e})")
        return pd.DataFrame()


# ==========================================
# GET FIXTURES
# ==========================================


def get_fixtures(league):
    empty_df = pd.DataFrame(columns=["Date", "Home", "Time", "Away", "League"])

    if league not in LEAGUE_URLS:
        print(f"[Fixtures] {league}: skipped (no URL mapping)")
        return empty_df

    country_slug, league_slug = LEAGUE_URLS[league]

    if "PLACEHOLDER" in country_slug or "PLACEHOLDER" in league_slug:
        print(f"[Fixtures] {league}: skipped (placeholder URL)")
        return empty_df

    url = (
        f"https://www.soccerstats.com/nodeweb/en-gb/leagues/"
        f"{country_slug}/{league_slug}/{league}/matchlist/bymonth"
    )

    try:
        res = requests.get(
            url,
            headers=HEADERS,
            impersonate="safari_ios",
        )
        res.raise_for_status()

        tables = pd.read_html(res.content)

        fixtures = []

        date_pattern = r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)" r"\s+\d{1,2}\s+[A-Za-z]{3}$"

        time_pattern = r"^\d{1,2}:\d{2}$"

        for table in tables:
            for _, row in table.iterrows():
                if len(row) < 3:
                    continue

                match_date = str(row.iloc[0]).strip()
                match_time = str(row.iloc[1]).strip()
                fixture = str(row.iloc[2]).strip()

                if not re.match(
                    date_pattern,
                    match_date,
                    re.IGNORECASE,
                ):
                    continue

                if not re.match(time_pattern, match_time):
                    continue

                teams = re.split(
                    r"\s+vs\s+",
                    fixture,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )

                if len(teams) != 2:
                    continue

                home = teams[0].strip()
                away = teams[1].strip()

                fixtures.append(
                    {
                        "Date": match_date,
                        "Home": home,
                        "Time": match_time,
                        "Away": away,
                        "League": league,
                    }
                )

        df = pd.DataFrame(fixtures)

        if df.empty:
            print(f"[Fixtures] {league}: no upcoming fixtures")
            return empty_df

        current_year = datetime.now().year

        df["Date"] = pd.to_datetime(
            df["Date"] + f" {current_year}",
            format="%a %d %b %Y",
        )

        df.loc[
            df["Date"] < TODAY - timedelta(days=180),
            "Date",
        ] += pd.DateOffset(years=1)

        df = filter_date_range(df, TODAY, END_DATE)

        delay = random.randint(5, 10)
        time.sleep(delay)

        print(f"[Fixtures] {league}: {len(df)} fixtures collected")

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"[Fixtures] {league}: error ({e})")
        return empty_df


# ==========================================
# MERGE FIXTURES AND PPI
# ==========================================


def merge_metrics(fixtures, metrics):
    fixtures = fixtures.drop_duplicates(
        subset=["League", "Date", "Time", "Home", "Away"]
    ).copy()

    home_metrics = metrics[["League", "Team", "GP", "PPI"]].rename(
        columns={
            "Team": "Home",
            "GP": "hGP",
            "PPI": "hPPI",
        }
    )

    away_metrics = metrics[["League", "Team", "GP", "PPI"]].rename(
        columns={
            "Team": "Away",
            "GP": "aGP",
            "PPI": "aPPI",
        }
    )

    combined = fixtures.merge(
        home_metrics,
        on=["League", "Home"],
        how="inner",
        validate="many_to_one",
    )

    combined = combined.merge(
        away_metrics,
        on=["League", "Away"],
        how="inner",
        validate="many_to_one",
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
    ]

    combined = combined[cols].copy()

    combined["hPPI"] = pd.to_numeric(
        combined["hPPI"],
        errors="coerce",
    )

    combined["aPPI"] = pd.to_numeric(
        combined["aPPI"],
        errors="coerce",
    )

    combined["PPI_DIFF"] = (combined["hPPI"] - combined["aPPI"]).abs().round(2)

    return combined.sort_values("PPI_DIFF").reset_index(drop=True)


# ==========================================
# MAIN: GET FIXTURES
# ==========================================

print(f"\nDate range: {TODAY.date()} to {END_DATE.date()}")
print("\nCollecting fixtures...")

fixtures = []

for league in main_leagues:
    try:
        fixtures.append(get_fixtures(league))
    except Exception as e:
        print(f"[Fixtures] {league}: error ({e})")
        continue

main_fixtures = pd.concat(
    fixtures,
    ignore_index=True,
)

print(f"\nTotal fixtures collected: {len(main_fixtures)}")


# ==========================================
# MAIN: GET PPI TABLES
# ==========================================

print("\nCollecting PPI tables...")

ppi_tables = []

for league in main_leagues:
    try:
        ppi_tables.append(get_ppi(league))
    except Exception as e:
        print(f"[PPI] {league}: error ({e})")
        continue

main_ppi_tables = pd.concat(
    ppi_tables,
    ignore_index=True,
)

print(f"\nTotal PPI teams collected: {len(main_ppi_tables)}")


# ==========================================
# MERGE AND EXPORT
# ==========================================

print("\nMerging fixtures and PPI data...")

merged = merge_metrics(
    main_fixtures,
    main_ppi_tables,
)

print(f"Fixtures matched: {len(merged)}")

output_file = f"{TODAY}_PPI_LATEST_MAIN.csv"

merged.to_csv(
    output_file,
    index=False,
)

print("\n" + "=" * 60)
print("SCRAPER COMPLETED")
print(f"Fixtures exported: {len(merged)}")
print(f"Output file: {output_file}")
print(f"Completed: {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"Log file: {LOG_FILE.resolve()}")
print("=" * 60)

log_handle.flush()
