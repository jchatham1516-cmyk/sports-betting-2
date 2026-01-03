import glob
import re
from datetime import datetime

import pandas as pd
import requests

SPORT_ENDPOINTS = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
}

# Simple alias normalization to avoid ESPN vs model naming mismatches.
# Add more as you see SKIPs due to name differences.
ALIASES = {
    # NBA common
    "LA CLIPPERS": "LOS ANGELES CLIPPERS",
    "LA LAKERS": "LOS ANGELES LAKERS",
    "NY KNICKS": "NEW YORK KNICKS",
    "GS WARRIORS": "GOLDEN STATE WARRIORS",
    "SA SPURS": "SAN ANTONIO SPURS",

    # NFL common
    "LA RAMS": "LOS ANGELES RAMS",
    "LA CHARGERS": "LOS ANGELES CHARGERS",

    # NHL common
    "LA KINGS": "LOS ANGELES KINGS",
    "NJ DEVILS": "NEW JERSEY DEVILS",
    "TB LIGHTNING": "TAMPA BAY LIGHTNING",
}

def norm_team(name: str) -> str:
    s = str(name).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return ALIASES.get(s, s)

def parse_date_any(s: str) -> str:
    """Return YYYY-MM-DD from common formats."""
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {s}")

def fetch_espn_results(sport: str, date_yyyy_mm_dd: str) -> dict:
    date_fmt = datetime.strptime(date_yyyy_mm_dd, "%Y-%m-%d").strftime("%Y%m%d")
    url = f"{SPORT_ENDPOINTS[sport]}?dates={date_fmt}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def extract_games(json_data: dict):
    games = []
    for ev in json_data.get("events", []):
        comp = ev["competitions"][0]
        if not comp["status"]["type"]["completed"]:
            continue

        home = away = None
        for c in comp["competitors"]:
            team = norm_team(c["team"]["displayName"])
            score = float(c["score"])
            if c["homeAway"] == "home":
                home = (team, score)
            else:
                away = (team, score)

        if home and away:
            games.append({
                "home": home[0],
                "away": away[0],
                "home_score": home[1],
                "away_score": away[1],
            })
    return games

def grade_row(row, games):
    rec = str(row.get("primary_recommendation", "")).upper()
    if "NO BET" in rec or rec.strip() == "":
        return None, None, None, None  # SKIP

    row_home = norm_team(row["home"])
    row_away = norm_team(row["away"])

    for g in games:
        if g["home"] == row_home and g["away"] == row_away:
            home_win = g["home_score"] > g["away_score"]
            away_win = g["away_score"] > g["home_score"]

            # Moneyline
            if "HOME ML" in rec:
                return True, g["home_score"], g["away_score"], "HOME"
            if "AWAY ML" in rec:
                return True, g["home_score"], g["away_score"], "AWAY"

            # ATS
            if "ATS" in rec:
                try:
                    spread = float(row["home_spread"])
                except Exception:
                    return None, g["home_score"], g["away_score"], None

                adj_home = g["home_score"] + spread
                if "HOME" in rec:
                    return (adj_home > g["away_score"]), g["home_score"], g["away_score"], "HOME"
                if "AWAY" in rec:
                    return (adj_home < g["away_score"]), g["home_score"], g["away_score"], "AWAY"

            # Totals (optional: only if you have total_points + total_pick_side)
            # You can add later.

            return None, g["home_score"], g["away_score"], None

    return None, None, None, None  # no matching game -> SKIP

def grade_file(csv_path: str, sport: str):
    df = pd.read_csv(csv_path)

    # Normalize date column to YYYY-MM-DD
    df["date"] = df["date"].apply(parse_date_any)

    # ESPN pull per-date (handles files that might include multiple days)
    graded_rows = []
    for date_val, sub in df.groupby("date", sort=False):
        results = fetch_espn_results(sport, date_val)
        games = extract_games(results)

        for idx, row in sub.iterrows():
            res, hs, as_, pick_side = grade_row(row, games)
            graded_rows.append((idx, res, hs, as_, pick_side))

    # Write back in original order
    df["home_score"] = None
    df["away_score"] = None
    df["pick_side"] = None
    df["graded"] = "SKIP"

    for idx, res, hs, as_, pick_side in graded_rows:
        df.at[idx, "home_score"] = hs
        df.at[idx, "away_score"] = as_
        df.at[idx, "pick_side"] = pick_side
        if res is True:
            df.at[idx, "graded"] = "WIN"
        elif res is False:
            df.at[idx, "graded"] = "LOSS"
        else:
            df.at[idx, "graded"] = "SKIP"

    out = csv_path.replace(".csv", "_graded.csv")
    df.to_csv(out, index=False)

    # quick summary
    played = df[df["graded"].isin(["WIN", "LOSS"])]
    wins = (played["graded"] == "WIN").sum()
    losses = (played["graded"] == "LOSS").sum()
    print(f"{csv_path} -> {out} | graded bets: {wins+losses} | W-L: {wins}-{losses}")

def main():
    # Grade all files in current folder like predictions_nba_*.csv, predictions_nfl_*.csv, predictions_nhl_*.csv
    for sport in ("nba", "nfl", "nhl"):
        for path in sorted(glob.glob(f"predictions_{sport}_*.csv")):
            grade_file(path, sport)

if __name__ == "__main__":
    main()
