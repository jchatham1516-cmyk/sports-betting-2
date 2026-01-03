# grade_predictions.py
import argparse
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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


def extract_games(json_data: dict) -> List[Dict[str, object]]:
    games: List[Dict[str, object]] = []
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
            games.append(
                {
                    "home": home[0],
                    "away": away[0],
                    "home_score": home[1],
                    "away_score": away[1],
                }
            )
    return games


def _pick_type_from_rec(rec_upper: str) -> str:
    # Handles your strings like:
    # "Model PICK: HOME ML (strong)"
    # "Model PICK ATS: AWAY (lean)"
    # "No ML bet ..."
    if "HOME ML" in rec_upper or "AWAY ML" in rec_upper:
        return "ML"
    if "ATS" in rec_upper:
        return "ATS"
    if "TOTAL" in rec_upper or "OVER" in rec_upper or "UNDER" in rec_upper:
        return "TOTAL"
    return "UNKNOWN"


def grade_row(row: pd.Series, games: List[Dict[str, object]]) -> Tuple[Optional[bool], Optional[float], Optional[float], Optional[str]]:
    rec = str(row.get("primary_recommendation", "")).upper()
    if rec.strip() == "" or "NO BET" in rec or rec.startswith("NO "):
        return None, None, None, None  # SKIP

    row_home = norm_team(row.get("home", ""))
    row_away = norm_team(row.get("away", ""))

    for g in games:
        if g["home"] == row_home and g["away"] == row_away:
            home_score = float(g["home_score"])
            away_score = float(g["away_score"])
            home_win = home_score > away_score
            away_win = away_score > home_score

            # Moneyline
            if "HOME ML" in rec:
                return home_win, home_score, away_score, "HOME"
            if "AWAY ML" in rec:
                return away_win, home_score, away_score, "AWAY"

            # ATS (spread) — assumes row["home_spread"] is the market home spread (e.g. -6.5 or +3.0)
            if "ATS" in rec:
                try:
                    spread = float(row.get("home_spread"))
                except Exception:
                    return None, home_score, away_score, None

                adj_home = home_score + spread
                if "HOME" in rec:
                    return (adj_home > away_score), home_score, away_score, "HOME"
                if "AWAY" in rec:
                    return (adj_home < away_score), home_score, away_score, "AWAY"

            # Totals could be added later if you want (needs total line + over/under side)
            return None, home_score, away_score, None

    return None, None, None, None  # no matching game -> SKIP


def grade_file(csv_path: str, sport: str) -> str:
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError(f"{csv_path} missing required column: date")
    if "home" not in df.columns or "away" not in df.columns:
        raise ValueError(f"{csv_path} missing required columns: home/away")

    # Normalize date column to YYYY-MM-DD
    df["date"] = df["date"].apply(parse_date_any)

    # ESPN pull per-date (handles files that might include multiple days)
    graded_rows: List[Tuple[int, Optional[bool], Optional[float], Optional[float], Optional[str]]] = []
    for date_val, sub in df.groupby("date", sort=False):
        results = fetch_espn_results(sport, date_val)
        games = extract_games(results)

        for idx, row in sub.iterrows():
            res, hs, as_, pick_side = grade_row(row, games)
            graded_rows.append((int(idx), res, hs, as_, pick_side))

    # Write back in original order
    df["home_score"] = None
    df["away_score"] = None
    df["pick_side"] = None
    df["pick_type"] = df.get("primary_recommendation", "").astype(str).str.upper().apply(_pick_type_from_rec)
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

    played = df[df["graded"].isin(["WIN", "LOSS"])]
    wins = int((played["graded"] == "WIN").sum())
    losses = int((played["graded"] == "LOSS").sum())
    print(f"{csv_path} -> {out} | graded bets: {wins + losses} | W-L: {wins}-{losses}")

    return out


def _list_prediction_files(preds_dir: str, sport: str) -> List[str]:
    # Your model saves: results/predictions_{sport}_{MM-DD-YYYY}.csv
    patt = os.path.join(preds_dir, f"predictions_{sport}_*.csv")
    return sorted(glob.glob(patt))


def _parse_date_from_filename(path: str) -> Optional[datetime]:
    # expects ...predictions_nba_01-02-2026.csv
    m = re.search(r"predictions_[a-z]+_(\d{2}-\d{2}-\d{4})\.csv$", os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%m-%d-%Y")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Grade model predictions using ESPN scoreboard results.")
    parser.add_argument("--sport", type=str, default="all", choices=["all", "nba", "nfl", "nhl"])
    parser.add_argument("--preds_dir", type=str, default="results", help="Folder where predictions_*.csv are saved (default: results).")
    parser.add_argument("--file", type=str, default=None, help="Grade a specific CSV path. If provided, ignores --sport scan.")
    parser.add_argument("--limit", type=int, default=0, help="If scanning files, only grade the most recent N per sport (0 = all).")
    args = parser.parse_args()

    if args.file:
        # Try to infer sport from filename if possible
        inferred = None
        base = os.path.basename(args.file).lower()
        for s in ("nba", "nfl", "nhl"):
            if f"predictions_{s}_" in base:
                inferred = s
                break
        sport = inferred or "nba"  # fallback
        grade_file(args.file, sport)
        return

    sports = ("nba", "nfl", "nhl") if args.sport == "all" else (args.sport,)
    for sport in sports:
        paths = _list_prediction_files(args.preds_dir, sport)
        if not paths:
            print(f"[skip] no files found for {sport} in {args.preds_dir}")
            continue

        # newest-first if limit
        if args.limit and args.limit > 0:
            paths_sorted = sorted(paths, key=lambda p: _parse_date_from_filename(p) or datetime.min, reverse=True)
            paths = list(reversed(paths_sorted[: int(args.limit)]))  # grade oldest->newest for nicer logs

        for path in paths:
            try:
                grade_file(path, sport)
            except Exception as e:
                print(f"[error] failed grading {path}: {e}")


if __name__ == "__main__":
    main()
