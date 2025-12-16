import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

print("[odds_api] key present:", bool(os.environ.get("ODDS_API_KEY")))

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_ODDS_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "pointsbetus", "caesars", "betrivers"]

SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}


def get_odds_api_key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    print("[odds_api] key length:", len(k))  # SAFE: does not reveal the key
    if not k:
        raise RuntimeError("ODDS_API_KEY environment variable is not set.")
    return k.strip()

def _iso_day_bounds(game_date_str: str) -> Tuple[str, str]:
    d = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    start = datetime(d.year, d.month, d.day, 0, 0, 0).isoformat() + "Z"
    end = datetime(d.year, d.month, d.day, 23, 59, 59).isoformat() + "Z"
    return start, end


def _pick_best_bookmaker(bookmakers: list, preferred: list[str]) -> Optional[dict]:
    if not bookmakers:
        return None
    by_key = {b.get("key"): b for b in bookmakers if b.get("key")}
    for k in preferred:
        if k in by_key:
            return by_key[k]
    return bookmakers[0]


def _extract_market(bookmaker: dict, market_key: str) -> Optional[dict]:
    markets = bookmaker.get("markets", []) or []
    for m in markets:
        if m.get("key") == market_key:
            return m
    return None

def _iso_wide_bounds(game_date_str: str) -> tuple[str, str]:
    """
    Wide UTC bounds to avoid Odds API timezone rollover issues.
    Pulls ±1 day around the local date.
    """
    d = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    start = datetime(d.year, d.month, d.day) - timedelta(days=1)
    end = datetime(d.year, d.month, d.day, 23, 59, 59) + timedelta(days=1)
    return start.isoformat() + "Z", end.isoformat() + "Z"

def fetch_odds_for_date_from_odds_api(
    game_date_str: str,
    *,
    sport_key: str,
    regions: str = "us",
    markets: str = "h2h,spreads",
    odds_format: str = "american",
    date_format: str = "iso",
    preferred_books: Optional[list[str]] = None,
) -> tuple[dict, dict]:
    """
    Returns (odds_dict, spreads_dict) keyed by (home_team, away_team)

      odds_dict[(home, away)] = {"home_ml": ..., "away_ml": ..., "home_spread": ...}
      spreads_dict[(home, away)] = home_spread
    """
    api_key = get_odds_api_key()
    preferred_books = preferred_books or DEFAULT_ODDS_BOOKMAKERS

    start_iso, end_iso = _iso_wide_bounds(game_date_str)

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": start_iso,
        "commenceTimeTo": end_iso,
    }

    r = requests.get(url, params=params, timeout=30)
    from datetime import datetime, timedelta  # make sure timedelta is imported

def _iso_wide_bounds(game_date_str: str) -> tuple[str, str]:
    """
    Use a wide UTC window so "today" in US time doesn't miss games that start after midnight UTC.
    """
    d = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    start = datetime(d.year, d.month, d.day, 0, 0, 0) - timedelta(days=1)
    end = datetime(d.year, d.month, d.day, 23, 59, 59) + timedelta(days=1)
    return start.isoformat() + "Z", end.isoformat() + "Z"

        # DEBUG: Odds API response info
    print("[odds_api DEBUG] status:", r.status_code)
    print("[odds_api DEBUG] remaining:", r.headers.get("x-requests-remaining"))
    print("[odds_api DEBUG] used:", r.headers.get("x-requests-used"))
    print("[odds_api DEBUG] last:", r.headers.get("x-requests-last"))
    print("[odds_api DEBUG] url:", r.url)
    
    r.raise_for_status()
    data = r.json()

    odds_dict: dict = {}
    spreads_dict: dict = {}

    for ev in data:
        home = ev.get("home_team")
        teams = ev.get("teams", []) or []
        if not home or len(teams) != 2:
            continue
        away = teams[0] if teams[1] == home else teams[1]

        bookmaker = _pick_best_bookmaker(ev.get("bookmakers", []) or [], preferred_books)
        if not bookmaker:
            continue

        # Moneyline (h2h)
        h2h = _extract_market(bookmaker, "h2h")
        home_ml = np.nan
        away_ml = np.nan
        if h2h:
            outs = h2h.get("outcomes", []) or []
            prices = {o.get("name"): o.get("price") for o in outs}
            if home in prices and prices[home] is not None:
                home_ml = float(prices[home])
            if away in prices and prices[away] is not None:
                away_ml = float(prices[away])

        # Spreads
        spreads = _extract_market(bookmaker, "spreads")
        home_spread = np.nan
        if spreads:
            outs = spreads.get("outcomes", []) or []
            points = {o.get("name"): o.get("point") for o in outs}
            if home in points and points[home] is not None:
                home_spread = float(points[home])

        key = (home, away)
        odds_dict[key] = {"home_ml": home_ml, "away_ml": away_ml, "home_spread": home_spread}
        spreads_dict[key] = home_spread

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(game_date_str: str, *, sport: str = "nba"):
    """
    Backwards compatible:
      - tries: odds/odds_<sport>_MM-DD-YYYY.csv
      - else: odds/odds_MM-DD-YYYY.csv  (your current naming)

    Required columns: home, away, home_ml, away_ml
    Optional: home_spread
    """
    date_part = game_date_str.replace("/", "-")
    fname1 = os.path.join("odds", f"odds_{sport}_{date_part}.csv")
    fname2 = os.path.join("odds", f"odds_{date_part}.csv")

    fname = fname1 if os.path.exists(fname1) else fname2

    if not os.path.exists(fname):
        print(f"[odds_csv] No odds file found at {fname1} or {fname2}.")
        return {}, {}

    print(f"[odds_csv] Loading odds from {fname}")
    df = pd.read_csv(fname)

    required_cols = {"home", "away", "home_ml", "away_ml"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Odds CSV {fname} must contain columns: {required_cols}. Found: {list(df.columns)}")

    def _parse_number(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    odds_dict = {}
    spreads_dict = {}

    for _, row in df.iterrows():
        home = str(row["home"]).strip()
        away = str(row["away"]).strip()
        key = (home, away)

        home_ml = _parse_number(row.get("home_ml"))
        away_ml = _parse_number(row.get("away_ml"))
        home_spread = _parse_number(row.get("home_spread")) if "home_spread" in df.columns else np.nan

        odds_dict[key] = {"home_ml": home_ml, "away_ml": away_ml, "home_spread": home_spread}
        spreads_dict[key] = home_spread

    return odds_dict, spreads_dict

