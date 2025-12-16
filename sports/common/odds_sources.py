import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_ODDS_BOOKMAKERS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "pointsbetus",
    "caesars",
    "betrivers",
]

SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}


# -----------------------------
# API key
# -----------------------------

def get_odds_api_key() -> str:
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    print("[odds_api] key present:", bool(key))
    print("[odds_api] key length:", len(key))
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set")
    return key


# -----------------------------
# Helpers
# -----------------------------

def _iso_wide_bounds(game_date_str: str) -> Tuple[str, str]:
    """
    Wide UTC window to avoid US-time → UTC rollover issues.
    """
    d = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    start = datetime(d.year, d.month, d.day) - timedelta(days=1)
    end = datetime(d.year, d.month, d.day, 23, 59, 59) + timedelta(days=1)
    return start.isoformat() + "Z", end.isoformat() + "Z"


def _pick_best_bookmaker(bookmakers: list, preferred: list[str]) -> Optional[dict]:
    if not bookmakers:
        return None
    by_key = {b.get("key"): b for b in bookmakers if b.get("key")}
    for k in preferred:
        if k in by_key:
            return by_key[k]
    return bookmakers[0]


def _extract_market(bookmaker: dict, market_key: str) -> Optional[dict]:
    for m in bookmaker.get("markets", []) or []:
        if m.get("key") == market_key:
            return m
    return None


# -----------------------------
# Odds API
# -----------------------------

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
    api_key = get_odds_api_key()
    preferred_books = preferred_books or DEFAULT_ODDS_BOOKMAKERS

    start_iso, end_iso = _iso_wide_bounds(game_date_str)
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"

    def _call(markets_str: str):
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": markets_str,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "commenceTimeFrom": start_iso,
            "commenceTimeTo": end_iso,
        }
        r = requests.get(url, params=params, timeout=30)

        print("[odds_api DEBUG] status:", r.status_code)
        print("[odds_api DEBUG] url:", r.url)
        print("[odds_api DEBUG] remaining:", r.headers.get("x-requests-remaining"))
        print("[odds_api DEBUG] used:", r.headers.get("x-requests-used"))

        r.raise_for_status()
        data = r.json()
        print("[odds_api DEBUG] events returned:", len(data))
        return data

    def _parse(data):
        odds_dict = {}
        spreads_dict = {}

        skipped_no_books = 0

        for ev in data:
            home = ev.get("home_team")
            teams = ev.get("teams") or []
            if not home or len(teams) != 2:
                continue

            away = teams[0] if teams[1] == home else teams[1]

            books = ev.get("bookmakers") or []
            if not books:
                skipped_no_books += 1
                continue

            bookmaker = _pick_best_bookmaker(books, preferred_books)
            if not bookmaker:
                continue

            # Moneyline (h2h)
            home_ml = np.nan
            away_ml = np.nan
            h2h = _extract_market(bookmaker, "h2h")
            if h2h:
                prices = {o.get("name"): o.get("price") for o in (h2h.get("outcomes") or [])}
                if prices.get(home) is not None:
                    home_ml = float(prices[home])
                if prices.get(away) is not None:
                    away_ml = float(prices[away])

            # Spreads (optional)
            home_spread = np.nan
            spreads = _extract_market(bookmaker, "spreads")
            if spreads:
                pts = {o.get("name"): o.get("point") for o in (spreads.get("outcomes") or [])}
                if pts.get(home) is not None:
                    home_spread = float(pts[home])

            key = (home, away)
            odds_dict[key] = {"home_ml": home_ml, "away_ml": away_ml, "home_spread": home_spread}
            spreads_dict[key] = home_spread

        if skipped_no_books:
            print("[odds_api DEBUG] events w/ no bookmakers:", skipped_no_books)

        return odds_dict, spreads_dict

    # 1) Try h2h + spreads
    data = _call("h2h,spreads")
    odds_dict, spreads_dict = _parse(data)

    # 2) If we got events but parsed nothing, retry with h2h only
    if len(data) > 0 and not odds_dict:
        print("[odds_api DEBUG] retrying with markets=h2h only (some sports/books omit spreads)")
        data2 = _call("h2h")
        odds_dict, spreads_dict = _parse(data2)

    return odds_dict, spreads_dict

# -----------------------------
# CSV fallback
# -----------------------------

def fetch_odds_for_date_from_csv(game_date_str: str, *, sport: str = "nba"):
    date_part = game_date_str.replace("/", "-")
    fname1 = f"odds/odds_{sport}_{date_part}.csv"
    fname2 = f"odds/odds_{date_part}.csv"

    fname = fname1 if os.path.exists(fname1) else fname2
    if not os.path.exists(fname):
        print("[odds_csv] No odds file found")
        return {}, {}

    print(f"[odds_csv] Loading odds from {fname}")
    df = pd.read_csv(fname)

    odds_dict = {}
    spreads_dict = {}

    for _, r in df.iterrows():
        key = (r["home"], r["away"])
        odds_dict[key] = {
            "home_ml": r["home_ml"],
            "away_ml": r["away_ml"],
            "home_spread": r.get("home_spread", np.nan),
        }
        spreads_dict[key] = r.get("home_spread", np.nan)

    return odds_dict, spreads_dict
