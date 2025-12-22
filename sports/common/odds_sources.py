# sports/common/odds_sources.py
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests

from sports.common.teams import canon_team

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_ODDS_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "pointsbetus", "caesars", "betrivers"]

SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}


def get_odds_api_key() -> str:
    k = (os.environ.get("ODDS_API_KEY") or "").strip()
    print("[odds_api] key present:", bool(k))
    print("[odds_api] key length:", len(k))  # safe: does not reveal key
    if not k:
        raise RuntimeError("ODDS_API_KEY environment variable is not set.")
    return k


def _iso_wide_bounds(game_date_str: str) -> Tuple[str, str]:
    """
    Wide UTC window so US evening games don't get missed by UTC rollover.
    Pulls from 00:00 UTC the day before through 23:59:59 UTC the day after.
    """
    d = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    start = datetime(d.year, d.month, d.day, 0, 0, 0) - timedelta(days=1)
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
    for m in (bookmaker.get("markets") or []):
        if m.get("key") == market_key:
            return m
    return None


def _request_odds(
    *,
    api_key: str,
    sport_key: str,
    regions: str,
    markets: str,
    odds_format: str,
    date_format: str,
    commence_from: str,
    commence_to: str,
) -> list:
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }

    r = requests.get(url, params=params, timeout=30)

    # DEBUG (safe)
    print("[odds_api DEBUG] status:", r.status_code)
    print("[odds_api DEBUG] url:", r.url)
    print("[odds_api DEBUG] remaining:", r.headers.get("x-requests-remaining"))
    print("[odds_api DEBUG] used:", r.headers.get("x-requests-used"))
    print("[odds_api DEBUG] last:", r.headers.get("x-requests-last"))

    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    print("[odds_api DEBUG] events returned:", len(data))
    return data


def fetch_odds_for_date_from_odds_api(
    game_date_str: str,
    *,
    sport_key: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
    preferred_books: Optional[list[str]] = None,
) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], float]]:
    """
    Returns:
      odds_dict[(home, away)] = {
          "home_ml": ...,
          "away_ml": ...,
          "home_spread": ...,
          "total_points": ...,
          "total_over_price": ...,
          "total_under_price": ...
      }
      spreads_dict[(home, away)] = home_spread

    IMPORTANT:
      Keys (home, away) are stored using canon_team() so they match your model's team names.
    """
    api_key = get_odds_api_key()
    preferred_books = preferred_books or DEFAULT_ODDS_BOOKMAKERS
    start_iso, end_iso = _iso_wide_bounds(game_date_str)

    data = _request_odds(
        api_key=api_key,
        sport_key=sport_key,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
        date_format=date_format,
        commence_from=start_iso,
        commence_to=end_iso,
    )

    # fallback if the requested markets were not returned
    if not data:
        print("[odds_api DEBUG] retrying with markets=h2h,spreads")
        data = _request_odds(
            api_key=api_key,
            sport_key=sport_key,
            regions=regions,
            markets="h2h,spreads",
            odds_format=odds_format,
            date_format=date_format,
            commence_from=start_iso,
            commence_to=end_iso,
        )

    odds_dict: Dict[Tuple[str, str], dict] = {}
    spreads_dict: Dict[Tuple[str, str], float] = {}

    for ev in data:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")

        # fallback if away_team missing
        if not away_raw:
            teams = ev.get("teams") or []
            if home_raw and len(teams) >= 2:
                away_raw = teams[0] if teams[1] == home_raw else teams[1]

        if not home_raw or not away_raw:
            continue

        home = canon_team(home_raw)
        away = canon_team(away_raw)

        books = ev.get("bookmakers") or []
        if not books:
            continue

        bookmaker = _pick_best_bookmaker(books, preferred_books)
        if not bookmaker:
            continue

        # Initialize structure
        home_ml = np.nan
        away_ml = np.nan
        home_spread = np.nan

        total_points = np.nan
        total_over_price = np.nan
        total_under_price = np.nan

        # --- Moneyline (h2h) ---
        h2h = _extract_market(bookmaker, "h2h")
        if h2h:
            prices = {}
            for o in (h2h.get("outcomes") or []):
                name = o.get("name")
                price = o.get("price")
                if name is None or price is None:
                    continue
                prices[canon_team(str(name))] = price

            if prices.get(home) is not None:
                home_ml = float(prices[home])
            if prices.get(away) is not None:
                away_ml = float(prices[away])

        # --- Spreads ---
        spreads = _extract_market(bookmaker, "spreads")
        if spreads:
            pts = {}
            for o in (spreads.get("outcomes") or []):
                name = o.get("name")
                point = o.get("point")
                if name is None or point is None:
                    continue
                pts[canon_team(str(name))] = point

            if pts.get(home) is not None:
                home_spread = float(pts[home])

        # --- Totals (Over/Under) ---
        totals = _extract_market(bookmaker, "totals")
        if totals:
            for o in (totals.get("outcomes") or []):
                name = str(o.get("name") or "").strip().lower()
                point = o.get("point")
                price = o.get("price")

                # point is required to understand the total
                if point is not None:
                    try:
                        pt = float(point)
                        if np.isnan(total_points):
                            total_points = pt  # set once
                    except Exception:
                        pass

                # price is optional (some feeds may omit it)
                if price is not None:
                    try:
                        pr = float(price)
                    except Exception:
                        pr = np.nan
                else:
                    pr = np.nan

                if "over" in name:
                    total_over_price = pr
                elif "under" in name:
                    total_under_price = pr

        key = (home, away)
        odds_dict[key] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "total_points": total_points,
            "total_over_price": total_over_price,
            "total_under_price": total_under_price,
        }
        spreads_dict[key] = home_spread

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(game_date_str: str, *, sport: str = "nba"):
    """
    Tries:
      - odds/odds_<sport>_MM-DD-YYYY.csv
      - odds/odds_MM-DD-YYYY.csv

    Required columns: home, away, home_ml, away_ml
    Optional: home_spread, total_points, total_over_price, total_under_price

    IMPORTANT:
      Keys (home, away) are stored using canon_team() so they match your model.
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
        home = canon_team(str(row["home"]))
        away = canon_team(str(row["away"]))
        key = (home, away)

        home_ml = _parse_number(row.get("home_ml"))
        away_ml = _parse_number(row.get("away_ml"))
        home_spread = _parse_number(row.get("home_spread")) if "home_spread" in df.columns else np.nan

        total_points = _parse_number(row.get("total_points")) if "total_points" in df.columns else np.nan
        total_over_price = _parse_number(row.get("total_over_price")) if "total_over_price" in df.columns else np.nan
        total_under_price = _parse_number(row.get("total_under_price")) if "total_under_price" in df.columns else np.nan

        odds_dict[key] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "total_points": total_points,
            "total_over_price": total_over_price,
            "total_under_price": total_under_price,
        }
        spreads_dict[key] = home_spread

    return odds_dict, spreads_dict
