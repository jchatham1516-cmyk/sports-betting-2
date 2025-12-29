# sports/common/odds_sources.py
from __future__ import annotations

import os
import csv
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional

import requests
import numpy as np

# Map your internal sport names to Odds API sport keys
SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _best_price_from_bookmakers(
    bookmakers: list,
    market_key: str,
    outcome_name: str,
) -> Optional[float]:
    """
    Grab the first available price for a named outcome in a given market.
    (Simple + stable: avoids tricky "best price" logic across books.)
    """
    try:
        for bk in bookmakers or []:
            for m in bk.get("markets", []) or []:
                if m.get("key") != market_key:
                    continue
                for o in m.get("outcomes", []) or []:
                    if str(o.get("name")).strip().lower() == str(outcome_name).strip().lower():
                        return _safe_float(o.get("price"), default=np.nan)
    except Exception:
        return None
    return None


def _spread_and_price_from_bookmakers(bookmakers: list, home_name: str) -> Tuple[float, float]:
    """
    Returns: (home_spread, spread_price)
    Spread market uses outcomes with 'point' and 'price'.
    """
    home_spread = np.nan
    spread_price = np.nan
    try:
        for bk in bookmakers or []:
            for m in bk.get("markets", []) or []:
                if m.get("key") != "spreads":
                    continue
                for o in m.get("outcomes", []) or []:
                    if str(o.get("name")).strip().lower() == str(home_name).strip().lower():
                        home_spread = _safe_float(o.get("point"), default=np.nan)
                        spread_price = _safe_float(o.get("price"), default=np.nan)
                        return (home_spread, spread_price)
    except Exception:
        pass
    return (home_spread, spread_price)


def _totals_from_bookmakers(bookmakers: list) -> Tuple[float, float, float]:
    """
    Returns: (total_points, over_price, under_price)
    Totals market uses outcomes 'Over'/'Under' with 'point' and 'price'.
    """
    total_points = np.nan
    over_price = np.nan
    under_price = np.nan
    try:
        for bk in bookmakers or []:
            for m in bk.get("markets", []) or []:
                if m.get("key") != "totals":
                    continue
                # totals point is shared; each outcome has point+price
                for o in m.get("outcomes", []) or []:
                    nm = str(o.get("name") or "").strip().lower()
                    pt = _safe_float(o.get("point"), default=np.nan)
                    pr = _safe_float(o.get("price"), default=np.nan)
                    if not np.isnan(pt):
                        total_points = pt
                    if nm == "over":
                        over_price = pr
                    elif nm == "under":
                        under_price = pr
                return (total_points, over_price, under_price)
    except Exception:
        pass
    return (total_points, over_price, under_price)


def fetch_odds_for_date_from_odds_api(
    game_date_str: str,
    *,
    sport_key: str,
    days_padding: int = 1,
    regions: str = "us",
    odds_format: str = "american",
    date_format: str = "iso",
    markets: str = "h2h,spreads,totals",
    sleep_s: float = 0.0,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], float]]:
    """
    Returns:
      odds_dict: {(home, away): {home_ml, away_ml, home_spread, spread_price, total_points, over_price, under_price, commence_time}}
      spreads_dict: {(home, away): home_spread}  (kept for backward compatibility)
    """
    api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY env var")

    # Parse date as MM/DD/YYYY (your runner format)
    dt = datetime.strptime(game_date_str, "%m/%d/%Y")
    start = (dt - timedelta(days=int(days_padding))).replace(hour=0, minute=0, second=0, microsecond=0)
    end = (dt + timedelta(days=int(days_padding))).replace(hour=23, minute=59, second=59, microsecond=0)

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Odds API HTTP {r.status_code}: {r.text[:200]}")

    events = r.json() or []
    odds_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    spreads_dict: Dict[Tuple[str, str], float] = {}

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        commence_time = ev.get("commence_time")
        bookmakers = ev.get("bookmakers", []) or []

        home_ml = _best_price_from_bookmakers(bookmakers, "h2h", home)
        away_ml = _best_price_from_bookmakers(bookmakers, "h2h", away)

        home_spread, spread_price = _spread_and_price_from_bookmakers(bookmakers, home)
        total_points, over_price, under_price = _totals_from_bookmakers(bookmakers)

        key = (str(home), str(away))
        odds_dict[key] = {
            "home_ml": _safe_float(home_ml),
            "away_ml": _safe_float(away_ml),
            "home_spread": _safe_float(home_spread),
            "spread_price": _safe_float(spread_price),
            "total_points": _safe_float(total_points),
            "over_price": _safe_float(over_price),
            "under_price": _safe_float(under_price),
            "commence_time": commence_time,
        }
        spreads_dict[key] = _safe_float(home_spread)

        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(game_date_str: str, *, sport: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], float]]:
    """
    CSV fallback.

    Expected file:
      odds/odds_MM-DD-YYYY.csv

    Expected columns:
      date,home,away,home_ml,away_ml,home_spread
    Optional:
      spread_price,total_points,total_over_price,total_under_price,commence_time
    """
    dt = datetime.strptime(game_date_str, "%m/%d/%Y")
    fname = f"odds/odds_{dt.strftime('%m-%d-%Y')}.csv"

    odds_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    spreads_dict: Dict[Tuple[str, str], float] = {}

    with open(fname, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = (row.get("home") or "").strip()
            away = (row.get("away") or "").strip()
            if not home or not away:
                continue

            key = (home, away)
            odds_dict[key] = {
                "home_ml": _safe_float(row.get("home_ml")),
                "away_ml": _safe_float(row.get("away_ml")),
                "home_spread": _safe_float(row.get("home_spread")),
                "spread_price": _safe_float(row.get("spread_price"), default=-110.0),
                "total_points": _safe_float(row.get("total_points")),
                "over_price": _safe_float(row.get("total_over_price"), default=-110.0),
                "under_price": _safe_float(row.get("total_under_price"), default=-110.0),
                "commence_time": row.get("commence_time"),
            }
            spreads_dict[key] = odds_dict[key]["home_spread"]

    return odds_dict, spreads_dict
