# sports/common/odds_sources.py
from __future__ import annotations

import os
import csv
import math
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional

import requests

# Map internal sport -> Odds API sport key
SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"


def _safe_float(x, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _pick_best_bookmaker(bookmakers: list) -> Optional[dict]:
    """
    Pick a single bookmaker entry from the list.
    Preference order:
      1) a bookmaker whose key is in env ODDS_BOOK_PREF (comma separated)
      2) first bookmaker in list
    """
    if not bookmakers:
        return None

    pref = os.getenv("ODDS_BOOK_PREF", "")
    pref_keys = [p.strip() for p in pref.split(",") if p.strip()]
    if pref_keys:
        for k in pref_keys:
            for b in bookmakers:
                if (b or {}).get("key") == k:
                    return b
    return bookmakers[0]


def _extract_market_outcomes(book: dict, market_key: str) -> list:
    for m in (book or {}).get("markets", []) or []:
        if (m or {}).get("key") == market_key:
            return (m or {}).get("outcomes", []) or []
    return []


def _extract_h2h(book: dict, home_name: str, away_name: str) -> Tuple[float, float]:
    outs = _extract_market_outcomes(book, "h2h")
    home_ml = float("nan")
    away_ml = float("nan")
    for o in outs:
        nm = (o or {}).get("name")
        price = _safe_float((o or {}).get("price"))
        if nm == home_name:
            home_ml = price
        elif nm == away_name:
            away_ml = price
    return home_ml, away_ml


def _extract_spreads(book: dict, home_name: str, away_name: str) -> Tuple[float, float]:
    """
    Returns (home_spread_points, spread_price_american_for_that_side)
    If unavailable, returns (nan, nan)
    """
    outs = _extract_market_outcomes(book, "spreads")
    home_spread = float("nan")
    home_spread_price = float("nan")

    # Odds API spreads outcomes look like:
    # { "name": "Team A", "point": -3.5, "price": -110 }
    for o in outs:
        nm = (o or {}).get("name")
        if nm != home_name:
            continue
        home_spread = _safe_float((o or {}).get("point"))
        home_spread_price = _safe_float((o or {}).get("price"))
        break

    return home_spread, home_spread_price


def _extract_totals(book: dict) -> Tuple[float, float, float]:
    """
    Returns (total_points, over_price, under_price)
    """
    outs = _extract_market_outcomes(book, "totals")
    total_points = float("nan")
    over_price = float("nan")
    under_price = float("nan")

    for o in outs:
        nm = (o or {}).get("name")
        pt = _safe_float((o or {}).get("point"))
        pr = _safe_float((o or {}).get("price"))
        if not math.isnan(pt):
            total_points = pt
        if str(nm).lower() == "over":
            over_price = pr
        elif str(nm).lower() == "under":
            under_price = pr

    return total_points, over_price, under_price


def fetch_odds_for_date_from_odds_api(
    game_date_str: str,
    *,
    sport_key: str,
    days_padding: int = 1,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], float]]:
    """
    Fetch odds for a date window around game_date_str (MM/DD/YYYY).

    Returns:
      odds_dict: {(home, away): {
          home_ml, away_ml,
          home_spread, spread_price,
          total_points, over_price, under_price,
          commence_time
      }}
      spreads_dict: {(home, away): home_spread}
    """
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ODDS_API_KEY missing")

    try:
        d = datetime.strptime(game_date_str, "%m/%d/%Y")
    except Exception as e:
        raise ValueError(f"bad game_date_str {game_date_str}: {e}")

    # Window in UTC
    start = (d - timedelta(days=int(days_padding))).strftime("%Y-%m-%dT00:00:00Z")
    end = (d + timedelta(days=int(days_padding))).strftime("%Y-%m-%dT23:59:59Z")

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": start,
        "commenceTimeTo": end,
    }

    r = requests.get(url, params=params, timeout=25)
    print(f"[odds_api DEBUG] status: {r.status_code}")
    print(f"[odds_api DEBUG] url: {r.url}")

    r.raise_for_status()
    events = r.json() or []

    odds_dict: Dict[Tuple[str, str], dict] = {}
    spreads_dict: Dict[Tuple[str, str], float] = {}

    for ev in events:
        home = (ev or {}).get("home_team")
        away = (ev or {}).get("away_team")
        if not home or not away:
            continue

        book = _pick_best_bookmaker((ev or {}).get("bookmakers") or [])
        if not book:
            continue

        home_ml, away_ml = _extract_h2h(book, home, away)
        home_spread, spread_price = _extract_spreads(book, home, away)
        total_points, over_price, under_price = _extract_totals(book)

        oi = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
            "commence_time": (ev or {}).get("commence_time"),
        }

        odds_dict[(home, away)] = oi
        spreads_dict[(home, away)] = home_spread

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(game_date_str: str, *, sport: str) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], float]]:
    """
    CSV fallback.
    Expected file:
      odds/odds_<MM-DD-YYYY>.csv

    Expected columns (minimum):
      date,home,away,home_ml,away_ml,home_spread

    Optional columns:
      spread_price,total_points,over_price,under_price,commence_time
    """
    fname = f"odds/odds_{game_date_str.replace('/','-')}.csv"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"CSV not found: {fname}")

    odds_dict: Dict[Tuple[str, str], dict] = {}
    spreads_dict: Dict[Tuple[str, str], float] = {}

    with open(fname, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = (row or {}).get("home")
            away = (row or {}).get("away")
            if not home or not away:
                continue

            oi = {
                "home_ml": _safe_float((row or {}).get("home_ml")),
                "away_ml": _safe_float((row or {}).get("away_ml")),
                "home_spread": _safe_float((row or {}).get("home_spread")),
                "spread_price": _safe_float((row or {}).get("spread_price"), default=float(os.getenv("DEFAULT_SPREAD_PRICE", "-110"))),
                "total_points": _safe_float((row or {}).get("total_points")),
                "over_price": _safe_float((row or {}).get("over_price"), default=float(os.getenv("DEFAULT_TOTAL_PRICE", "-110"))),
                "under_price": _safe_float((row or {}).get("under_price"), default=float(os.getenv("DEFAULT_TOTAL_PRICE", "-110"))),
                "commence_time": (row or {}).get("commence_time"),
            }

            odds_dict[(home, away)] = oi
            spreads_dict[(home, away)] = oi["home_spread"]

    return odds_dict, spreads_dict
