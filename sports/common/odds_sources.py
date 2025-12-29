# sports/common/odds_sources.py
from __future__ import annotations

import os
import math
import glob
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Used by your runner + models
SPORT_TO_ODDS_KEY: Dict[str, str] = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

# ----------------------------
# Helpers
# ----------------------------
def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _median_or_nan(vals: List[float]) -> float:
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return float("nan")
    return float(np.median(np.array(vals, dtype=float)))


def _pick_price_closest_to_minus110(prices: List[float]) -> float:
    """
    Pick a price that's closest to -110 (typical spread/total juice).
    If empty, return -110.
    """
    prices = [float(p) for p in prices if p is not None and not (isinstance(p, float) and np.isnan(p))]
    if not prices:
        return -110.0
    target = -110.0
    return float(sorted(prices, key=lambda p: abs(p - target))[0])


def _ensure_mmddyyyy(game_date_str: str) -> str:
    # Accept MM/DD/YYYY and also YYYY-MM-DD
    s = str(game_date_str).strip()
    if "/" in s:
        return s
    if "-" in s:
        # try YYYY-MM-DD
        try:
            dt = datetime.strptime(s, "%Y-%m-%d")
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return s
    return s


def _date_to_window_utc(game_date_str: str, days_padding: int) -> Tuple[str, str]:
    """
    Create commenceTimeFrom/To in UTC ISO for Odds API.
    Padding expands the query window in days.
    """
    d = datetime.strptime(_ensure_mmddyyyy(game_date_str), "%m/%d/%Y")
    start = (d - timedelta(days=int(days_padding))).replace(hour=0, minute=0, second=0, microsecond=0)
    end = (d + timedelta(days=int(days_padding))).replace(hour=23, minute=59, second=59, microsecond=0)
    return (start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ"))


def _odds_api_get(path: str, params: dict) -> Any:
    url = f"{ODDS_API_BASE_URL}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=30)
    # print basic debug if caller wants to see it
    if os.getenv("ODDS_API_DEBUG", "1") == "1":
        try:
            print(f"[odds_api DEBUG] status: {r.status_code}")
            print(f"[odds_api DEBUG] url: {r.url}")
            print(f"[odds_api DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
            print(f"[odds_api DEBUG] used: {r.headers.get('x-requests-used')}")
            print(f"[odds_api DEBUG] last: {r.headers.get('x-requests-last')}")
        except Exception:
            pass
    r.raise_for_status()
    return r.json()


def _extract_market_lines(bookmakers: list, market_key: str, home: str, away: str) -> dict:
    """
    Collect all lines across bookmakers for a given market.
    Returns dict with aggregated info.
    """
    out = {
        "home_ml": [],
        "away_ml": [],
        "home_spread": [],
        "spread_price": [],
        "total_points": [],
        "over_price": [],
        "under_price": [],
    }

    for bk in bookmakers or []:
        for mk in (bk.get("markets") or []):
            if mk.get("key") != market_key:
                continue
            outs = mk.get("outcomes") or []

            if market_key == "h2h":
                for o in outs:
                    name = str(o.get("name") or "")
                    price = _safe_float(o.get("price"))
                    if name == home:
                        out["home_ml"].append(price)
                    elif name == away:
                        out["away_ml"].append(price)

            elif market_key == "spreads":
                # outcomes have name, point, price
                for o in outs:
                    name = str(o.get("name") or "")
                    point = _safe_float(o.get("point"))
                    price = _safe_float(o.get("price"))
                    if name == home:
                        out["home_spread"].append(point)
                        out["spread_price"].append(price)

            elif market_key == "totals":
                for o in outs:
                    name = str(o.get("name") or "")
                    point = _safe_float(o.get("point"))
                    price = _safe_float(o.get("price"))
                    if name.lower() == "over":
                        out["total_points"].append(point)
                        out["over_price"].append(price)
                    elif name.lower() == "under":
                        out["total_points"].append(point)
                        out["under_price"].append(price)

    return out


# ----------------------------
# Public API
# ----------------------------
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
    Returns:
      odds_dict: {(home, away): {home_ml, away_ml, home_spread, spread_price, total_points, over_price, under_price, commence_time}}
      spreads_dict: legacy mapping (home, away) -> home_spread (kept for compatibility)
    """
    api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY env var")

    commence_from, commence_to = _date_to_window_utc(game_date_str, days_padding)

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    }

    events = _odds_api_get(f"sports/{sport_key}/odds", params=params) or []
    odds_dict: Dict[Tuple[str, str], dict] = {}
    spreads_dict: Dict[Tuple[str, str], float] = {}

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        commence_time = ev.get("commence_time")
        bookmakers = ev.get("bookmakers") or []

        h2h = _extract_market_lines(bookmakers, "h2h", home, away)
        spr = _extract_market_lines(bookmakers, "spreads", home, away)
        tot = _extract_market_lines(bookmakers, "totals", home, away)

        home_ml = _median_or_nan(h2h["home_ml"])
        away_ml = _median_or_nan(h2h["away_ml"])

        home_spread = _median_or_nan(spr["home_spread"])
        spread_price = _pick_price_closest_to_minus110(spr["spread_price"])

        total_points = _median_or_nan(tot["total_points"])
        over_price = _pick_price_closest_to_minus110(tot["over_price"])
        under_price = _pick_price_closest_to_minus110(tot["under_price"])

        odds_dict[(str(home), str(away))] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
            "commence_time": commence_time,
        }

        if not np.isnan(home_spread):
            spreads_dict[(str(home), str(away))] = float(home_spread)

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(game_date_str: str, *, sport: str = "nba") -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], float]]:
    """
    CSV fallback loader.

    Supported filenames:
      odds/odds_MM-DD-YYYY.csv
      odds/odds_<sport>_MM-DD-YYYY.csv
      odds/odds_MM-DD-YY.csv (best-effort)
    Required columns:
      date, home, away, home_ml, away_ml, home_spread
    Optional columns:
      spread_price, total_points, over_price, under_price, commence_time
    """
    d = datetime.strptime(_ensure_mmddyyyy(game_date_str), "%m/%d/%Y")
    mmddyyyy = d.strftime("%m-%d-%Y")

    candidates = [
        f"odds/odds_{mmddyyyy}.csv",
        f"odds/odds_{sport}_{mmddyyyy}.csv",
        f"odds/odds_{mmddyyyy.replace('-0', '-')}.csv",
        f"odds/odds_{sport}_{mmddyyyy.replace('-0', '-')}.csv",
    ]

    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        # last resort: any odds file with the date inside the name
        hits = glob.glob(f"odds/*{mmddyyyy}*.csv")
        if hits:
            path = hits[0]

    if not path:
        raise FileNotFoundError(f"No odds CSV found for {game_date_str} (looked for {candidates})")

    df = pd.read_csv(path)
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    needed = {"home", "away", "home_ml", "away_ml", "home_spread"}
    missing = [c for c in needed if c not in set(df.columns)]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing} in {path}")

    odds_dict: Dict[Tuple[str, str], dict] = {}
    spreads_dict: Dict[Tuple[str, str], float] = {}

    for _, r in df.iterrows():
        home = str(r.get("home") or "").strip()
        away = str(r.get("away") or "").strip()
        if not home or not away:
            continue

        home_ml = _safe_float(r.get("home_ml"))
        away_ml = _safe_float(r.get("away_ml"))
        home_spread = _safe_float(r.get("home_spread"))
        spread_price = _safe_float(r.get("spread_price"), default=-110.0)

        total_points = _safe_float(r.get("total_points"))
        over_price = _safe_float(r.get("over_price"), default=-110.0)
        under_price = _safe_float(r.get("under_price"), default=-110.0)

        odds_dict[(home, away)] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
            "commence_time": r.get("commence_time", None),
        }
        if not np.isnan(home_spread):
            spreads_dict[(home, away)] = float(home_spread)

    return odds_dict, spreads_dict
