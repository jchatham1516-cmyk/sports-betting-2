# sports/common/odds_sources.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional

import pandas as pd
import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Canon sport keys used by the Odds API
SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

DEFAULT_REGIONS = os.getenv("ODDS_REGIONS", "us")
DEFAULT_MARKETS = os.getenv("ODDS_MARKETS", "h2h,spreads,totals")
DEFAULT_ODDS_FORMAT = os.getenv("ODDS_FORMAT", "american")
DEFAULT_DATE_FORMAT = os.getenv("ODDS_DATE_FORMAT", "iso")


def _get_odds_api_key() -> str:
    k = os.getenv("ODDS_API_KEY", "") or os.getenv("THE_ODDS_API_KEY", "")
    if not k:
        raise RuntimeError("Missing ODDS_API_KEY in environment.")
    return k


def _safe_float(x, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _parse_mmddyyyy(game_date: str) -> datetime:
    return datetime.strptime(str(game_date), "%m/%d/%Y")


def _iso(dt: datetime) -> str:
    # always treat as UTC-ish; Odds API accepts ISO without timezone too,
    # but we’ll include Z style by leaving it naive and letting requests encode
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick_best_bookmaker(bookmakers: list[dict]) -> Optional[dict]:
    """
    Choose a bookmaker object to read lines from.
    Preference: the one with most markets present.
    """
    if not bookmakers:
        return None

    best = None
    best_score = -1
    for bk in bookmakers:
        mkts = bk.get("markets") or []
        mset = {m.get("key") for m in mkts if m.get("key")}
        score = 0
        if "h2h" in mset:
            score += 3
        if "spreads" in mset:
            score += 2
        if "totals" in mset:
            score += 2
        # tie-break: number of outcomes
        if score > best_score:
            best = bk
            best_score = score
    return best


def _extract_market(bk: dict, key: str) -> Optional[dict]:
    for m in (bk.get("markets") or []):
        if m.get("key") == key:
            return m
    return None


def _extract_h2h(bk: dict, home: str, away: str) -> Tuple[float, float]:
    """
    Returns (home_ml, away_ml) in American odds.
    """
    h2h = _extract_market(bk, "h2h")
    if not h2h:
        return (float("nan"), float("nan"))

    home_ml = float("nan")
    away_ml = float("nan")
    for o in (h2h.get("outcomes") or []):
        nm = o.get("name")
        price = _safe_float(o.get("price"))
        if nm == home:
            home_ml = price
        elif nm == away:
            away_ml = price
    return (home_ml, away_ml)


def _extract_spreads(bk: dict, home: str, away: str) -> Tuple[float, float]:
    """
    Returns (home_spread, spread_price) where spread_price is the American price
    for the home spread line (if available). If price differs per side, we still
    grab the home side’s price.
    """
    sp = _extract_market(bk, "spreads")
    if not sp:
        return (float("nan"), float("nan"))

    home_spread = float("nan")
    home_price = float("nan")
    for o in (sp.get("outcomes") or []):
        nm = o.get("name")
        point = _safe_float(o.get("point"))
        price = _safe_float(o.get("price"))
        if nm == home:
            home_spread = point
            home_price = price
    return (home_spread, home_price)


def _extract_totals(bk: dict) -> Tuple[float, float, float]:
    """
    Returns (total_points, over_price, under_price).
    """
    tot = _extract_market(bk, "totals")
    if not tot:
        return (float("nan"), float("nan"), float("nan"))

    total_points = float("nan")
    over_price = float("nan")
    under_price = float("nan")

    for o in (tot.get("outcomes") or []):
        nm = (o.get("name") or "").lower()
        point = _safe_float(o.get("point"))
        price = _safe_float(o.get("price"))
        # "Over"/"Under"
        if "over" in nm:
            total_points = point
            over_price = price
        elif "under" in nm:
            total_points = point
            under_price = price

    return (total_points, over_price, under_price)


def fetch_odds_for_date_from_odds_api(
    game_date: str,
    *,
    sport_key: str,
    days_padding: int = 1,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
    odds_format: str = DEFAULT_ODDS_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], dict]]:
    """
    Returns:
      odds_dict[(home, away)] = {
        home_ml, away_ml, home_spread, spread_price,
        total_points, over_price, under_price,
        commence_time, bookmakers(optional)
      }

      spreads_dict[(home, away)] = {"home_spread":..., "spread_price":...}
    """
    api_key = _get_odds_api_key()

    d0 = _parse_mmddyyyy(game_date)
    start = d0 - timedelta(days=int(days_padding))
    end = d0 + timedelta(days=int(days_padding), hours=23, minutes=59, seconds=59)

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": _iso(start),
        "commenceTimeTo": _iso(end),
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:200]}")

    events = r.json() or []
    odds_dict: Dict[Tuple[str, str], dict] = {}
    spreads_dict: Dict[Tuple[str, str], dict] = {}

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence_time = ev.get("commence_time")
        if not home or not away:
            continue

        bk = _pick_best_bookmaker(ev.get("bookmakers") or [])
        if not bk:
            # still record teams/commence_time so downstream can see game exists
            odds_dict[(home, away)] = {
                "home_ml": float("nan"),
                "away_ml": float("nan"),
                "home_spread": float("nan"),
                "spread_price": float("nan"),
                "total_points": float("nan"),
                "over_price": float("nan"),
                "under_price": float("nan"),
                "commence_time": commence_time,
            }
            continue

        home_ml, away_ml = _extract_h2h(bk, home, away)
        home_spread, spread_price = _extract_spreads(bk, home, away)
        total_points, over_price, under_price = _extract_totals(bk)

        payload = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
            "commence_time": commence_time,
        }

        odds_dict[(home, away)] = payload
        spreads_dict[(home, away)] = {"home_spread": home_spread, "spread_price": spread_price}

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(game_date: str, *, sport: str) -> Tuple[Dict[Tuple[str, str], dict], Dict[Tuple[str, str], dict]]:
    """
    CSV fallback. Expects:
      odds/odds_MM-DD-YYYY.csv
    with at least:
      date,home,away,home_ml,away_ml,home_spread
    Optional columns:
      spread_price,total_points,over_price,under_price,commence_time
    """
    mmddyyyy = _parse_mmddyyyy(game_date)
    fname = f"odds/odds_{mmddyyyy.strftime('%m-%d-%Y')}.csv"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"CSV odds file not found: {fname}")

    df = pd.read_csv(fname)
    if df is None or df.empty:
        return {}, {}

    odds_dict: Dict[Tuple[str, str], dict] = {}
    spreads_dict: Dict[Tuple[str, str], dict] = {}

    for _, r in df.iterrows():
        home = str(r.get("home") or "").strip()
        away = str(r.get("away") or "").strip()
        if not home or not away:
            continue

        payload = {
            "home_ml": _safe_float(r.get("home_ml")),
            "away_ml": _safe_float(r.get("away_ml")),
            "home_spread": _safe_float(r.get("home_spread")),
            "spread_price": _safe_float(r.get("spread_price")),
            "total_points": _safe_float(r.get("total_points")),
            "over_price": _safe_float(r.get("over_price")),
            "under_price": _safe_float(r.get("under_price")),
            "commence_time": str(r.get("commence_time") or ""),
        }
        odds_dict[(home, away)] = payload
        spreads_dict[(home, away)] = {"home_spread": payload["home_spread"], "spread_price": payload["spread_price"]}

    return odds_dict, spreads_dict
