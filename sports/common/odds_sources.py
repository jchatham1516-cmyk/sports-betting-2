# sports/common/odds_sources.py
from __future__ import annotations

import os
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, Any

import pandas as pd
import requests

from sports.common.teams import canon_team

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Used by models
SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

# ---------------------------
# Helpers
# ---------------------------
def _safe_float(x, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _best_price_from_bookmakers(bookmakers, market_key: str):
    """
    We keep it simple: grab the first bookmaker that has the market.
    You can later improve this by selecting best price or consensus.
    """
    if not bookmakers:
        return None
    for b in bookmakers:
        mkts = b.get("markets") or []
        for m in mkts:
            if m.get("key") == market_key:
                return m
    return None


def _parse_h2h(event: dict) -> Tuple[float, float]:
    m = _best_price_from_bookmakers(event.get("bookmakers"), "h2h")
    if not m:
        return (float("nan"), float("nan"))
    outs = m.get("outcomes") or []
    # outcomes have {name, price}
    # We will map by matching event home_team/away_team names
    home_team = event.get("home_team")
    away_team = None
    for t in (event.get("away_team"),):
        away_team = t
    home_ml = float("nan")
    away_ml = float("nan")
    for o in outs:
        nm = o.get("name")
        pr = _safe_float(o.get("price"))
        if nm == home_team:
            home_ml = pr
        elif nm == away_team:
            away_ml = pr
    return home_ml, away_ml


def _parse_spreads(event: dict) -> Tuple[float, float]:
    """
    Returns: (home_spread, spread_price)
    spread_price is the price for the HOME spread outcome (american).
    """
    m = _best_price_from_bookmakers(event.get("bookmakers"), "spreads")
    if not m:
        return (float("nan"), float("nan"))
    outs = m.get("outcomes") or []
    home_team = event.get("home_team")
    home_spread = float("nan")
    home_price = float("nan")
    for o in outs:
        nm = o.get("name")
        pt = _safe_float(o.get("point"))
        pr = _safe_float(o.get("price"))
        if nm == home_team:
            home_spread = pt
            home_price = pr
            break
    return home_spread, home_price


def _parse_totals(event: dict) -> Tuple[float, float, float]:
    """
    Returns: (total_points, over_price, under_price)
    """
    m = _best_price_from_bookmakers(event.get("bookmakers"), "totals")
    if not m:
        return (float("nan"), float("nan"), float("nan"))
    outs = m.get("outcomes") or []
    total_points = float("nan")
    over_price = float("nan")
    under_price = float("nan")

    for o in outs:
        nm = str(o.get("name") or "").strip().lower()
        pt = _safe_float(o.get("point"))
        pr = _safe_float(o.get("price"))
        if not math.isnan(pt):
            total_points = pt
        if nm == "over":
            over_price = pr
        elif nm == "under":
            under_price = pr

    return total_points, over_price, under_price


def _event_key(home: str, away: str) -> Tuple[str, str]:
    return (canon_team(home), canon_team(away))


# ---------------------------
# Odds API loader
# ---------------------------
def fetch_odds_from_api(
    sport: str,
    *,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
    hours_back: int = 24,
    hours_forward: int = 72,
    sleep_s: float = 0.0,
) -> Dict[Tuple[str, str], dict]:
    """
    Returns dict keyed by (home, away) canonical team names:
      {
        ("Buffalo Bills","Miami Dolphins"): {
           "commence_time": "...",
           "home_ml": -120, "away_ml": 100,
           "home_spread": -2.5, "spread_price": -110,
           "total_points": 47.5, "over_price": -110, "under_price": -110
        }
      }
    """
    sport_key = SPORT_TO_ODDS_KEY.get(sport)
    if not sport_key:
        raise ValueError(f"Unknown sport: {sport}")

    api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY / THE_ODDS_API_KEY env var")

    t0 = _now_utc() - timedelta(hours=int(hours_back))
    t1 = _now_utc() + timedelta(hours=int(hours_forward))

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": _iso(t0),
        "commenceTimeTo": _iso(t1),
    }

    r = requests.get(url, params=params, timeout=30)
    print(f"[odds_api DEBUG] status: {r.status_code}")
    print(f"[odds_api DEBUG] url: {r.url}")
    try:
        print(f"[odds_api DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
        print(f"[odds_api DEBUG] used: {r.headers.get('x-requests-used')}")
        print(f"[odds_api DEBUG] last: {r.headers.get('x-requests-last')}")
    except Exception:
        pass
    r.raise_for_status()

    events = r.json() or []
    out: Dict[Tuple[str, str], dict] = {}

    for ev in events:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        if not home_raw or not away_raw:
            continue

        home = canon_team(home_raw)
        away = canon_team(away_raw)
        if not home or not away:
            continue

        home_ml, away_ml = _parse_h2h(ev)
        home_spread, spread_price = _parse_spreads(ev)
        total_points, over_price, under_price = _parse_totals(ev)

        out[(home, away)] = {
            "commence_time": ev.get("commence_time"),
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
        }

    if sleep_s and sleep_s > 0:
        time.sleep(float(sleep_s))
    return out


# ---------------------------
# CSV loader
# ---------------------------
def fetch_odds_for_date_from_csv(csv_path: str) -> Dict[Tuple[str, str], dict]:
    """
    Supports columns (case-insensitive):
      date, home, away, home_ml, away_ml, home_spread, spread_price,
      total_points, over_price, under_price
    """
    df = pd.read_csv(csv_path)
    # normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]

    def col(name: str) -> str:
        return name.lower()

    required = ["home", "away"]
    for c in required:
        if col(c) not in df.columns:
            raise ValueError(f"CSV missing column: {c}")

    out: Dict[Tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        home_raw = row.get("home")
        away_raw = row.get("away")
        if not home_raw or not away_raw:
            continue
        home = canon_team(home_raw)
        away = canon_team(away_raw)
        if not home or not away:
            continue

        out[(home, away)] = {
            "commence_time": row.get("commence_time") if "commence_time" in df.columns else None,
            "home_ml": _safe_float(row.get("home_ml")),
            "away_ml": _safe_float(row.get("away_ml")),
            "home_spread": _safe_float(row.get("home_spread")),
            "spread_price": _safe_float(row.get("spread_price")),
            "total_points": _safe_float(row.get("total_points")),
            "over_price": _safe_float(row.get("over_price")),
            "under_price": _safe_float(row.get("under_price")),
        }
    return out
