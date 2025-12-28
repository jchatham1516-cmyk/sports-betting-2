# sports/common/odds_sources.py
from __future__ import annotations

import os
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Any, Optional

import pandas as pd
import numpy as np
import requests

# Canon map from our internal sport names -> The Odds API keys
SPORT_TO_ODDS_KEY: Dict[str, str] = {
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


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _best_market(bookmakers: list, market_key: str) -> Optional[dict]:
    """
    Pick the first available market from the first bookmaker that has it.
    (Simple and robust; you can upgrade later to "best price" selection.)
    """
    for b in bookmakers or []:
        for m in (b.get("markets") or []):
            if m.get("key") == market_key:
                return m
    return None


def _extract_h2h(oi: dict, home_name: str, away_name: str) -> Tuple[float, float]:
    """
    Returns (home_ml, away_ml)
    """
    home_ml = np.nan
    away_ml = np.nan
    m = _best_market(oi.get("bookmakers") or [], "h2h")
    if not m:
        return (home_ml, away_ml)

    for out in m.get("outcomes") or []:
        n = str(out.get("name") or "")
        p = _safe_float(out.get("price"))
        if not np.isnan(p):
            if n == home_name:
                home_ml = p
            elif n == away_name:
                away_ml = p
    return (home_ml, away_ml)


def _extract_spread(oi: dict, home_name: str, away_name: str) -> Tuple[float, float]:
    """
    Returns (home_spread, spread_price)
    Where:
      - home_spread is the line for the HOME team (e.g. -3.5 means home favored)
      - spread_price is the price (American) for betting that HOME spread
    """
    home_spread = np.nan
    spread_price = np.nan

    m = _best_market(oi.get("bookmakers") or [], "spreads")
    if not m:
        return (home_spread, spread_price)

    for out in m.get("outcomes") or []:
        n = str(out.get("name") or "")
        pt = _safe_float(out.get("point"))
        pr = _safe_float(out.get("price"))
        if n == home_name:
            home_spread = pt
            spread_price = pr
            break

    return (home_spread, spread_price)


def _extract_totals(oi: dict) -> Tuple[float, float, float]:
    """
    Returns (total_points, over_price, under_price)
    """
    total_points = np.nan
    over_price = np.nan
    under_price = np.nan

    m = _best_market(oi.get("bookmakers") or [], "totals")
    if not m:
        return (total_points, over_price, under_price)

    for out in m.get("outcomes") or []:
        nm = str(out.get("name") or "").lower()
        pt = _safe_float(out.get("point"))
        pr = _safe_float(out.get("price"))
        if not np.isnan(pt) and np.isnan(total_points):
            total_points = pt
        if "over" in nm:
            over_price = pr
        elif "under" in nm:
            under_price = pr

    return (total_points, over_price, under_price)


def fetch_odds_for_date_from_odds_api(
    sport: str,
    game_date_str: str,
    *,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
) -> Dict[Tuple[str, str], dict]:
    """
    Pull odds for a target date using The Odds API.

    Returns:
      dict keyed by (home_team, away_team) with values:
        {
          "home_ml": ...,
          "away_ml": ...,
          "home_spread": ...,
          "spread_price": ...,
          "total_points": ...,
          "over_price": ...,
          "under_price": ...,
          "commence_time": ...,
        }
    """
    api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY / THE_ODDS_API_KEY in environment")

    sport_key = SPORT_TO_ODDS_KEY.get(sport)
    if not sport_key:
        raise ValueError(f"Unknown sport '{sport}'. Expected one of: {sorted(SPORT_TO_ODDS_KEY.keys())}")

    # We query a window around the date to catch late-night / timezone issues.
    # game_date_str in your runner is mm/dd/YYYY.
    try:
        target = datetime.strptime(game_date_str, "%m/%d/%Y").replace(tzinfo=timezone.utc)
    except Exception:
        # fallback: allow YYYY-MM-DD too
        target = datetime.strptime(game_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    commence_from = (target - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    commence_to = (target + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")

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
    r.raise_for_status()
    data = r.json() or []

    odds_dict: Dict[Tuple[str, str], dict] = {}

    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        home_ml, away_ml = _extract_h2h(ev, home, away)
        home_spread, spread_price = _extract_spread(ev, home, away)
        total_points, over_price, under_price = _extract_totals(ev)

        odds_dict[(home, away)] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
            "commence_time": ev.get("commence_time"),
            "id": ev.get("id"),
        }

    return odds_dict


def fetch_odds_for_date_from_csv(game_date_str: str, *, folder: str = "odds") -> Dict[Tuple[str, str], dict]:
    """
    Load odds from local CSV.
    Expected columns:
      date,home,away,home_ml,away_ml,home_spread
    Optional columns:
      spread_price,total_points,over_price,under_price,commence_time
    """
    # supports either odds_MM-DD-YYYY.csv or odds_12-16-2025.csv patterns you used before
    mmddyyyy = None
    try:
        d = datetime.strptime(game_date_str, "%m/%d/%Y")
        mmddyyyy = d.strftime("%m-%d-%Y")
    except Exception:
        try:
            d = datetime.strptime(game_date_str, "%Y-%m-%d")
            mmddyyyy = d.strftime("%m-%d-%Y")
        except Exception:
            pass

    candidates = []
    if mmddyyyy:
        candidates.append(os.path.join(folder, f"odds_{mmddyyyy}.csv"))
    candidates.append(os.path.join(folder, f"odds_{game_date_str}.csv"))

    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if not path:
        raise FileNotFoundError(f"No odds CSV found for date={game_date_str}. Looked for: {candidates}")

    df = pd.read_csv(path)
    # normalize
    df.columns = [str(c).strip() for c in df.columns]

    required = {"home", "away"}
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Odds CSV missing required column '{c}' in {path}")

    out: Dict[Tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        home = str(row.get("home") or "").strip()
        away = str(row.get("away") or "").strip()
        if not home or not away:
            continue

        out[(home, away)] = {
            "home_ml": _safe_float(row.get("home_ml")),
            "away_ml": _safe_float(row.get("away_ml")),
            "home_spread": _safe_float(row.get("home_spread")),
            "spread_price": _safe_float(row.get("spread_price"), default=np.nan),
            "total_points": _safe_float(row.get("total_points"), default=np.nan),
            "over_price": _safe_float(row.get("over_price"), default=np.nan),
            "under_price": _safe_float(row.get("under_price"), default=np.nan),
            "commence_time": row.get("commence_time"),
        }

    return out


def load_odds_for_date(
    sport: str,
    game_date_str: str,
    *,
    prefer_api: bool = True,
) -> Dict[Tuple[str, str], dict]:
    """
    Convenience wrapper: try API first, fall back to CSV.
    """
    if prefer_api:
        try:
            return fetch_odds_for_date_from_odds_api(sport=sport, game_date_str=game_date_str)
        except Exception as e:
            print(f"[odds_sources] WARNING: API odds failed, falling back to CSV: {e}")
    return fetch_odds_for_date_from_csv(game_date_str)
