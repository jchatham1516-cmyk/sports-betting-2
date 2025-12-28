# sports/common/odds_sources.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, date
from typing import Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Map your internal sport names to Odds API sport keys
SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _to_mmddyyyy(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def _parse_date_any(s: str) -> Optional[date]:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def _canon_key(home: str, away: str) -> Tuple[str, str]:
    # upstream models call canon_team themselves; keep odds_sources “dumb” here
    return (str(home).strip(), str(away).strip())


def _pick_best_bookmaker(ev: dict) -> Optional[dict]:
    """
    Pick one bookmaker. Preference order:
    - Use the first bookmaker in list (Odds API is typically sorted by availability),
      OR allow env override.
    """
    bms = ev.get("bookmakers") or []
    if not bms:
        return None

    preferred = os.getenv("ODDS_API_BOOK", "").strip().lower()
    if preferred:
        for bm in bms:
            if str(bm.get("key", "")).lower() == preferred or str(bm.get("title", "")).lower() == preferred:
                return bm
    return bms[0]


def _extract_market_prices(bm: dict) -> Dict[str, Any]:
    """
    From a single bookmaker object, pull:
      - h2h home_ml / away_ml
      - spreads home_spread + spread_price
      - totals total_points + over_price + under_price
    """
    out = {
        "home_ml": np.nan,
        "away_ml": np.nan,
        "home_spread": np.nan,
        "spread_price": np.nan,
        "total_points": np.nan,
        "over_price": np.nan,
        "under_price": np.nan,
    }

    markets = bm.get("markets") or []
    for m in markets:
        key = m.get("key")
        outcomes = m.get("outcomes") or []

        if key == "h2h":
            # outcomes like [{"name":"Team A","price":-120},{"name":"Team B","price":+100}]
            # We'll return raw prices; the model will no-vig it.
            # We'll map later using event home/away.
            out["_h2h_outcomes"] = outcomes

        elif key == "spreads":
            # outcomes like:
            # [{"name":"Home Team","point":-1.5,"price":-110}, {"name":"Away Team","point":+1.5,"price":-110}]
            out["_spreads_outcomes"] = outcomes

        elif key == "totals":
            # outcomes like:
            # [{"name":"Over","point":47.5,"price":-110}, {"name":"Under","point":47.5,"price":-110}]
            out["_totals_outcomes"] = outcomes

    return out


def _apply_event_team_mapping(ev: dict, extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use ev["home_team"]/ev["away_team"] to map outcomes.
    """
    home = ev.get("home_team")
    away = ev.get("away_team")

    # Moneyline
    for o in extracted.get("_h2h_outcomes", []) or []:
        nm = o.get("name")
        price = _safe_float(o.get("price"))
        if nm == home:
            extracted["home_ml"] = price
        elif nm == away:
            extracted["away_ml"] = price

    # Spreads
    # We want: home_spread = the "point" for the home team, and spread_price = its price (fallback -110 elsewhere).
    for o in extracted.get("_spreads_outcomes", []) or []:
        nm = o.get("name")
        if nm == home:
            extracted["home_spread"] = _safe_float(o.get("point"))
            extracted["spread_price"] = _safe_float(o.get("price"))

    # Totals
    # total_points = the market point; over/under prices accordingly
    for o in extracted.get("_totals_outcomes", []) or []:
        nm = str(o.get("name", "")).lower()
        pt = _safe_float(o.get("point"))
        price = _safe_float(o.get("price"))
        if not np.isnan(pt):
            extracted["total_points"] = pt
        if "over" in nm:
            extracted["over_price"] = price
        elif "under" in nm:
            extracted["under_price"] = price

    # keep useful event metadata too
    extracted["commence_time"] = ev.get("commence_time")
    return extracted


def _odds_api_get(url: str, params: dict, *, max_retries: int = 5, sleep_base: float = 2.0) -> Any:
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if os.getenv("ODDS_API_DEBUG", "1") == "1":
                print(f"[odds_api DEBUG] status: {r.status_code}")
                print(f"[odds_api DEBUG] url: {r.url}")
                try:
                    print(f"[odds_api DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
                    print(f"[odds_api DEBUG] used: {r.headers.get('x-requests-used')}")
                    print(f"[odds_api DEBUG] last: {r.headers.get('x-requests-last')}")
                except Exception:
                    pass
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (1.5 ** i))
    raise RuntimeError(f"Odds API request failed after {max_retries} attempts: {last_err}")


# -----------------------------
# CSV Odds Loader
# -----------------------------
def fetch_odds_for_date_from_csv(csv_path: str) -> Dict[Tuple[str, str], dict]:
    """
    CSV format expected (you used this before):
      date,home,away,home_ml,away_ml,home_spread
    Optional columns also supported:
      spread_price,total_points,over_price,under_price,commence_time
    """
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Odds CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out: Dict[Tuple[str, str], dict] = {}

    for _, row in df.iterrows():
        home = str(row.get("home", "")).strip()
        away = str(row.get("away", "")).strip()
        if not home or not away:
            continue

        k = _canon_key(home, away)
        out[k] = {
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


# -----------------------------
# Odds API Loader (THIS fixes your ImportError)
# -----------------------------
def fetch_odds_for_date_from_odds_api(
    *,
    sport: str,
    game_date: str,
    api_key: Optional[str] = None,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
) -> Dict[Tuple[str, str], dict]:
    """
    Returns odds_dict keyed by (home_team, away_team):
      {
        ("Team A","Team B"): {
            home_ml, away_ml, home_spread, spread_price,
            total_points, over_price, under_price,
            commence_time
        },
        ...
      }
    """
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY missing")

    odds_key = SPORT_TO_ODDS_KEY.get(str(sport).lower())
    if not odds_key:
        raise ValueError(f"Unknown sport '{sport}'. Expected one of: {sorted(SPORT_TO_ODDS_KEY.keys())}")

    d = _parse_date_any(game_date)
    if d is None:
        raise ValueError(f"Could not parse game_date '{game_date}'. Use MM/DD/YYYY or YYYY-MM-DD.")

    # Build a 3-day window around date (matches your logs pattern)
    commence_from = datetime(d.year, d.month, d.day) - timedelta(days=1)
    commence_to = datetime(d.year, d.month, d.day) + timedelta(days=1, hours=23, minutes=59, seconds=59)

    url = f"{ODDS_API_BASE_URL}/sports/{odds_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": commence_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": commence_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    events = _odds_api_get(url, params=params) or []
    out: Dict[Tuple[str, str], dict] = {}

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        bm = _pick_best_bookmaker(ev)
        if not bm:
            continue

        extracted = _extract_market_prices(bm)
        extracted = _apply_event_team_mapping(ev, extracted)

        # Remove internal temp keys
        extracted.pop("_h2h_outcomes", None)
        extracted.pop("_spreads_outcomes", None)
        extracted.pop("_totals_outcomes", None)

        out[_canon_key(home, away)] = extracted

    return out
