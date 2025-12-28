# sports/common/odds_sources.py
from __future__ import annotations

import os
import json
import time
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Tuple, Optional, Any

import requests

# ------------------------------------------------------------
# Sports keys (Odds API sport keys)
# ------------------------------------------------------------
SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _safe_float(x, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _utc_window_for_date(game_date: date) -> Tuple[str, str]:
    """
    Odds API expects UTC ISO strings.
    We use a forgiving window: from game_date-1 00:00Z to game_date+1 23:59Z.
    This prevents timezone/late games from dropping out.
    """
    start = datetime(game_date.year, game_date.month, game_date.day, tzinfo=timezone.utc) - timedelta(days=1)
    end = datetime(game_date.year, game_date.month, game_date.day, tzinfo=timezone.utc) + timedelta(days=1, hours=23, minutes=59, seconds=59)
    return (start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z"))


def _pick_bookmaker(bookmakers: list) -> Optional[dict]:
    """
    Prefer a common sharp-ish/consistent book if present, otherwise first.
    """
    if not bookmakers:
        return None
    preferred = {"draftkings", "fanduel", "betmgm", "pointsbetus", "caesars", "williamhill_us"}
    for b in bookmakers:
        try:
            if str(b.get("key", "")).lower() in preferred:
                return b
        except Exception:
            continue
    return bookmakers[0]


def _extract_markets(bookmaker: dict) -> Dict[str, dict]:
    out = {}
    for m in (bookmaker or {}).get("markets", []) or []:
        key = m.get("key")
        if key:
            out[str(key)] = m
    return out


def _extract_h2h(home: str, away: str, markets: Dict[str, dict]) -> Tuple[float, float]:
    home_ml = float("nan")
    away_ml = float("nan")
    m = markets.get("h2h")
    if not m:
        return home_ml, away_ml
    for o in m.get("outcomes", []) or []:
        name = o.get("name")
        price = o.get("price")
        if name == home:
            home_ml = _safe_float(price)
        elif name == away:
            away_ml = _safe_float(price)
    return home_ml, away_ml


def _extract_spreads(home: str, away: str, markets: Dict[str, dict]) -> Tuple[float, float]:
    """
    Returns:
      home_spread: points for HOME (e.g. -3.5 means home favored by 3.5)
      spread_price: price for the HOME spread (usually -110). If missing, NaN.
    """
    home_spread = float("nan")
    spread_price = float("nan")
    m = markets.get("spreads")
    if not m:
        return home_spread, spread_price

    for o in m.get("outcomes", []) or []:
        name = o.get("name")
        point = o.get("point")
        price = o.get("price")
        if name == home:
            home_spread = _safe_float(point)
            spread_price = _safe_float(price)
            break
    return home_spread, spread_price


def _extract_totals(markets: Dict[str, dict]) -> Tuple[float, float, float]:
    """
    Returns:
      total_points, over_price, under_price
    """
    total_points = float("nan")
    over_price = float("nan")
    under_price = float("nan")
    m = markets.get("totals")
    if not m:
        return total_points, over_price, under_price

    # totals outcomes often look like:
    # {"name":"Over","point":47.5,"price":-110}, {"name":"Under","point":47.5,"price":-110}
    for o in m.get("outcomes", []) or []:
        nm = str(o.get("name", "")).lower()
        pt = o.get("point")
        pr = o.get("price")
        if not (pt is None):
            total_points = _safe_float(pt, default=total_points)
        if nm == "over":
            over_price = _safe_float(pr)
        elif nm == "under":
            under_price = _safe_float(pr)
    return total_points, over_price, under_price


def _odds_api_get(url: str, params: dict, *, max_retries: int = 5) -> Any:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            # helpful debug
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
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 20)
            time.sleep(sleep_s)
    raise RuntimeError(f"Odds API request failed after retries: {last_err}")


# ------------------------------------------------------------
# Public API: Odds loaders
# ------------------------------------------------------------
def fetch_odds_for_date_from_odds_api(
    *,
    sport_key: str,
    game_date: date,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
) -> Dict[Tuple[str, str], dict]:
    """
    Returns odds_dict keyed by (home_team, away_team) with fields:
      home_ml, away_ml, home_spread, spread_price,
      total_points, over_price, under_price,
      commence_time
    """
    api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY (or THE_ODDS_API_KEY) in environment.")

    commence_from, commence_to = _utc_window_for_date(game_date)

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

    events = _odds_api_get(url, params=params) or []
    odds_dict: Dict[Tuple[str, str], dict] = {}

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        bm = _pick_bookmaker(ev.get("bookmakers") or [])
        mkts = _extract_markets(bm or {})
        home_ml, away_ml = _extract_h2h(home, away, mkts)
        home_spread, spread_price = _extract_spreads(home, away, mkts)
        total_points, over_price, under_price = _extract_totals(mkts)

        odds_dict[(str(home), str(away))] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
            "commence_time": ev.get("commence_time"),
            "bookmaker_key": (bm or {}).get("key"),
        }

    print(f"[odds_api] Loaded odds for {len(odds_dict)} games.")
    return odds_dict


def fetch_odds_for_date_from_csv(csv_path: str) -> Dict[Tuple[str, str], dict]:
    """
    Reads a CSV in your common format:
      date,home,away,home_ml,away_ml,home_spread
    Optional columns supported if present:
      spread_price,total_points,over_price,under_price,commence_time
    """
    import pandas as pd

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    out: Dict[Tuple[str, str], dict] = {}

    for _, r in df.iterrows():
        home = str(r.get("home", "")).strip()
        away = str(r.get("away", "")).strip()
        if not home or not away:
            continue

        out[(home, away)] = {
            "home_ml": _safe_float(r.get("home_ml")),
            "away_ml": _safe_float(r.get("away_ml")),
            "home_spread": _safe_float(r.get("home_spread")),
            "spread_price": _safe_float(r.get("spread_price")),
            "total_points": _safe_float(r.get("total_points")),
            "over_price": _safe_float(r.get("over_price")),
            "under_price": _safe_float(r.get("under_price")),
            "commence_time": r.get("commence_time"),
        }

    return out


def save_odds_cache(path: str, odds_dict: Dict[Tuple[str, str], dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {f"{k[0]}|||{k[1]}": v for k, v in (odds_dict or {}).items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_odds_cache(path: str) -> Dict[Tuple[str, str], dict]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        out = {}
        for k, v in payload.items():
            if "|||" in k:
                a, b = k.split("|||", 1)
                out[(a, b)] = v
        return out
    except Exception:
        return {}
