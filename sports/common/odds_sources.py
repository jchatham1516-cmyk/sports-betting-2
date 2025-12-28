# sports/common/odds_sources.py
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Your repo uses this mapping to choose odds-api sport keys.
# Keep it consistent with the rest of your codebase.
SPORT_TO_ODDS_KEY = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _odds_api_key() -> str:
    key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or ""
    return str(key).strip()


def _get(url: str, params: dict, *, timeout: int = 30, debug: bool = False) -> Any:
    r = requests.get(url, params=params, timeout=timeout)
    if debug:
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


def _extract_h2h(ev: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (home_ml, away_ml) in American odds.
    """
    try:
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bm in ev.get("bookmakers", []) or []:
            for mkt in bm.get("markets", []) or []:
                if mkt.get("key") != "h2h":
                    continue
                outcomes = mkt.get("outcomes") or []
                hm = None
                am = None
                for o in outcomes:
                    name = o.get("name")
                    price = _safe_float(o.get("price"))
                    if name == home:
                        hm = price
                    elif name == away:
                        am = price
                if hm is not None or am is not None:
                    return hm, am
    except Exception:
        pass
    return None, None


def _extract_spreads(ev: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (home_spread, spread_price).
    home_spread = points for HOME team (e.g. -3.5 means home favored by 3.5)
    spread_price = best-available price for the chosen spread (if present).
    """
    try:
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bm in ev.get("bookmakers", []) or []:
            for mkt in bm.get("markets", []) or []:
                if mkt.get("key") != "spreads":
                    continue
                outcomes = mkt.get("outcomes") or []
                hs = None
                hp = None
                for o in outcomes:
                    name = o.get("name")
                    point = _safe_float(o.get("point"))
                    price = _safe_float(o.get("price"))
                    if name == home:
                        hs = point
                        hp = price
                    elif name == away:
                        # if only away is listed, infer home spread
                        if point is not None:
                            hs = -float(point)
                return hs, hp
    except Exception:
        pass
    return None, None


def _extract_totals(ev: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (total_points, over_price, under_price).
    total_points is the market total number.
    """
    try:
        for bm in ev.get("bookmakers", []) or []:
            for mkt in bm.get("markets", []) or []:
                if mkt.get("key") != "totals":
                    continue
                outcomes = mkt.get("outcomes") or []
                total_points = None
                over_price = None
                under_price = None
                for o in outcomes:
                    name = (o.get("name") or "").lower().strip()
                    point = _safe_float(o.get("point"))
                    price = _safe_float(o.get("price"))
                    if point is not None:
                        total_points = point
                    if "over" in name:
                        over_price = price
                    if "under" in name:
                        under_price = price
                return total_points, over_price, under_price
    except Exception:
        pass
    return None, None, None


# -----------------------------
# Core loaders
# -----------------------------
def load_odds_for_date_from_api(
    *,
    sport_key: str,
    commence_from: datetime,
    commence_to: datetime,
    debug: bool = True,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Loads h2h + spreads + totals for a time window.
    Returns:
      {(home, away): {... fields ...}}
    IMPORTANT: includes commence_time so models can filter by date correctly.
    """
    api_key = _odds_api_key()
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY (or THE_ODDS_API_KEY)")

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": commence_from.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "commenceTimeTo": commence_to.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    events = _get(url, params=params, debug=debug) or []
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for ev in events:
        home = (ev.get("home_team") or "").strip()
        away = (ev.get("away_team") or "").strip()
        if not home or not away:
            continue

        home_ml, away_ml = _extract_h2h(ev)
        home_spread, spread_price = _extract_spreads(ev)
        total_points, over_price, under_price = _extract_totals(ev)

        out[(home, away)] = {
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),  # ✅ critical for date filtering
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_points": total_points,
            "over_price": over_price,
            "under_price": under_price,
        }

    return out


def load_odds_for_date_from_csv(csv_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    CSV columns supported:
      date, home, away, home_ml, away_ml, home_spread
    Optional columns:
      spread_price, total_points, over_price, under_price, commence_time
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = (row.get("home") or "").strip()
            away = (row.get("away") or "").strip()
            if not home or not away:
                continue

            out[(home, away)] = {
                "id": row.get("id"),
                "commence_time": row.get("commence_time") or row.get("date"),
                "home_ml": _safe_float(row.get("home_ml")),
                "away_ml": _safe_float(row.get("away_ml")),
                "home_spread": _safe_float(row.get("home_spread")),
                "spread_price": _safe_float(row.get("spread_price")),
                "total_points": _safe_float(row.get("total_points")),
                "over_price": _safe_float(row.get("over_price")),
                "under_price": _safe_float(row.get("under_price")),
            }

    return out


# -------------------------------------------------------------------
# BACKWARDS-COMPAT: these are what your runner imports
# -------------------------------------------------------------------
def fetch_odds_for_date_from_odds_api(
    game_date_str: str,
    *,
    sport_key: str,
    days_padding: int = 1,
    debug: bool = True,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], float]]:
    """
    Returns:
      odds_dict: {(home,away): {home_ml,away_ml,home_spread,spread_price,total_points,over_price,under_price,commence_time}}
      spreads_dict: {(home,away): home_spread}
    """
    dt = datetime.strptime(game_date_str, "%m/%d/%Y")
    dt0 = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

    pad = int(days_padding or 0)
    commence_from = dt0 - timedelta(days=pad)
    commence_to = dt0 + timedelta(days=pad + 1) - timedelta(seconds=1)

    odds_dict = load_odds_for_date_from_api(
        sport_key=sport_key,
        commence_from=commence_from,
        commence_to=commence_to,
        debug=debug,
    )

    spreads_dict: Dict[Tuple[str, str], float] = {}
    for k, v in (odds_dict or {}).items():
        hs = v.get("home_spread")
        if hs is not None:
            try:
                spreads_dict[k] = float(hs)
            except Exception:
                pass

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(
    game_date_str: str,
    *,
    sport: str,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], float]]:
    """
    Looks for: odds/odds_MM-DD-YYYY.csv
    """
    dt = datetime.strptime(game_date_str, "%m/%d/%Y")
    fname = f"odds/odds_{dt.strftime('%m-%d-%Y')}.csv"
    odds_dict = load_odds_for_date_from_csv(fname)

    spreads_dict: Dict[Tuple[str, str], float] = {}
    for k, v in (odds_dict or {}).items():
        hs = v.get("home_spread")
        if hs is not None:
            try:
                spreads_dict[k] = float(hs)
            except Exception:
                pass

    return odds_dict, spreads_dict
