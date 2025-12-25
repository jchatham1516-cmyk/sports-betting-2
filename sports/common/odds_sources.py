# sports/common/odds_sources.py
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


SPORT_TO_ODDS_KEY: Dict[str, str] = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "nhl": "icehockey_nhl",
}

ODDS_API_HOST = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 20

ODDS_MAX_REQUESTS = int(os.getenv("ODDS_MAX_REQUESTS", "40"))
ODDS_MIN_REMAINING = int(os.getenv("ODDS_MIN_REMAINING", "10"))
ODDS_HARD_STOP_ON_401 = os.getenv("ODDS_HARD_STOP_ON_401", "1") == "1"


class _OddsBudget:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.count = 0
        self.hard_stopped = False

    def bump(self):
        self.count += 1
        if self.count > self.limit:
            raise RuntimeError(f"[odds_api] Request budget exceeded: {self.count}>{self.limit}")


_BUDGET = _OddsBudget(ODDS_MAX_REQUESTS)


def _get_api_key() -> Optional[str]:
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")


def _debug_headers(r: requests.Response, prefix: str = "[odds_api DEBUG]") -> None:
    try:
        print(f"{prefix} status: {r.status_code}")
        print(f"{prefix} url: {r.url}")
        print(f"{prefix} remaining: {r.headers.get('x-requests-remaining')}")
        print(f"{prefix} used: {r.headers.get('x-requests-used')}")
        print(f"{prefix} last: {r.headers.get('x-requests-last')}")
    except Exception:
        pass


def _remaining_credits(r: requests.Response) -> Optional[int]:
    rem = r.headers.get("x-requests-remaining")
    if rem is None:
        return None
    try:
        return int(rem)
    except Exception:
        return None


def _odds_api_get(url: str, params: dict, *, timeout: int = DEFAULT_TIMEOUT) -> Optional[List[Dict[str, Any]]]:
    if _BUDGET.hard_stopped:
        return None

    _BUDGET.bump()
    r = requests.get(url, params=params, timeout=timeout)
    _debug_headers(r)

    if r.status_code == 401:
        print("[odds_api] WARNING: 401 unauthorized (often means credits exhausted).")
        if ODDS_HARD_STOP_ON_401:
            _BUDGET.hard_stopped = True
            print("[odds_api] HARD STOP enabled -> no more Odds API calls this run.")
            return None
        return []

    rem = _remaining_credits(r)
    if rem is not None and rem < ODDS_MIN_REMAINING:
        print(f"[odds_api] WARNING: low remaining credits ({rem}<{ODDS_MIN_REMAINING}). Stopping further calls.")
        _BUDGET.hard_stopped = True

    if r.status_code == 429:
        time.sleep(2.0)

    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    return data


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_best_bookmaker_market(event: dict, market_key: str) -> Optional[dict]:
    books = event.get("bookmakers") or []
    for b in books:
        markets = b.get("markets") or []
        for m in markets:
            if m.get("key") == market_key:
                return m
    return None


def _extract_h2h(event: dict) -> Tuple[Optional[float], Optional[float]]:
    m = _extract_best_bookmaker_market(event, "h2h")
    if not m:
        return (None, None)
    outcomes = m.get("outcomes") or []
    home_name = event.get("home_team")
    away_name = event.get("away_team")
    home_ml = None
    away_ml = None
    for o in outcomes:
        if o.get("name") == home_name:
            home_ml = o.get("price")
        elif o.get("name") == away_name:
            away_ml = o.get("price")
    return (home_ml, away_ml)


def _extract_spreads(event: dict) -> Tuple[Optional[float], Optional[float]]:
    m = _extract_best_bookmaker_market(event, "spreads")
    if not m:
        return (None, None)
    outcomes = m.get("outcomes") or []
    home_name = event.get("home_team")
    home_spread = None
    spread_price = None
    for o in outcomes:
        if o.get("name") == home_name:
            home_spread = o.get("point")
            spread_price = o.get("price")
    return (home_spread, spread_price)


def _extract_totals(event: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    m = _extract_best_bookmaker_market(event, "totals")
    if not m:
        return (None, None, None)
    outcomes = m.get("outcomes") or []
    total_points = None
    over_price = None
    under_price = None
    for o in outcomes:
        nm = str(o.get("name", "")).lower()
        if total_points is None and o.get("point") is not None:
            total_points = o.get("point")
        if "over" in nm:
            over_price = o.get("price")
        elif "under" in nm:
            under_price = o.get("price")
    return (total_points, over_price, under_price)


def load_odds_for_date_from_api(
    sport_key: str,
    commence_from: datetime,
    commence_to: datetime,
    *,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    api_key = _get_api_key()
    if not api_key:
        print("[odds_api] WARNING: ODDS_API_KEY missing.")
        return {}

    url = f"{ODDS_API_HOST}/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": _iso_z(commence_from),
        "commenceTimeTo": _iso_z(commence_to),
    }

    data = _odds_api_get(url, params=params)
    if data is None or not data:
        return {}

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ev in data:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        home_ml, away_ml = _extract_h2h(ev)
        home_spread, spread_price = _extract_spreads(ev)
        total_points, over_price, under_price = _extract_totals(ev)

        out[(home, away)] = {
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
    if not csv_path or not os.path.exists(csv_path):
        return {}

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        def sf(x):
            try:
                if x is None or str(x).strip() == "":
                    return None
                return float(x)
            except Exception:
                return None

        for row in reader:
            home = (row.get("home") or "").strip()
            away = (row.get("away") or "").strip()
            if not home or not away:
                continue

            out[(home, away)] = {
                "home_ml": sf(row.get("home_ml")),
                "away_ml": sf(row.get("away_ml")),
                "home_spread": sf(row.get("home_spread")),
                "spread_price": sf(row.get("spread_price")),
                "total_points": sf(row.get("total_points")),
                "over_price": sf(row.get("over_price")),
                "under_price": sf(row.get("under_price")),
            }
    return out


# ----------------------------
# Compatibility wrappers (what your runner imports)
# ----------------------------
def fetch_odds_for_date_from_odds_api(
    game_date_str: str,
    *,
    sport_key: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    days_padding: int = 1,  # accept it even if caller passes it
) -> Tuple[dict, dict]:
    """
    Returns (odds_dict, spreads_dict). spreads_dict is kept for legacy; we just return {}.
    """
    try:
        d = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        d = datetime.now(timezone.utc).date()

    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc) - timedelta(days=int(days_padding))
    end = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc) + timedelta(days=int(days_padding))

    odds_dict = load_odds_for_date_from_api(
        sport_key=sport_key,
        commence_from=start,
        commence_to=end,
        regions=regions,
        markets=markets,
    )
    return odds_dict, {}


def fetch_odds_for_date_from_csv(
    game_date_str: str,
    *,
    sport: str,
) -> Tuple[dict, dict]:
    mmddyyyy = str(game_date_str).replace("/", "-")
    csv_path = os.path.join("odds", f"odds_{mmddyyyy}.csv")
    odds_dict = load_odds_for_date_from_csv(csv_path)
    print(f"[odds_csv] games found: {len(odds_dict)}")
    return odds_dict, {}
