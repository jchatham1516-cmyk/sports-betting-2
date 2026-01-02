# sports/common/odds_sources.py
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from sports.common.util import implied_prob_from_american, remove_vig_two_way


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
ODDS_PREFERRED_BOOKMAKER = os.getenv("ODDS_PREFERRED_BOOKMAKER", "").lower()


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


def _odds_api_key_present() -> bool:
    return bool(os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API"))


def _get_odds_api_key() -> str:
    key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API") or ""
    if not key:
        raise RuntimeError("Missing ODDS API key. Set ODDS_API_KEY.")
    return key


def _odds_get(path: str, params: Dict[str, Any]) -> Any:
    if _BUDGET.hard_stopped:
        raise RuntimeError("[odds_api] Hard-stopped due to prior 401 or low budget.")

    _BUDGET.bump()
    url = f"{ODDS_API_HOST}{path}"
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    remaining = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    last = r.headers.get("x-requests-last")

    print(f"[odds_api DEBUG] status: {r.status_code}")
    print(f"[odds_api DEBUG] url: {r.url}")
    if remaining is not None:
        print(f"[odds_api DEBUG] remaining: {remaining}")
    if used is not None:
        print(f"[odds_api DEBUG] used: {used}")
    if last is not None:
        print(f"[odds_api DEBUG] last: {last}")

    if r.status_code == 401 and ODDS_HARD_STOP_ON_401:
        _BUDGET.hard_stopped = True
        raise RuntimeError("401 Unauthorized from Odds API. Check ODDS_API_KEY / subscription.")

    r.raise_for_status()

    try:
        rem = int(remaining) if remaining is not None else None
        if rem is not None and rem < ODDS_MIN_REMAINING:
            print(f"[odds_api] WARNING: low remaining requests: {rem}")
    except Exception:
        pass

    return r.json()


def _parse_best_price_from_bookmakers(
    bookmakers: List[Dict[str, Any]],
    market_key: str,
    outcome_name: str,
) -> Optional[int]:
    best: Optional[int] = None
    for bm in bookmakers or []:
        for m in (bm.get("markets") or []):
            if m.get("key") != market_key:
                continue
            for o in (m.get("outcomes") or []):
                if str(o.get("name")) != str(outcome_name):
                    continue
                price = o.get("price")
                if price is None:
                    continue
                try:
                    price = int(price)
                except Exception:
                    continue
                if best is None:
                    best = price
                else:
                    # best payout for user: higher is better for +, less negative is better for -
                    if price > 0 and best > 0:
                        if price > best:
                            best = price
                    elif price < 0 and best < 0:
                        if price > best:  # -110 beats -120
                            best = price
                    else:
                        # mixed signs, compare implied prob is overkill; keep first
                        pass
    return best


def _extract_moneyline_pairs(
    bookmakers: List[Dict[str, Any]], home_team: str, away_team: str
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    for bm in bookmakers or []:
        bm_key = str(bm.get("key") or "")
        bm_title = str(bm.get("title") or "")
        home_price: Optional[int] = None
        away_price: Optional[int] = None

        for m in (bm.get("markets") or []):
            if m.get("key") != "h2h":
                continue

            for o in (m.get("outcomes") or []):
                name = str(o.get("name"))
                price = o.get("price")
                if price is None:
                    continue
                try:
                    price_i = int(price)
                except Exception:
                    continue

                if name == str(home_team):
                    home_price = price_i
                elif name == str(away_team):
                    away_price = price_i

            # only one h2h market per bookmaker expected
            break

        if home_price is None or away_price is None:
            continue

        p_home = implied_prob_from_american(home_price)
        p_away = implied_prob_from_american(away_price)
        vig_removed = remove_vig_two_way(p_home, p_away)
        if vig_removed is None:
            continue
        nv_home, nv_away = vig_removed

        pairs.append(
            {
                "home_ml": home_price,
                "away_ml": away_price,
                "no_vig_home": nv_home,
                "bm_key": bm_key,
                "bm_title": bm_title,
            }
        )

    return pairs


def _select_moneyline_pair(
    bookmakers: List[Dict[str, Any]], home_team: str, away_team: str
) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    pairs = _extract_moneyline_pairs(bookmakers, home_team, away_team)
    if not pairs:
        return None, None, None, None

    pairs.sort(key=lambda p: p.get("no_vig_home", 0.5))
    selected = pairs[len(pairs) // 2]

    return (
        selected.get("home_ml"),
        selected.get("away_ml"),
        selected.get("bm_key"),
        selected.get("bm_title"),
    )


def _parse_spread_from_bookmakers(
    bookmakers: List[Dict[str, Any]],
    home_team: str,
) -> Tuple[Optional[float], Optional[int]]:
    """
    Returns (home_spread, spread_price) from the best-available bookmaker line.
    spread_price is the HOME side price when available, else None.
    """
    best_spread = None
    best_price = None

    for bm in bookmakers or []:
        for m in (bm.get("markets") or []):
            if m.get("key") != "spreads":
                continue
            outs = m.get("outcomes") or []
            # find home outcome
            for o in outs:
                if str(o.get("name")) != str(home_team):
                    continue
                pt = o.get("point")
                pr = o.get("price")
                if pt is None:
                    continue
                try:
                    pt = float(pt)
                except Exception:
                    continue
                if best_spread is None:
                    best_spread = pt
                    try:
                        best_price = int(pr) if pr is not None else None
                    except Exception:
                        best_price = None
    return best_spread, best_price


def _parse_total_from_bookmakers(
    bookmakers: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Returns (total_points, over_price, under_price).
    """
    total_points = None
    over_price = None
    under_price = None

    for bm in bookmakers or []:
        for m in (bm.get("markets") or []):
            if m.get("key") != "totals":
                continue
            outs = m.get("outcomes") or []
            for o in outs:
                name = str(o.get("name") or "")
                pt = o.get("point")
                pr = o.get("price")
                if pt is None:
                    continue
                try:
                    pt = float(pt)
                except Exception:
                    continue

                if total_points is None:
                    total_points = pt

                try:
                    pr_i = int(pr) if pr is not None else None
                except Exception:
                    pr_i = None

                if name.lower() == "over":
                    over_price = pr_i
                elif name.lower() == "under":
                    under_price = pr_i

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
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], Any]]:
    """
    Returns:
      odds_dict[(home, away)] = {
        home_ml, away_ml, home_spread, spread_price, total_points, over_price, under_price, commence_time
      }
      spreads_dict is kept for backward-compat (may be unused)
    """
    if not _odds_api_key_present():
        raise RuntimeError("ODDS_API_KEY not set.")

    key = _get_odds_api_key()

    try:
        dt = datetime.strptime(game_date_str, "%m/%d/%Y").replace(tzinfo=timezone.utc)
    except Exception:
        # fallback: today UTC
        dt = datetime.utcnow().replace(tzinfo=timezone.utc)

    start = (dt - timedelta(days=int(days_padding))).replace(hour=0, minute=0, second=0, microsecond=0)
    end = (dt + timedelta(days=int(days_padding))).replace(hour=23, minute=59, second=59, microsecond=0)

    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "commenceTimeFrom": start.isoformat().replace("+00:00", "Z"),
        "commenceTimeTo": end.isoformat().replace("+00:00", "Z"),
    }

    data = _odds_get(f"/v4/sports/{sport_key}/odds", params=params)
    odds_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    spreads_dict: Dict[Tuple[str, str], Any] = {}

    for ev in data or []:
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not home or not away:
            continue

        event_id = ev.get("id")

        bms = ev.get("bookmakers") or []
        if ODDS_PREFERRED_BOOKMAKER:
            preferred = [
                bm
                for bm in bms
                if str(bm.get("key", "")).lower() == ODDS_PREFERRED_BOOKMAKER
            ]
            if preferred:
                bms = preferred
        home_ml, away_ml, ml_book_key, ml_book_title = _select_moneyline_pair(
            bms, home, away
        )

        home_spread, spread_price = _parse_spread_from_bookmakers(bms, home)
        total_points, over_price, under_price = _parse_total_from_bookmakers(bms)

        odds_dict[(str(home), str(away))] = {
            "event_id": event_id or "",
            "home_ml": float(home_ml) if home_ml is not None else float("nan"),
            "away_ml": float(away_ml) if away_ml is not None else float("nan"),
            "home_spread": float(home_spread) if home_spread is not None else float("nan"),
            "spread_price": float(spread_price) if spread_price is not None else float("nan"),
            "total_points": float(total_points) if total_points is not None else float("nan"),
            "over_price": float(over_price) if over_price is not None else float("nan"),
            "under_price": float(under_price) if under_price is not None else float("nan"),
            "commence_time": ev.get("commence_time"),
            "ml_book_key": ml_book_key or "",
            "ml_book_title": ml_book_title or "",
        }

    return odds_dict, spreads_dict


def fetch_odds_for_date_from_csv(game_date_str: str, *, sport: str) -> Tuple[dict, dict]:
    """
    Fallback loader for local odds CSVs: odds/odds_MM-DD-YYYY.csv
    Expected columns include at least: date, home, away, home_ml, away_ml, home_spread
    Optional: spread_price, total_points, over_price, under_price
    """
    try:
        dt = datetime.strptime(game_date_str, "%m/%d/%Y")
    except Exception:
        dt = datetime.utcnow()
    fname = f"odds/odds_{dt.strftime('%m-%d-%Y')}.csv"
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    odds_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    spreads_dict: Dict[Tuple[str, str], Any] = {}

    with open(fname, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            home = (row.get("home") or "").strip()
            away = (row.get("away") or "").strip()
            if not home or not away:
                continue

            event_id = row.get("event_id")

            def sf(x):
                try:
                    return float(x)
                except Exception:
                    return float("nan")

            odds_dict[(home, away)] = {
                "event_id": event_id or "",
                "home_ml": sf(row.get("home_ml")),
                "away_ml": sf(row.get("away_ml")),
                "home_spread": sf(row.get("home_spread")),
                "spread_price": sf(row.get("spread_price")),
                "total_points": sf(row.get("total_points")),
                "over_price": sf(row.get("over_price")),
                "under_price": sf(row.get("under_price")),
                "commence_time": row.get("commence_time") or "",
            }

    return odds_dict, spreads_dict
