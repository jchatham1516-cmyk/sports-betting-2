# sports/common/historical_totals.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


ODDS_API_HOST = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 20

# ---- Fix #2: hard request budget ----
MAX_HIST_REQUESTS = int(os.getenv("ODDS_MAX_HIST_REQUESTS", "60"))

# ---- Fix #3: stop once you have enough samples ----
TARGET_LINES_PER_TEAM = int(os.getenv("ODDS_TOTAL_LINES_PER_TEAM", "8"))

# ---- Fix #4: keep history pulls narrow ----
HIST_REGIONS = os.getenv("ODDS_HIST_REGIONS", "us")
HIST_BOOKMAKERS = os.getenv("ODDS_HIST_BOOKMAKER", "draftkings")  # ONE book only
HIST_MARKETS = "totals"


class _HistBudget:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.count = 0
        self.hard_stopped = False

    def bump(self):
        self.count += 1
        if self.count > self.limit:
            raise RuntimeError(f"[hist_totals] Request budget exceeded: {self.count}>{self.limit}")


_BUDGET = _HistBudget(MAX_HIST_REQUESTS)


def _get_api_key() -> Optional[str]:
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _debug_headers(r: requests.Response, prefix: str = "[hist_totals DEBUG]") -> None:
    try:
        print(f"{prefix} status: {r.status_code} url: {r.url}")
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


def _hist_get(url: str, params: dict) -> Optional[dict]:
    """
    Returns dict json on success.
    Returns None if we should stop this run (401/low credits/budget).
    """
    if _BUDGET.hard_stopped:
        return None

    _BUDGET.bump()
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    _debug_headers(r)

    if r.status_code == 401:
        print("[hist_totals] WARNING: 401 unauthorized. Check key OR credits exhausted.")
        _BUDGET.hard_stopped = True
        return None

    rem = _remaining_credits(r)
    if rem is not None and rem < 10:
        print(f"[hist_totals] WARNING: low remaining credits ({rem}). Stopping further history calls.")
        _BUDGET.hard_stopped = True

    if r.status_code == 429:
        time.sleep(2.0)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        return {}
    return data


def _extract_total_from_event_odds(ev_odds: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Given a single event odds payload from /odds endpoint,
    pull totals: (points, over_price, under_price).
    """
    books = ev_odds.get("bookmakers") or []
    for b in books:
        # enforce single bookmaker if present
        if HIST_BOOKMAKERS and b.get("key") != HIST_BOOKMAKERS:
            continue
        markets = b.get("markets") or []
        for m in markets:
            if m.get("key") != "totals":
                continue
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
    return (None, None, None)


def build_team_historical_total_lines(
    sport_key: str,
    days_back: int = 14,
    minutes_before_commence: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Builds per-team historical MARKET total lines using the historical endpoint.

    Output:
      { team_name: { "avg": float, "sd": float, "n": int } }

    This is intentionally request-limited:
      - hard budget MAX_HIST_REQUESTS
      - early-stop once TARGET_LINES_PER_TEAM reached for teams seen
      - 1 region, 1 bookmaker, totals market only
    """
    api_key = _get_api_key()
    if not api_key:
        print("[hist_totals] WARNING: missing ODDS_API_KEY")
        return {}

    # We collect historical totals lines for teams
    team_lines: Dict[str, List[float]] = {}

    now = datetime.now(timezone.utc)
    # Walk back day by day
    for d in range(days_back, 0, -1):
        # query “events at date”
        dt = now - timedelta(days=d)
        date_param = _iso_z(dt.replace(hour=23, minute=0, second=0, microsecond=0))

        events_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events"
        events_params = {"apiKey": api_key, "date": date_param}

        events_payload = _hist_get(events_url, events_params)
        if events_payload is None:
            break

        events = events_payload.get("data") or events_payload.get("events") or []
        if not isinstance(events, list) or not events:
            # nothing this day
            continue

        # For each event, call historical odds (totals only, 1 book)
        for ev in events:
            if _BUDGET.hard_stopped:
                break

            home = ev.get("home_team")
            away = ev.get("away_team")
            event_id = ev.get("id")
            commence = ev.get("commence_time")

            if not home or not away or not event_id or not commence:
                continue

            # Fix #3: skip odds calls if both teams already have enough samples
            if len(team_lines.get(home, [])) >= TARGET_LINES_PER_TEAM and len(team_lines.get(away, [])) >= TARGET_LINES_PER_TEAM:
                continue

            # Pull odds “minutes_before_commence” prior
            # (Odds API historical uses a "date" parameter - we approximate by using commence minus minutes)
            try:
                ct = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
            except Exception:
                continue
            query_time = ct - timedelta(minutes=int(minutes_before_commence))
            query_time = query_time.astimezone(timezone.utc)

            odds_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events/{event_id}/odds"
            odds_params = {
                "apiKey": api_key,
                "date": _iso_z(query_time),
                "regions": HIST_REGIONS,
                "markets": HIST_MARKETS,
                "oddsFormat": "american",
                "dateFormat": "iso",
                "bookmakers": HIST_BOOKMAKERS,
            }

            odds_payload = _hist_get(odds_url, odds_params)
            if odds_payload is None:
                break

            # the event odds data is usually under odds_payload["data"] (varies by plan/version)
            ev_odds = odds_payload.get("data") if isinstance(odds_payload.get("data"), dict) else odds_payload
            if not isinstance(ev_odds, dict):
                continue

            total_points, _, _ = _extract_total_from_event_odds(ev_odds)
            if total_points is None:
                continue

            try:
                tp = float(total_points)
            except Exception:
                continue

            team_lines.setdefault(home, []).append(tp)
            team_lines.setdefault(away, []).append(tp)

            # Fix #3: early-stop once all teams we’ve seen are “full”
            if team_lines and all(len(v) >= TARGET_LINES_PER_TEAM for v in team_lines.values()):
                break

        if team_lines and all(len(v) >= TARGET_LINES_PER_TEAM for v in team_lines.values()):
            break

    # Aggregate
    out: Dict[str, Dict[str, float]] = {}
    for team, lines in team_lines.items():
        if not lines:
            continue
        arr = np.array(lines, dtype=float)
        avg = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")
        out[team] = {"avg": avg, "sd": sd, "n": int(len(lines))}

    return out
