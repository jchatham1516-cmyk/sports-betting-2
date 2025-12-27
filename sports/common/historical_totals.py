# sports/common/historical_totals.py
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

ODDS_API_HOST = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 20

# Hard cap on total HTTP requests this module will make per run
HIST_MAX_REQUESTS = int(os.getenv("ODDS_HIST_TOTALS_MAX_REQUESTS", "60"))

# Additional caps to prevent "events explosion"
HIST_MAX_EVENT_ODDS_CALLS = int(os.getenv("ODDS_HIST_TOTALS_MAX_EVENT_ODDS_CALLS", "35"))
HIST_MAX_EVENTS_PER_DAY = int(os.getenv("ODDS_HIST_TOTALS_MAX_EVENTS_PER_DAY", "50"))

# Tiny spacing to be polite to API (and reduce bursty 429s)
HIST_SLEEP_S = float(os.getenv("ODDS_HIST_TOTALS_SLEEP_S", "0.15"))

# Default bookmaker to anchor totals history
HIST_BOOKMAKERS = os.getenv("ODDS_HIST_TOTALS_BOOKMAKERS", "draftkings")

# NEW: How far back we FETCH (once per day). Example: 365, 730, etc.
HIST_FETCH_DAYS = int(os.getenv("ODDS_HIST_TOTALS_FETCH_DAYS", "365"))

# NEW: If set, never hit the API; use cache only.
HIST_DISABLE_FETCH = os.getenv("ODDS_HIST_TOTALS_DISABLE_FETCH", "0") == "1"


@dataclass
class _HistBudget:
    limit: int
    used: int = 0
    hard_stop: bool = False

    def allow_one_more(self) -> bool:
        if self.hard_stop:
            return False
        if self.used >= self.limit:
            self.hard_stop = True
            return False
        self.used += 1
        return True


def _get_api_key() -> Optional[str]:
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")


def _debug_headers(r: requests.Response) -> None:
    try:
        print(f"[hist_totals DEBUG] status: {r.status_code} url: {r.url}")
        print(f"[hist_totals DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
        print(f"[hist_totals DEBUG] used: {r.headers.get('x-requests-used')}")
        print(f"[hist_totals DEBUG] last: {r.headers.get('x-requests-last')}")
    except Exception:
        pass


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_json(url: str, params: dict, budget: _HistBudget) -> Any:
    if not budget.allow_one_more():
        return None

    try:
        r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    except Exception:
        return None

    _debug_headers(r)

    if r.status_code == 401:
        print("[hist_totals] WARNING: 401 unauthorized; stopping historical totals calls.")
        budget.hard_stop = True
        return None

    if r.status_code == 429:
        time.sleep(1.5)

    try:
        r.raise_for_status()
    except Exception:
        return None

    time.sleep(HIST_SLEEP_S)

    try:
        return r.json()
    except Exception:
        return None


def _cache_path(sport_key: str) -> str:
    os.makedirs("results", exist_ok=True)
    safe = sport_key.replace("/", "_")
    return os.path.join("results", f"hist_totals_cache_{safe}.json")


def _load_cache(sport_key: str) -> Optional[dict]:
    p = _cache_path(sport_key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(sport_key: str, payload: dict) -> None:
    p = _cache_path(sport_key)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _extract_total_from_hist_odds_payload(payload: dict) -> Optional[float]:
    if payload is None:
        return None

    ev = payload.get("data") if isinstance(payload, dict) else None
    if ev is None and isinstance(payload, dict):
        ev = payload

    books = (ev or {}).get("bookmakers") or []
    for b in books:
        markets = b.get("markets") or []
        for m in markets:
            if m.get("key") != "totals":
                continue
            outs = m.get("outcomes") or []
            for o in outs:
                pt = o.get("point")
                if pt is None:
                    continue
                try:
                    return float(pt)
                except Exception:
                    continue
    return None


def _as_utc_date(d: str) -> Optional[date]:
    try:
        return datetime.fromisoformat(d).date()
    except Exception:
        return None


def build_team_historical_total_lines(
    *,
    sport_key: str,
    days_back: int = 14,
    minutes_before_commence: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      { team_name: {"avg": float, "sd": float, "n": int} }

    New behavior:
      - Fetch RAW totals lines (with dates) up to HIST_FETCH_DAYS ONCE per UTC day.
      - Reuse that raw cache for any 'days_back' <= cached window with ZERO extra API calls.
      - Optional cache-only mode: set ODDS_HIST_TOTALS_DISABLE_FETCH=1
    """
    api_key = _get_api_key()
    if not api_key and not HIST_DISABLE_FETCH:
        print("[hist_totals] WARNING: ODDS_API_KEY missing. Returning empty totals history.")
        return {}

    today_utc = datetime.now(timezone.utc).date().isoformat()

    cache = _load_cache(sport_key) or {}

    # Cache is "daily raw": depends on date + bookmaker + mins + fetch_days (NOT days_back)
    cache_key = (
        f"{today_utc}|fetch_days={int(HIST_FETCH_DAYS)}|mins={int(minutes_before_commence)}|bm={HIST_BOOKMAKERS}"
    )

    raw = None
    if cache.get("cache_key") == cache_key and isinstance(cache.get("raw_team_lines"), dict):
        raw = cache.get("raw_team_lines")

    if raw is None:
        if HIST_DISABLE_FETCH:
            print("[hist_totals] cache-only mode enabled but no valid cache found; returning empty.")
            return {}

        # ---- FETCH once per day ----
        budget = _HistBudget(limit=HIST_MAX_REQUESTS)
        raw_team_lines: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        odds_calls_used = 0

        for d in range(int(HIST_FETCH_DAYS)):
            if budget.hard_stop:
                break
            if odds_calls_used >= HIST_MAX_EVENT_ODDS_CALLS:
                break

            query_dt = datetime.now(timezone.utc) - timedelta(days=d)
            query_dt = query_dt.replace(hour=12, minute=0, second=0, microsecond=0)

            events_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events"
            events_params = {"apiKey": api_key, "date": _iso_z(query_dt), "dateFormat": "iso"}

            events = _get_json(events_url, events_params, budget)
            if events is None:
                break

            if not isinstance(events, list) or not events:
                continue

            events = events[: max(1, int(HIST_MAX_EVENTS_PER_DAY))]

            for ev in events:
                if budget.hard_stop or odds_calls_used >= HIST_MAX_EVENT_ODDS_CALLS:
                    break
                try:
                    event_id = ev.get("id")
                    home = ev.get("home_team")
                    away = ev.get("away_team")
                    commence = ev.get("commence_time")
                    if not event_id or not home or not away or not commence:
                        continue

                    try:
                        cdt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                    except Exception:
                        cdt = query_dt

                    hist_dt = cdt - timedelta(minutes=int(minutes_before_commence))

                    odds_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events/{event_id}/odds"
                    odds_params = {
                        "apiKey": api_key,
                        "date": _iso_z(hist_dt),
                        "regions": "us",
                        "markets": "totals",
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                        "bookmakers": HIST_BOOKMAKERS,
                    }

                    payload = _get_json(odds_url, odds_params, budget)
                    if payload is None:
                        budget.hard_stop = True
                        break

                    odds_calls_used += 1

                    total_line = _extract_total_from_hist_odds_payload(payload)
                    if total_line is None:
                        continue

                    # store with the game's UTC date so we can filter later
                    game_day = cdt.astimezone(timezone.utc).date().isoformat()
                    raw_team_lines[str(home)].append((game_day, float(total_line)))
                    raw_team_lines[str(away)].append((game_day, float(total_line)))

                except Exception:
                    continue

            # early stop if we’ve mostly spent odds calls
            if odds_calls_used >= max(10, int(HIST_MAX_EVENT_ODDS_CALLS * 0.8)):
                break

        raw = {k: v for k, v in raw_team_lines.items()}

        _save_cache(
            sport_key,
            {
                "cache_key": cache_key,
                "asof_date": today_utc,
                "raw_team_lines": raw,
                "meta": {
                    "requests_used": int(budget.used),
                    "odds_calls_used": int(odds_calls_used),
                    "max_requests": int(HIST_MAX_REQUESTS),
                    "max_event_odds_calls": int(HIST_MAX_EVENT_ODDS_CALLS),
                    "fetch_days": int(HIST_FETCH_DAYS),
                    "minutes_before_commence": int(minutes_before_commence),
                    "bookmakers": str(HIST_BOOKMAKERS),
                },
            },
        )

    # ---- COMPUTE stats from cached raw for requested days_back ----
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=int(days_back))
    out: Dict[str, Dict[str, float]] = {}

    for team, pts in (raw or {}).items():
        if not isinstance(pts, list) or not pts:
            continue

        vals: List[float] = []
        for d_str, line in pts:
            dd = _as_utc_date(str(d_str))
            if dd is None:
                continue
            if dd < cutoff:
                continue
            try:
                vals.append(float(line))
            except Exception:
                continue

        if not vals:
            continue

        arr = np.array(vals, dtype=float)
        if arr.size < 2:
            avg = float(np.mean(arr))
            sd = float("nan")
        else:
            avg = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1))

        out[str(team)] = {"avg": avg, "sd": sd, "n": int(arr.size)}

    return out
