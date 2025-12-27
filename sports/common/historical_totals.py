# sports/common/historical_totals.py
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

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

# When days_back is large, sample 1 day every N days (keeps requests sane)
HIST_SAMPLE_STEP_DAYS = int(os.getenv("ODDS_HIST_TOTALS_SAMPLE_STEP_DAYS", "7"))
HIST_SAMPLE_ONLY_IF_DAYS_BACK_GE = int(os.getenv("ODDS_HIST_TOTALS_SAMPLE_ONLY_IF_DAYS_BACK_GE", "60"))


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


def build_team_historical_total_lines(
    *,
    sport_key: str,
    days_back: int = 14,
    minutes_before_commence: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      { team_name: {"avg": float, "sd": float, "n": int} }

    Cache behavior:
      - Cache is keyed by *today's UTC date* + params, so it refreshes at most once per day.

    Sampling behavior:
      - If days_back is large, we sample every HIST_SAMPLE_STEP_DAYS to keep requests sane.
    """
    api_key = _get_api_key()
    if not api_key:
        print("[hist_totals] WARNING: ODDS_API_KEY missing. Returning empty totals history.")
        return {}

    today_utc = datetime.now(timezone.utc).date().isoformat()

    # sampling step
    step = 1
    if int(days_back) >= int(HIST_SAMPLE_ONLY_IF_DAYS_BACK_GE):
        step = max(1, int(HIST_SAMPLE_STEP_DAYS))

    cache = _load_cache(sport_key) or {}
    cache_key = (
        f"{today_utc}|days_back={int(days_back)}|step={int(step)}|mins={int(minutes_before_commence)}|"
        f"bm={HIST_BOOKMAKERS}|maxreq={int(HIST_MAX_REQUESTS)}|maxodds={int(HIST_MAX_EVENT_ODDS_CALLS)}"
    )
    if cache.get("cache_key") == cache_key and isinstance(cache.get("team_totals"), dict):
        return cache["team_totals"]

    budget = _HistBudget(limit=HIST_MAX_REQUESTS)
    totals_by_team: Dict[str, List[float]] = defaultdict(list)
    odds_calls_used = 0

    for d in range(0, int(days_back), int(step)):
        if budget.hard_stop:
            break
        if odds_calls_used >= HIST_MAX_EVENT_ODDS_CALLS:
            break

        query_dt = datetime.now(timezone.utc) - timedelta(days=int(d))
        query_dt = query_dt.replace(hour=12, minute=0, second=0, microsecond=0)

        # 1) list events for that date
        events_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events"
        events_params = {"apiKey": api_key, "date": _iso_z(query_dt), "dateFormat": "iso"}

        events = _get_json(events_url, events_params, budget)
        if events is None:
            break

        if not isinstance(events, list) or not events:
            continue

        events = events[: max(1, int(HIST_MAX_EVENTS_PER_DAY))]

        # 2) per-event totals odds
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

                totals_by_team[str(home)].append(float(total_line))
                totals_by_team[str(away)].append(float(total_line))

            except Exception:
                continue

        if odds_calls_used >= max(10, int(HIST_MAX_EVENT_ODDS_CALLS * 0.85)):
            break

    out: Dict[str, Dict[str, float]] = {}
    for team, lines in totals_by_team.items():
        if not lines:
            continue
        arr = np.array(lines, dtype=float)
        avg = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan")
        out[str(team)] = {"avg": avg, "sd": sd, "n": int(arr.size)}

    _save_cache(
        sport_key,
        {
            "cache_key": cache_key,
            "asof_date": today_utc,
            "team_totals": out,
            "meta": {
                "requests_used": int(budget.used),
                "odds_calls_used": int(odds_calls_used),
                "max_requests": int(HIST_MAX_REQUESTS),
                "max_event_odds_calls": int(HIST_MAX_EVENT_ODDS_CALLS),
                "days_back": int(days_back),
                "sample_step_days": int(step),
                "minutes_before_commence": int(minutes_before_commence),
                "bookmakers": str(HIST_BOOKMAKERS),
            },
        },
    )

    return out
