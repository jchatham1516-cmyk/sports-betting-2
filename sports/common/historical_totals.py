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

# OPTION C knobs
HIST_MAX_REQUESTS = int(os.getenv("ODDS_HIST_TOTALS_MAX_REQUESTS", "60"))  # hard cap
HIST_SLEEP_S = float(os.getenv("ODDS_HIST_TOTALS_SLEEP_S", "0.15"))        # tiny spacing
HIST_BOOKMAKERS = os.getenv("ODDS_HIST_TOTALS_BOOKMAKERS", "draftkings")   # can change later

# Cache safety: don't reuse an empty cache all day
HIST_CACHE_MIN_TEAMS = int(os.getenv("ODDS_HIST_TOTALS_CACHE_MIN_TEAMS", "10"))
HIST_CACHE_MIN_LINES = int(os.getenv("ODDS_HIST_TOTALS_CACHE_MIN_LINES", "30"))


@dataclass
class _HistBudget:
    limit: int
    used: int = 0
    hard_stop: bool = False

    def bump(self):
        self.used += 1
        if self.used > self.limit:
            raise RuntimeError(f"[hist_totals] Request budget exceeded: {self.used}>{self.limit}")


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
    if budget.hard_stop:
        return None
    budget.bump()
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    _debug_headers(r)

    if r.status_code == 401:
        print("[hist_totals] WARNING: 401 unauthorized; stopping historical totals calls.")
        budget.hard_stop = True
        return None

    if r.status_code == 429:
        time.sleep(1.5)

    r.raise_for_status()
    time.sleep(HIST_SLEEP_S)
    return r.json()


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


def _extract_events_list(events_payload: Any) -> List[dict]:
    """
    Odds API historical events endpoint can return:
      - list[events]
      - {"data": list[events]}
      - {"data": {"events": list[events]}} (rare)
    This normalizes to list[dict].
    """
    if events_payload is None:
        return []

    if isinstance(events_payload, list):
        return [e for e in events_payload if isinstance(e, dict)]

    if isinstance(events_payload, dict):
        d = events_payload.get("data")
        if isinstance(d, list):
            return [e for e in d if isinstance(e, dict)]
        if isinstance(d, dict):
            maybe = d.get("events")
            if isinstance(maybe, list):
                return [e for e in maybe if isinstance(e, dict)]
    return []


def _extract_total_from_hist_odds_payload(payload: Any) -> Optional[float]:
    """
    Historical odds endpoint commonly returns:
      - {"data": {event-like dict with bookmakers...}}
      - {"data": [event-like dicts...]}   (sometimes)
      - {event-like dict...}
    We look for totals market point.
    """
    if payload is None:
        return None

    candidates: List[dict] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), dict):
            candidates.append(payload["data"])
        elif isinstance(payload.get("data"), list):
            for x in payload["data"]:
                if isinstance(x, dict):
                    candidates.append(x)
        else:
            candidates.append(payload)
    elif isinstance(payload, list):
        for x in payload:
            if isinstance(x, dict):
                candidates.append(x)

    for ev in candidates:
        books = (ev or {}).get("bookmakers") or []
        for b in books:
            markets = b.get("markets") or []
            for m in markets:
                if m.get("key") != "totals":
                    continue
                outs = m.get("outcomes") or []
                for o in outs:
                    pt = o.get("point")
                    if pt is not None:
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

    OPTION C behavior:
      - uses cache if already computed today AND it's not trivially empty
      - caps total HTTP requests (events list + per-event odds) via HIST_MAX_REQUESTS
    """
    api_key = _get_api_key()
    if not api_key:
        print("[hist_totals] WARNING: ODDS_API_KEY missing. Returning empty totals history.")
        return {}

    today_utc = datetime.now(timezone.utc).date().isoformat()
    cache = _load_cache(sport_key) or {}

    cached_team_totals = cache.get("team_totals") if isinstance(cache, dict) else None
    if cache.get("asof_date") == today_utc and isinstance(cached_team_totals, dict):
        # Only trust cache if it's non-trivially populated
        try:
            teams_n = len(cached_team_totals)
            lines_n = int(sum(int(v.get("n", 0)) for v in cached_team_totals.values() if isinstance(v, dict)))
        except Exception:
            teams_n, lines_n = 0, 0

        if teams_n >= HIST_CACHE_MIN_TEAMS and lines_n >= HIST_CACHE_MIN_LINES:
            return cached_team_totals
        # Otherwise ignore cache and recompute (prevents "empty all day" problem)

    budget = _HistBudget(limit=HIST_MAX_REQUESTS)
    totals_by_team: Dict[str, List[float]] = defaultdict(list)

    for d in range(int(days_back)):
        # Stable “query time”
        query_dt = datetime.now(timezone.utc) - timedelta(days=d)
        query_dt = query_dt.replace(hour=12, minute=0, second=0, microsecond=0)

        # 1) list events
        events_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events"
        events_params = {"apiKey": api_key, "date": _iso_z(query_dt), "dateFormat": "iso"}

        try:
            events_payload = _get_json(events_url, events_params, budget)
        except Exception as e:
            print(f"[hist_totals] WARNING: events list failed for {query_dt.date()}: {e}")
            break

        if events_payload is None:
            break

        events = _extract_events_list(events_payload)
        if not events:
            continue

        # 2) for each event, fetch totals odds once
        for ev in events:
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
                    break

                total_line = _extract_total_from_hist_odds_payload(payload)
                if total_line is None:
                    continue

                totals_by_team[str(home)].append(float(total_line))
                totals_by_team[str(away)].append(float(total_line))

            except Exception as e:
                msg = str(e)
                if "Request budget exceeded" in msg:
                    print(f"[hist_totals] {msg} -> stopping.")
                    budget.hard_stop = True
                    break
                continue

        if budget.hard_stop:
            break

    out: Dict[str, Dict[str, float]] = {}
    for team, lines in totals_by_team.items():
        if not lines:
            continue
        arr = np.array(lines, dtype=float)
        avg = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan")
        out[str(team)] = {"avg": avg, "sd": sd, "n": int(arr.size)}

    # Save cache even if small, but tomorrow's run will recompute if too small
    _save_cache(sport_key, {"asof_date": today_utc, "team_totals": out})
    return out
