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

HIST_MAX_REQUESTS = int(os.getenv("ODDS_HIST_TOTALS_MAX_REQUESTS", "60"))  # OPTION C cap
HIST_SLEEP_S = float(os.getenv("ODDS_HIST_TOTALS_SLEEP_S", "0.15"))
HIST_BOOKMAKERS = os.getenv("ODDS_HIST_TOTALS_BOOKMAKERS", "draftkings")  # can be ""


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


def _extract_total_from_hist_odds_payload(payload: Any) -> Optional[float]:
    if payload is None:
        return None

    if isinstance(payload, dict):
        ev = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    else:
        return None

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
    OPTION C:
      - cache per UTC day
      - hard cap HTTP calls via HIST_MAX_REQUESTS
      - tries bookmaker filter first; if totals missing, retries once without filter (if budget allows)
    """
    api_key = _get_api_key()
    if not api_key:
        print("[hist_totals] WARNING: ODDS_API_KEY missing. Returning empty totals history.")
        return {}

    today_utc = datetime.now(timezone.utc).date().isoformat()
    cache = _load_cache(sport_key) or {}
    if cache.get("asof_date") == today_utc and isinstance(cache.get("team_totals"), dict):
        return cache["team_totals"]

    budget = _HistBudget(limit=HIST_MAX_REQUESTS)
    totals_by_team: Dict[str, List[float]] = defaultdict(list)

    for d in range(int(days_back)):
        query_dt = datetime.now(timezone.utc) - timedelta(days=d)
        query_dt = query_dt.replace(hour=12, minute=0, second=0, microsecond=0)

        events_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events"
        events_params = {"apiKey": api_key, "date": _iso_z(query_dt), "dateFormat": "iso"}

        try:
            events = _get_json(events_url, events_params, budget)
        except Exception as e:
            print(f"[hist_totals] WARNING: events list failed for {query_dt.date()}: {e}")
            break

        if events is None:
            break
        if not isinstance(events, list) or not events:
            continue

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

                def _odds_params(bookmakers: Optional[str]) -> dict:
                    p = {
                        "apiKey": api_key,
                        "date": _iso_z(hist_dt),
                        "regions": "us",
                        "markets": "totals",
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                    }
                    if bookmakers:
                        p["bookmakers"] = bookmakers
                    return p

                # Try with bookmaker filter first (cheaper payload)
                payload = _get_json(odds_url, _odds_params(HIST_BOOKMAKERS or None), budget)
                if payload is None:
                    break

                total_line = _extract_total_from_hist_odds_payload(payload)

                # If missing and we still have budget, retry once with NO bookmaker filter
                if total_line is None and not budget.hard_stop and (budget.used + 1) <= budget.limit:
                    payload2 = _get_json(odds_url, _odds_params(None), budget)
                    if payload2 is None:
                        break
                    total_line = _extract_total_from_hist_odds_payload(payload2)

                if total_line is None:
                    continue

                totals_by_team[str(home)].append(float(total_line))
                totals_by_team[str(away)].append(float(total_line))

            except Exception as e:
                if "Request budget exceeded" in str(e):
                    print(f"[hist_totals] {e} -> stopping.")
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

    _save_cache(sport_key, {"asof_date": today_utc, "team_totals": out})
    return out
