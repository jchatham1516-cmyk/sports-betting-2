# sports/common/historical_totals.py
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

ODDS_API_HOST = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 20

# OPTION B knobs (safe + not thousands of requests)
HIST_MAX_REQUESTS = int(os.getenv("ODDS_HIST_TOTALS_MAX_REQUESTS", "60"))     # hard cap
HIST_SLEEP_S = float(os.getenv("ODDS_HIST_TOTALS_SLEEP_S", "0.10"))           # spacing
HIST_DAYS_BACK_DEFAULT = int(os.getenv("ODDS_HIST_TOTALS_DAYS_BACK", "14"))   # typical 14
HIST_EVENTS_PER_DAY = int(os.getenv("ODDS_HIST_EVENTS_PER_DAY", "6"))         # sample only a few events/day
HIST_MIN_TOTAL_LINES = int(os.getenv("ODDS_HIST_MIN_TOTAL_LINES", "20"))      # if cache < this, recompute
HIST_BOOKMAKERS = os.getenv("ODDS_HIST_TOTALS_BOOKMAKERS", "draftkings,fanduel,betmgm,caesars")


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
        time.sleep(1.0)

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


def _extract_total_from_hist_odds_payload(payload: dict) -> Optional[float]:
    if payload is None:
        return None

    # historical odds endpoint often returns {"data": {...event...}}
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
                if pt is not None:
                    try:
                        return float(pt)
                    except Exception:
                        continue
    return None


def build_team_historical_total_lines(
    *,
    sport_key: str,
    days_back: int = HIST_DAYS_BACK_DEFAULT,
    minutes_before_commence: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    OPTION B:
      - cache daily
      - if cache exists but is "too empty" -> recompute
      - request cap via HIST_MAX_REQUESTS
      - sample only HIST_EVENTS_PER_DAY events/day
      - bookmaker fallback list
    """
    api_key = _get_api_key()
    if not api_key:
        print("[hist_totals] WARNING: ODDS_API_KEY missing. Returning empty totals history.")
        return {}

    today_utc = datetime.now(timezone.utc).date().isoformat()
    cache = _load_cache(sport_key) or {}

    # Cache validation: must be same day AND contain enough data
    if cache.get("asof_date") == today_utc and isinstance(cache.get("team_totals"), dict):
        n_lines = int(cache.get("n_total_lines", 0) or 0)
        if n_lines >= HIST_MIN_TOTAL_LINES:
            return cache["team_totals"]
        else:
            print(f"[hist_totals] Cache exists but too small (n_total_lines={n_lines}<{HIST_MIN_TOTAL_LINES}), recomputing...")

    budget = _HistBudget(limit=HIST_MAX_REQUESTS)
    totals_by_team: Dict[str, List[float]] = defaultdict(list)

    bookmakers = [b.strip() for b in str(HIST_BOOKMAKERS).split(",") if b.strip()]

    # loop backward over days
    for d in range(int(days_back)):
        if budget.hard_stop:
            break

        query_dt = datetime.now(timezone.utc) - timedelta(days=d)
        query_dt = query_dt.replace(hour=12, minute=0, second=0, microsecond=0)

        # 1) list events for that date
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

        # sample only a few events/day to control calls
        sampled = 0

        for ev in events:
            if sampled >= HIST_EVENTS_PER_DAY:
                break
            if budget.hard_stop:
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
                    continue

                hist_dt = cdt - timedelta(minutes=int(minutes_before_commence))

                got_line = None
                for bm in bookmakers:
                    odds_url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events/{event_id}/odds"
                    odds_params = {
                        "apiKey": api_key,
                        "date": _iso_z(hist_dt),
                        "regions": "us",
                        "markets": "totals",
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                        "bookmakers": bm,
                    }

                    payload = _get_json(odds_url, odds_params, budget)
                    if payload is None:
                        budget.hard_stop = True
                        break

                    got_line = _extract_total_from_hist_odds_payload(payload)
                    if got_line is not None:
                        break  # bookmaker success

                if got_line is None:
                    continue

                totals_by_team[str(home)].append(float(got_line))
                totals_by_team[str(away)].append(float(got_line))
                sampled += 1

            except Exception as e:
                if "Request budget exceeded" in str(e):
                    print(f"[hist_totals] {e} -> stopping.")
                    budget.hard_stop = True
                    break
                continue

    out: Dict[str, Dict[str, float]] = {}
    n_total_lines = 0
    for team, lines in totals_by_team.items():
        if not lines:
            continue
        arr = np.array(lines, dtype=float)
        n_total_lines += int(arr.size)
        avg = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan")
        out[str(team)] = {"avg": avg, "sd": sd, "n": int(arr.size)}

    _save_cache(sport_key, {"asof_date": today_utc, "team_totals": out, "n_total_lines": int(n_total_lines)})
    return out
