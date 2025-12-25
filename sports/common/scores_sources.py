# sports/common/scores_sources.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

ODDS_API_HOST = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 20

# Separate budget for scores calls
SCORES_MAX_REQUESTS = int(os.getenv("ODDS_MAX_SCORES_REQUESTS", "12"))
SCORES_MIN_REMAINING = int(os.getenv("ODDS_MIN_REMAINING", "10"))
SCORES_HARD_STOP_ON_401 = os.getenv("ODDS_HARD_STOP_ON_401", "1") == "1"


class _ScoresBudget:
    def __init__(self, limit: int):
        self.limit = int(limit)
        self.count = 0
        self.hard_stopped = False

    def bump(self):
        self.count += 1
        if self.count > self.limit:
            raise RuntimeError(f"[scores_api] Request budget exceeded: {self.count}>{self.limit}")


_BUDGET = _ScoresBudget(SCORES_MAX_REQUESTS)


def _get_api_key() -> Optional[str]:
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")


def _debug_headers(r: requests.Response) -> None:
    try:
        print(f"[scores_api DEBUG] status: {r.status_code}")
        print(f"[scores_api DEBUG] url: {r.url}")
        print(f"[scores_api DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
        print(f"[scores_api DEBUG] used: {r.headers.get('x-requests-used')}")
        print(f"[scores_api DEBUG] last: {r.headers.get('x-requests-last')}")
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


def _scores_api_get(url: str, params: dict) -> Optional[List[Dict[str, Any]]]:
    """
    Returns list on success, [] on empty, None if we should stop this run (401/low credits).
    """
    if _BUDGET.hard_stopped:
        return None

    _BUDGET.bump()
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    _debug_headers(r)

    if r.status_code == 401:
        print("[scores] WARNING: 401 unauthorized (often credits exhausted).")
        if SCORES_HARD_STOP_ON_401:
            _BUDGET.hard_stopped = True
            return None
        return []

    rem = _remaining_credits(r)
    if rem is not None and rem < SCORES_MIN_REMAINING:
        print(f"[scores] WARNING: low remaining credits ({rem}<{SCORES_MIN_REMAINING}). Stopping further calls.")
        _BUDGET.hard_stopped = True

    if r.status_code == 422:
        print("[scores] WARNING: 422 from scores endpoint. Returning no scores.")
        return []

    if r.status_code == 429:
        time.sleep(2.0)

    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    return data


def fetch_recent_scores(sport_key: str, days_from: int = 3) -> List[Dict[str, Any]]:
    """
    Odds API docs: scores endpoint daysFrom valid 1..3.
    We clamp to [1,3].
    """
    api_key = _get_api_key()
    if not api_key:
        print("[scores] WARNING: ODDS_API_KEY missing. Returning no scores.")
        return []

    try:
        df = int(days_from)
    except Exception:
        df = 3
    df = max(1, min(3, df))

    url = f"{ODDS_API_HOST}/v4/sports/{sport_key}/scores/"
    params = {"apiKey": api_key, "daysFrom": df, "dateFormat": "iso"}

    for attempt in range(3):
        try:
            data = _scores_api_get(url, params=params)
            if data is None:
                return []
            return data
        except requests.exceptions.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 429:
                print(f"[scores] WARNING: failed to fetch scores (HTTPError: {e}). Returning no scores.")
                return []
            sleep_s = 3 + 2 * attempt
            print(f"[scores] Rate limited (429) on scores, attempt {attempt+1}/3. Sleeping {sleep_s}s...")
            time.sleep(sleep_s)
        except Exception as e:
            print(f"[scores] WARNING: failed to fetch scores ({type(e).__name__}: {e}). Returning no scores.")
            return []
    return []
