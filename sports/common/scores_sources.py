# sports/common/scores_sources.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from sports.common.odds_sources import SPORT_TO_ODDS_KEY

ODDS_API_HOST = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 20

def _get_api_key() -> Optional[str]:
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")

def _odds_api_get(url: str, params: dict) -> List[Dict[str, Any]]:
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    # helpful debug (your runner already prints these, but keeping it safe)
    try:
        print(f"[scores_api DEBUG] status: {r.status_code}")
        print(f"[scores_api DEBUG] url: {r.url}")
        print(f"[scores_api DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
        print(f"[scores_api DEBUG] used: {r.headers.get('x-requests-used')}")
        print(f"[scores_api DEBUG] last: {r.headers.get('x-requests-last')}")
    except Exception:
        pass

    # 401 should not crash the whole model run
    if r.status_code == 401:
        print("[scores] WARNING: unauthorized (401). Check ODDS_API_KEY / plan. Returning no scores.")
        return []

    # 422 happens if daysFrom is invalid (Odds API docs: valid 1..3)
    if r.status_code == 422:
        print("[scores] WARNING: 422 from Odds API scores endpoint. Returning no scores.")
        return []

    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    return data

def fetch_recent_scores(sport_key: str, days_from: int = 3) -> List[Dict[str, Any]]:
    """
    # Odds API scores endpoint often rejects large daysFrom for some sports (422).
# We clamp to 3 to avoid returning zero scores.
if sport_key in ("basketball_nba", "americanfootball_nfl") and days_from > 3:
    days_from = 3
    Fetch recent scores for a sport using The Odds API scores endpoint.

    IMPORTANT:
    Odds API docs: daysFrom valid integers from 1 to 3.
    If caller requests >3, we clamp to 3 so we don't get HTTP 422.
    """
    api_key = _get_api_key()
    if not api_key:
        print("[scores] WARNING: ODDS_API_KEY missing. Returning no scores.")
        return []

    # clamp days_from to [1, 3] because Odds API scores endpoint only supports 1..3
    try:
        df = int(days_from)
    except Exception:
        df = 3
    if df < 1:
        df = 1
    if df > 3:
        df = 3

    url = f"{ODDS_API_HOST}/v4/sports/{sport_key}/scores/"

    params = {
        "apiKey": api_key,
        "daysFrom": df,
        "dateFormat": "iso",
    }

    # light retry for transient issues
    for attempt in range(3):
        try:
            return _odds_api_get(url, params=params)
        except requests.exceptions.HTTPError as e:
            # if it’s not 429, just bail
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
