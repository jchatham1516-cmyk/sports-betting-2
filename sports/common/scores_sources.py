# sports/common/scores_sources.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"


def get_odds_api_key() -> str:
    k = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not k:
        raise RuntimeError("ODDS_API_KEY environment variable is not set.")
    return k


def fetch_recent_scores(
    *,
    sport_key: str,
    days_from: int = 3,
    date_format: str = "iso",
    event_ids: Optional[list[str]] = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Calls:
      GET /v4/sports/{sport}/scores/?apiKey=...&daysFrom=1..3&dateFormat=iso

    Notes:
    - Odds API only supports daysFrom integers 1..3. If you pass higher, it 422s.
    - If daysFrom is omitted, it returns upcoming/live only (no finished scores).
    """
    api_key = get_odds_api_key()

    # clamp daysFrom to valid range (1..3)
    try:
        days_from_int = int(days_from)
    except Exception:
        days_from_int = 3

    if days_from_int < 1:
        days_from_int = 1
    if days_from_int > 3:
        days_from_int = 3

    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/scores/"
    params = {
        "apiKey": api_key,
        "dateFormat": date_format,
        "daysFrom": days_from_int,
    }
    if event_ids:
        params["eventIds"] = ",".join(event_ids)

    r = requests.get(url, params=params, timeout=timeout)

    # If anything weird happens, try a safe fallback (daysFrom=3), then try without daysFrom
    if r.status_code == 422:
        params["daysFrom"] = 3
        r = requests.get(url, params=params, timeout=timeout)

    if r.status_code == 422:
        params.pop("daysFrom", None)
        r = requests.get(url, params=params, timeout=timeout)

    r.raise_for_status()
    return r.json()
