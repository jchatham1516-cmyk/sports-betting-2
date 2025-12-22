# sports/common/scores_sources.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from sports.common.odds_sources import ODDS_API_BASE_URL, get_odds_api_key

# NOTE:
# - This module is intentionally "fail-soft".
# - If Odds API returns 401/403 (bad key / plan cap / disabled) or 429 (rate limit),
#   we return [] so your Elo/rest/form code can continue using existing state.


def fetch_recent_scores(
    *,
    sport_key: str,
    days_from: int = 3,
    date_format: str = "iso",
) -> List[Dict[str, Any]]:
    """
    Fetch finalized scores from The Odds API "scores" endpoint.

    Returns a list of events (dicts). On failure (401/403/429/network), returns [].

    Your models use this for:
      - Elo updates
      - rest map
      - recent form / totals pacing

    IMPORTANT:
      If this returns [], downstream code should just skip updates gracefully.
    """
    api_key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not api_key:
        # keep compatibility with your existing debug behavior
        try:
            api_key = get_odds_api_key()
        except Exception as e:
            print(f"[scores] WARNING: ODDS_API_KEY missing ({e}). Returning no scores.")
            return []

    try:
        days_from_int = int(days_from)
    except Exception:
        days_from_int = 3

    # Endpoint (Odds API v4):
    # /v4/sports/{sport_key}/scores/?apiKey=...&daysFrom=...&dateFormat=iso
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/scores/"
    params = {
        "apiKey": api_key,
        "daysFrom": max(1, min(days_from_int, 30)),  # clamp to a sane range
        "dateFormat": date_format,
    }

    try:
        r = requests.get(url, params=params, timeout=30)

        # Safe debug
        print("[scores_api DEBUG] status:", r.status_code)
        print("[scores_api DEBUG] url:", r.url)
        print("[scores_api DEBUG] remaining:", r.headers.get("x-requests-remaining"))
        print("[scores_api DEBUG] used:", r.headers.get("x-requests-used"))
        print("[scores_api DEBUG] last:", r.headers.get("x-requests-last"))

        # Fail-soft cases
        if r.status_code in (401, 403):
            # Unauthorized / forbidden (bad key, disabled key, or plan cap behavior)
            print(f"[scores] WARNING: unauthorized ({r.status_code}). Check ODDS_API_KEY / plan. Returning no scores.")
            return []

        if r.status_code == 429:
            print("[scores] WARNING: rate limited (429). Returning no scores.")
            return []

        r.raise_for_status()

        data = r.json()
        if not isinstance(data, list):
            return []

        # Optional: only keep items that look like score events
        # (don’t over-filter; your parsers handle missing fields)
        return data

    except Exception as e:
        print(f"[scores] WARNING: failed to fetch scores ({type(e).__name__}: {e}). Returning no scores.")
        return []
