# sports/common/scores_sources.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import requests

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"


def get_odds_api_key() -> str:
    k = os.environ.get("ODDS_API_KEY", "").strip()
    if not k:
        raise RuntimeError("ODDS_API_KEY environment variable is not set.")
    return k


def fetch_recent_scores(
    *,
    sport_key: str,
    days_from: int = 7,
    date_format: str = "iso",
) -> List[Dict[str, Any]]:
    """
    The Odds API scores endpoint returns recent games + final scores (for supported sports).
    daysFrom must be >= 1. (See Odds API error code INVALID_SCORES_DAYS_FROM.) :contentReference[oaicite:0]{index=0}
    """
    api_key = get_odds_api_key()
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/scores"
    params = {
        "apiKey": api_key,
        "daysFrom": int(days_from),
        "dateFormat": date_format,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json() or []
