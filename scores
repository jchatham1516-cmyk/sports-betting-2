import json
from pathlib import Path

def fetch_scores_from_odds_api(
    sport_key: str,
    *,
    days_from: int = 3,         # 1..3
    date_format: str = "iso",   # "iso" or "unix"
) -> list[dict]:
    api_key = get_odds_api_key()
    url = f"{ODDS_API_BASE_URL}/sports/{sport_key}/scores/"
    params = {
        "apiKey": api_key,
        "daysFrom": int(days_from),
        "dateFormat": date_format,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()
