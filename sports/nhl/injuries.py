# sports/nhl/injuries.py
import requests
from typing import List
from sports.common.injuries import Injury, normalize_status

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

def fetch_nhl_injuries_from_odds_api(api_key: str) -> List[Injury]:
    url = f"{ODDS_API_BASE_URL}/sports/icehockey_nhl/injuries"
    r = requests.get(url, params={"apiKey": api_key, "regions": "us"})
    r.raise_for_status()
    data = r.json()

    out: List[Injury] = []
    for row in data:
        out.append(
            Injury(
                player=row.get("name", ""),
                team=row.get("team", ""),
                status=normalize_status(row.get("status", "")),
                detail=row.get("description") or row.get("injury"),
                source="odds_api",
                updated_at=row.get("updated_at"),
            )
        )
    return out
