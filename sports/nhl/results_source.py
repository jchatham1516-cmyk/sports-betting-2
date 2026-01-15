from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict

import requests


NHL_SCHEDULE_URL = "https://statsapi.web.nhl.com/api/v1/schedule"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def fetch_nhl_completed_games(days_from: int) -> List[Dict]:
    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=int(days_from or 0))
    params = {
        "sportId": 1,
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": today.strftime("%Y-%m-%d"),
    }
    try:
        resp = requests.get(
            NHL_SCHEDULE_URL,
            params=params,
            timeout=20,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        )
        if resp.status_code != 200:
            print(f"[nhl results] WARNING: schedule status={resp.status_code} html_len={len(resp.text or '')}")
            return []
        payload = resp.json() or {}
    except Exception as exc:
        print(f"[nhl results] WARNING: failed to fetch schedule: {exc}")
        return []

    results: List[Dict] = []
    for day in payload.get("dates") or []:
        date_str = day.get("date")
        for game in day.get("games") or []:
            status = (game.get("status") or {}).get("abstractGameState") or ""
            detailed = (game.get("status") or {}).get("detailedState") or ""
            if "final" not in str(status).lower() and "final" not in str(detailed).lower():
                continue
            teams = game.get("teams") or {}
            home_team = (teams.get("home") or {}).get("team", {}).get("name")
            away_team = (teams.get("away") or {}).get("team", {}).get("name")
            home_score = (teams.get("home") or {}).get("score")
            away_score = (teams.get("away") or {}).get("score")
            if home_team is None or away_team is None:
                continue
            if home_score is None or away_score is None:
                continue
            game_date = date_str
            if not game_date:
                try:
                    game_date = datetime.fromisoformat(
                        str(game.get("gameDate") or "").replace("Z", "+00:00")
                    ).date().strftime("%Y-%m-%d")
                except Exception:
                    game_date = ""
            results.append(
                {
                    "home_team": str(home_team),
                    "away_team": str(away_team),
                    "home_score": float(home_score),
                    "away_score": float(away_score),
                    "date": str(game_date or ""),
                }
            )
    return results
