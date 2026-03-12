from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sports_betting.models.types import GameContext, MarketOdds
from sports_betting.sports.common.odds import american_to_decimal


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_games(path: Path, sport: str) -> list[GameContext]:
    payload = load_json(path)
    games: list[GameContext] = []
    for row in payload.get("games", []):
        markets = [
            MarketOdds(
                market=m["market"],
                side=m["side"],
                line=m.get("line"),
                american_odds=m["odds"],
                decimal_odds=american_to_decimal(m["odds"]),
                sportsbook=m.get("sportsbook", "consensus"),
            )
            for m in row.get("markets", [])
        ]
        games.append(
            GameContext(
                event_id=row["event_id"],
                date=datetime.fromisoformat(row["date"]),
                sport=sport,
                home_team=row["home_team"],
                away_team=row["away_team"],
                markets=markets,
                meta=row.get("meta", {}),
            )
        )
    return games
