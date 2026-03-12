from __future__ import annotations

from sports_betting.models.types import GameContext


def validate_game(game: GameContext) -> list[str]:
    flags: list[str] = []
    if not game.markets:
        flags.append("missing_markets")
    if game.home_team == game.away_team:
        flags.append("invalid_matchup")
    return flags
