from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SportBetConfig:
    min_edge_cal: float
    longshot_odds: float = 400.0
    longshot_cap_units: float = 0.25
    disagree_cap_edge: float = 0.20
    disagree_cap_units: float = 0.25
    max_units: float = 1.0


SPORT_BET_CONFIGS = {
    "nba": SportBetConfig(
        min_edge_cal=0.03,
        longshot_odds=400.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=1.0,
    ),
    "nfl": SportBetConfig(
        min_edge_cal=0.04,
        longshot_odds=400.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=0.5,
    ),
    "nhl": SportBetConfig(
        min_edge_cal=0.06,
        longshot_odds=400.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.18,
        disagree_cap_units=0.25,
        max_units=0.5,
    ),
}


def get_sport_bet_config(sport: str) -> SportBetConfig:
    return SPORT_BET_CONFIGS.get(str(sport).lower(), SPORT_BET_CONFIGS["nba"])
