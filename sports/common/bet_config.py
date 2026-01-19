from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SportBetConfig:
    min_edge_cal: float
    longshot_odds: float = 400.0
    longshot_cap_units: float = 0.25
    disagree_cap_edge: float = 0.20
    disagree_cap_units: float = 0.25
    max_units: float = 1.0
    anchor_weight: float = 0.60
    underdog_cap_med_prob: float = 0.25
    underdog_cap_low_prob: float = 0.15
    underdog_cap_med_add: float = 0.10
    underdog_cap_low_add: float = 0.07
    disagree_pass_edge: float = 0.25
    disagree_pass_min_edge: float = 0.08
    disagree_pass_max_units: float = 0.10


SPORT_BET_CONFIGS = {
    "nba": SportBetConfig(
        min_edge_cal=float(os.getenv("NBA_MIN_EDGE_CAL", "0.045")),
        longshot_odds=400.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=1.0,
        anchor_weight=float(os.getenv("NBA_ANCHOR_WEIGHT", "0.55")),
    ),
    "nfl": SportBetConfig(
        min_edge_cal=0.04,
        longshot_odds=400.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=0.5,
        anchor_weight=0.60,
    ),
    "nhl": SportBetConfig(
        min_edge_cal=float(os.getenv("NHL_MIN_EDGE_CAL", "0.04")),
        longshot_odds=250.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=float(os.getenv("NHL_MAX_UNITS", "0.5")),
        anchor_weight=0.75,
    ),
}


def get_sport_bet_config(sport: str) -> SportBetConfig:
    return SPORT_BET_CONFIGS.get(str(sport).lower(), SPORT_BET_CONFIGS["nba"])
