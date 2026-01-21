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
    uncertainty_edge_mult: float = 1.25
    uncertainty_anchor_mult: float = 0.60
    uncertainty_floor: float = 0.03
    uncalibrated_edge_add: float = 0.02
    total_sd_min: float = 8.0
    uncertainty_flat_threshold: float = 0.06
    flat_units_when_uncertain: float = 0.5
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
        uncertainty_edge_mult=float(os.getenv("NBA_UNCERTAINTY_EDGE_MULT", "1.25")),
        uncertainty_anchor_mult=float(os.getenv("NBA_UNCERTAINTY_ANCHOR_MULT", "0.60")),
        uncertainty_floor=float(os.getenv("NBA_UNCERTAINTY_FLOOR", "0.03")),
        uncalibrated_edge_add=float(os.getenv("NBA_UNCALIBRATED_EDGE_ADD", "0.02")),
        total_sd_min=float(os.getenv("NBA_TOTAL_SD_MIN", "8.0")),
        uncertainty_flat_threshold=float(os.getenv("NBA_UNCERTAINTY_FLAT_THRESHOLD", "0.06")),
        flat_units_when_uncertain=float(os.getenv("NBA_FLAT_UNITS_WHEN_UNCERTAIN", "0.5")),
    ),
    "nfl": SportBetConfig(
        min_edge_cal=0.04,
        longshot_odds=400.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=0.5,
        anchor_weight=0.60,
        uncertainty_edge_mult=float(os.getenv("NFL_UNCERTAINTY_EDGE_MULT", "1.10")),
        uncertainty_anchor_mult=float(os.getenv("NFL_UNCERTAINTY_ANCHOR_MULT", "0.50")),
        uncertainty_floor=float(os.getenv("NFL_UNCERTAINTY_FLOOR", "0.03")),
        uncalibrated_edge_add=float(os.getenv("NFL_UNCALIBRATED_EDGE_ADD", "0.02")),
        total_sd_min=float(os.getenv("NFL_TOTAL_SD_MIN", "6.0")),
        uncertainty_flat_threshold=float(os.getenv("NFL_UNCERTAINTY_FLAT_THRESHOLD", "0.06")),
        flat_units_when_uncertain=float(os.getenv("NFL_FLAT_UNITS_WHEN_UNCERTAIN", "0.5")),
    ),
    "nhl": SportBetConfig(
        min_edge_cal=float(os.getenv("NHL_MIN_EDGE_CAL", "0.04")),
        longshot_odds=250.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=float(os.getenv("NHL_MAX_UNITS", "0.5")),
        anchor_weight=0.75,
        uncertainty_edge_mult=float(os.getenv("NHL_UNCERTAINTY_EDGE_MULT", "1.10")),
        uncertainty_anchor_mult=float(os.getenv("NHL_UNCERTAINTY_ANCHOR_MULT", "0.50")),
        uncertainty_floor=float(os.getenv("NHL_UNCERTAINTY_FLOOR", "0.03")),
        uncalibrated_edge_add=float(os.getenv("NHL_UNCALIBRATED_EDGE_ADD", "0.02")),
        total_sd_min=float(os.getenv("NHL_TOTAL_SD_MIN", "0.9")),
        uncertainty_flat_threshold=float(os.getenv("NHL_UNCERTAINTY_FLAT_THRESHOLD", "0.06")),
        flat_units_when_uncertain=float(os.getenv("NHL_FLAT_UNITS_WHEN_UNCERTAIN", "0.5")),
    ),
}


def get_sport_bet_config(sport: str) -> SportBetConfig:
    return SPORT_BET_CONFIGS.get(str(sport).lower(), SPORT_BET_CONFIGS["nba"])
