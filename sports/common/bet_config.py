from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


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
    uncertainty_edge_cap_add: float = 0.04
    calibration_risk_multiplier: float = 0.55
    uncertainty_unit_scale: float = 6.0
    uncertainty_sample_ref: float = 120.0
    uncertainty_sample_exp: float = 0.5
    uncertainty_quality_mult: float = 0.6
    goalie_unconfirmed_units_mult: float = 0.65
    injury_low_units_mult: float = 0.7
    injury_unknown_units_mult: float = 0.85
    min_edge_soft_floor: float = 0.01
    soft_edge_base_units: float = 0.2
    test_bet_min_units_enabled: bool = False
    test_bet_min_units: float = 0.1
    parlay_min_edge: float = 0.045
    parlay_disagree_cap: float = 0.20
    parlay_disagree_huge_edge: float = 0.12
    nhl_parlay_min_edge: float | None = None
    parlay_min_pwin: float = 0.57
    parlay_max_disagreement: float = 0.12
    parlay_big_edge_override: float = 0.10
    parlay_reliability_weights: dict[str, float] = field(
        default_factory=lambda: {"ML": 1.0, "ATS": 0.7, "TOTAL": 0.6}
    )


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
        uncertainty_edge_cap_add=float(os.getenv("NBA_UNCERTAINTY_EDGE_CAP_ADD", "0.04")),
        calibration_risk_multiplier=float(os.getenv("NBA_CALIBRATION_RISK_MULT", "0.6")),
        uncertainty_unit_scale=float(os.getenv("NBA_UNCERTAINTY_UNIT_SCALE", "6.0")),
        uncertainty_sample_ref=float(os.getenv("NBA_UNCERTAINTY_SAMPLE_REF", "120")),
        uncertainty_sample_exp=float(os.getenv("NBA_UNCERTAINTY_SAMPLE_EXP", "0.5")),
        uncertainty_quality_mult=float(os.getenv("NBA_UNCERTAINTY_QUALITY_MULT", "0.6")),
        goalie_unconfirmed_units_mult=float(os.getenv("NBA_GOALIE_UNCONF_UNITS_MULT", "1.0")),
        injury_low_units_mult=float(os.getenv("NBA_INJURY_LOW_UNITS_MULT", "0.7")),
        injury_unknown_units_mult=float(os.getenv("NBA_INJURY_UNKNOWN_UNITS_MULT", "0.85")),
        min_edge_soft_floor=float(os.getenv("NBA_MIN_EDGE_SOFT_FLOOR", "0.012")),
        soft_edge_base_units=float(os.getenv("NBA_SOFT_EDGE_BASE_UNITS", "0.2")),
        test_bet_min_units_enabled=bool(int(os.getenv("NBA_TEST_BET_MIN_UNITS_ON", "0"))),
        test_bet_min_units=float(os.getenv("NBA_TEST_BET_MIN_UNITS", "0.1")),
        parlay_min_edge=float(os.getenv("NBA_PARLAY_MIN_EDGE", "0.06")),
        parlay_disagree_cap=float(os.getenv("NBA_PARLAY_DISAGREE_CAP", "0.20")),
        parlay_disagree_huge_edge=float(os.getenv("NBA_PARLAY_HUGE_EDGE", "0.12")),
        nhl_parlay_min_edge=None,
        parlay_min_pwin=float(os.getenv("NBA_PARLAY_MIN_PWIN", "0.57")),
        parlay_max_disagreement=float(os.getenv("NBA_PARLAY_MAX_DISAGREE", "0.12")),
        parlay_big_edge_override=float(os.getenv("NBA_PARLAY_BIG_EDGE", "0.10")),
        parlay_reliability_weights={"ML": 1.0, "ATS": 0.7, "TOTAL": 0.6},
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
        uncertainty_edge_cap_add=float(os.getenv("NFL_UNCERTAINTY_EDGE_CAP_ADD", "0.035")),
        calibration_risk_multiplier=float(os.getenv("NFL_CALIBRATION_RISK_MULT", "0.55")),
        uncertainty_unit_scale=float(os.getenv("NFL_UNCERTAINTY_UNIT_SCALE", "6.5")),
        uncertainty_sample_ref=float(os.getenv("NFL_UNCERTAINTY_SAMPLE_REF", "120")),
        uncertainty_sample_exp=float(os.getenv("NFL_UNCERTAINTY_SAMPLE_EXP", "0.5")),
        uncertainty_quality_mult=float(os.getenv("NFL_UNCERTAINTY_QUALITY_MULT", "0.6")),
        goalie_unconfirmed_units_mult=float(os.getenv("NFL_GOALIE_UNCONF_UNITS_MULT", "1.0")),
        injury_low_units_mult=float(os.getenv("NFL_INJURY_LOW_UNITS_MULT", "0.7")),
        injury_unknown_units_mult=float(os.getenv("NFL_INJURY_UNKNOWN_UNITS_MULT", "0.85")),
        min_edge_soft_floor=float(os.getenv("NFL_MIN_EDGE_SOFT_FLOOR", "0.01")),
        soft_edge_base_units=float(os.getenv("NFL_SOFT_EDGE_BASE_UNITS", "0.2")),
        test_bet_min_units_enabled=bool(int(os.getenv("NFL_TEST_BET_MIN_UNITS_ON", "0"))),
        test_bet_min_units=float(os.getenv("NFL_TEST_BET_MIN_UNITS", "0.1")),
        parlay_min_edge=float(os.getenv("NFL_PARLAY_MIN_EDGE", "0.05")),
        parlay_disagree_cap=float(os.getenv("NFL_PARLAY_DISAGREE_CAP", "0.20")),
        parlay_disagree_huge_edge=float(os.getenv("NFL_PARLAY_HUGE_EDGE", "0.12")),
        nhl_parlay_min_edge=None,
        parlay_min_pwin=float(os.getenv("NFL_PARLAY_MIN_PWIN", "0.57")),
        parlay_max_disagreement=float(os.getenv("NFL_PARLAY_MAX_DISAGREE", "0.12")),
        parlay_big_edge_override=float(os.getenv("NFL_PARLAY_BIG_EDGE", "0.10")),
        parlay_reliability_weights={"ML": 1.0, "ATS": 0.7, "TOTAL": 0.6},
    ),
    "nhl": SportBetConfig(
        min_edge_cal=float(os.getenv("NHL_MIN_EDGE_CAL", "0.04")),
        longshot_odds=250.0,
        longshot_cap_units=0.25,
        disagree_cap_edge=0.20,
        disagree_cap_units=0.25,
        max_units=float(os.getenv("NHL_MAX_UNITS", "0.5")),
        anchor_weight=float(os.getenv("NHL_ANCHOR_WEIGHT", "0.65")),
        uncertainty_edge_mult=float(os.getenv("NHL_UNCERTAINTY_EDGE_MULT", "0.75")),
        uncertainty_anchor_mult=float(os.getenv("NHL_UNCERTAINTY_ANCHOR_MULT", "0.25")),
        uncertainty_floor=float(os.getenv("NHL_UNCERTAINTY_FLOOR", "0.03")),
        uncalibrated_edge_add=float(os.getenv("NHL_UNCALIBRATED_EDGE_ADD", "0.02")),
        total_sd_min=float(os.getenv("NHL_TOTAL_SD_MIN", "0.9")),
        uncertainty_flat_threshold=float(os.getenv("NHL_UNCERTAINTY_FLAT_THRESHOLD", "0.06")),
        flat_units_when_uncertain=float(os.getenv("NHL_FLAT_UNITS_WHEN_UNCERTAIN", "0.5")),
        uncertainty_edge_cap_add=float(os.getenv("NHL_UNCERTAINTY_EDGE_CAP_ADD", "0.03")),
        calibration_risk_multiplier=float(os.getenv("NHL_CALIBRATION_RISK_MULT", "0.5")),
        uncertainty_unit_scale=float(os.getenv("NHL_UNCERTAINTY_UNIT_SCALE", "7.5")),
        uncertainty_sample_ref=float(os.getenv("NHL_UNCERTAINTY_SAMPLE_REF", "120")),
        uncertainty_sample_exp=float(os.getenv("NHL_UNCERTAINTY_SAMPLE_EXP", "0.5")),
        uncertainty_quality_mult=float(os.getenv("NHL_UNCERTAINTY_QUALITY_MULT", "0.7")),
        goalie_unconfirmed_units_mult=float(os.getenv("NHL_GOALIE_UNCONF_UNITS_MULT", "0.6")),
        injury_low_units_mult=float(os.getenv("NHL_INJURY_LOW_UNITS_MULT", "0.85")),
        injury_unknown_units_mult=float(os.getenv("NHL_INJURY_UNKNOWN_UNITS_MULT", "0.9")),
        min_edge_soft_floor=float(os.getenv("NHL_MIN_EDGE_SOFT_FLOOR", "0.012")),
        soft_edge_base_units=float(os.getenv("NHL_SOFT_EDGE_BASE_UNITS", "0.15")),
        test_bet_min_units_enabled=bool(int(os.getenv("NHL_TEST_BET_MIN_UNITS_ON", "0"))),
        test_bet_min_units=float(os.getenv("NHL_TEST_BET_MIN_UNITS", "0.1")),
        parlay_min_edge=float(os.getenv("NHL_PARLAY_MIN_EDGE", "0.05")),
        parlay_disagree_cap=float(os.getenv("NHL_PARLAY_DISAGREE_CAP", "0.18")),
        parlay_disagree_huge_edge=float(os.getenv("NHL_PARLAY_HUGE_EDGE", "0.1")),
        nhl_parlay_min_edge=float(os.getenv("NHL_PARLAY_MIN_EDGE", "0.05")),
        parlay_min_pwin=float(os.getenv("NHL_PARLAY_MIN_PWIN", "0.57")),
        parlay_max_disagreement=float(os.getenv("NHL_PARLAY_MAX_DISAGREE", "0.12")),
        parlay_big_edge_override=float(os.getenv("NHL_PARLAY_BIG_EDGE", "0.10")),
        parlay_reliability_weights={"ML": 1.0, "ATS": 0.7, "TOTAL": 0.6},
    ),
}


def _thresholds_path(sport: str) -> str:
    return os.getenv("BET_THRESHOLDS_PATH", f"results/thresholds_{str(sport).lower()}.json")


def _load_threshold_overrides(sport: str) -> dict[str, object]:
    path = _thresholds_path(sport)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _strategy_config_path() -> str:
    return os.getenv("STRATEGY_CONFIG_PATH", "results/strategy_config.json")


def _load_strategy_overrides(sport: str) -> dict[str, object]:
    path = _strategy_config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    sport_key = str(sport).lower()
    overrides: dict[str, object] = {}
    for key in ("default", "global", "all"):
        if isinstance(data.get(key), dict):
            overrides.update(data[key])
    if isinstance(data.get("sports"), dict) and isinstance(data["sports"].get(sport_key), dict):
        overrides.update(data["sports"][sport_key])
    if isinstance(data.get(sport_key), dict):
        overrides.update(data[sport_key])
    return overrides


def get_sport_bet_config(sport: str) -> SportBetConfig:
    base = SPORT_BET_CONFIGS.get(str(sport).lower(), SPORT_BET_CONFIGS["nba"])
    overrides = {}
    overrides.update(_load_threshold_overrides(sport))
    overrides.update(_load_strategy_overrides(sport))
    if not overrides:
        return base
    valid = {k: v for k, v in overrides.items() if hasattr(base, k)}
    return SportBetConfig(**{**base.__dict__, **valid})
