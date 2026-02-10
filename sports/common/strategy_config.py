from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from typing import Any, Dict


DEFAULT_STRATEGY_CONFIG: Dict[str, Any] = {
    "staking": {
        "kelly_fraction": 0.15,
        "max_units_per_bet": 1.0,
        "min_units_floor": 0.0,
    },
    "markets": {
        "ML": {
            "nba": {
                "min_edge": 0.060,
                "odds_min": -220,
                "odds_max": 200,
                "longshot_max": 240,
                "longshot_min_edge": 0.10,
                "longshot_requires_calibrated": True,
            },
            "nhl": {
                "min_edge": 0.055,
                "odds_min": -200,
                "odds_max": 180,
                "longshot_max": 220,
                "longshot_min_edge": 0.09,
                "longshot_requires_goalie_confirmed": True,
            },
            "nfl": {
                "min_edge": 0.060,
                "odds_min": -240,
                "odds_max": 210,
                "longshot_max": 240,
                "longshot_min_edge": 0.10,
                "longshot_requires_calibrated": True,
            },
        },
        "ATS": {
            "all": {
                "min_edge": 0.035,
                "require_margin_calibrated": True,
            }
        },
        "TOTAL": {
            "all": {
                "min_edge": 0.030,
                "require_sanity_pass": True,
            }
        },
    },
    "multipliers": {
        "calibration": {"calibrated": 1.0, "uncalibrated": 0.55},
        "uncertainty": {"k": 7.0, "floor": 0.25},
        "goalie": {"confirmed": 1.0, "projected": 0.65, "unknown": 0.40, "pass_if_both_unknown": True},
        "injury": {"clean": 1.0, "partial": 0.80, "bad": 0.60},
        "disagreement": {"pass_threshold": 0.22, "require_calibrated": True},
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _strategy_path() -> str:
    return os.getenv("STRATEGY_CONFIG_PATH", "results/strategy_config.json")


@lru_cache(maxsize=1)
def load_strategy_config() -> Dict[str, Any]:
    path = _strategy_path()
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULT_STRATEGY_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    except Exception:
        return copy.deepcopy(DEFAULT_STRATEGY_CONFIG)
    if not isinstance(user_cfg, dict):
        return copy.deepcopy(DEFAULT_STRATEGY_CONFIG)
    return _deep_merge(DEFAULT_STRATEGY_CONFIG, user_cfg)


def get_market_config(cfg: Dict[str, Any], market: str, sport: str) -> Dict[str, Any]:
    market_cfg = ((cfg.get("markets") or {}).get(str(market).upper()) or {})
    return market_cfg.get(str(sport).lower()) or market_cfg.get("all") or {}


def reload_strategy_config() -> None:
    load_strategy_config.cache_clear()
