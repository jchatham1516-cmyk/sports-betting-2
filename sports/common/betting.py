from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sports.common.ev import expected_value


@dataclass
class BettingConfig:
    bankroll: float = 100.0
    fractional_kelly: float = 0.25
    max_bet_units: float = 3.0
    min_bet_units: float = 0.1
    min_edge: float = 0.02
    min_ev: float = 0.01


def kelly_fraction(prob_win: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - prob_win
    if b <= 0:
        return 0.0
    f = (b * prob_win - q) / b
    return float(max(0.0, f))


def size_bet_units(
    prob_win: float,
    decimal_odds: float,
    edge: float,
    config: BettingConfig,
) -> tuple[float, str]:
    ev = expected_value(prob_win, decimal_odds)
    if edge < config.min_edge:
        return 0.0, "edge_below_threshold"
    if ev < config.min_ev:
        return 0.0, "ev_below_threshold"

    base_fraction = kelly_fraction(prob_win, decimal_odds)
    adjusted_fraction = base_fraction * config.fractional_kelly
    units = adjusted_fraction * config.bankroll
    units = float(np.clip(units, 0.0, config.max_bet_units))
    if units < config.min_bet_units:
        return 0.0, "bet_below_min_units"
    return round(units, 3), "play"
