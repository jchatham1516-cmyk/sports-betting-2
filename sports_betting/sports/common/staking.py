from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StakingConfig:
    mode: str = "fractional_kelly"
    bankroll: float = 10000.0
    unit_size: float = 100.0
    fractional_kelly: float = 0.25
    max_units: float = 2.0
    min_units: float = 0.25
    uncertainty_penalty: float = 0.35


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    frac = ((b * probability) - (1 - probability)) / b
    return max(0.0, frac)


def recommend_units(
    probability: float,
    decimal_odds: float,
    confidence: float,
    config: StakingConfig,
) -> float:
    if config.mode == "flat":
        return config.min_units

    base = kelly_fraction(probability, decimal_odds) * config.fractional_kelly
    confidence_adj = max(0.2, confidence) ** (1 + config.uncertainty_penalty)
    units = (base * config.bankroll * confidence_adj) / config.unit_size
    units = min(config.max_units, max(0.0, units))
    return round(units if units >= config.min_units else 0.0, 2)
