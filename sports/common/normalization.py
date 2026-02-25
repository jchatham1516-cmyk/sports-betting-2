from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


TEAM_ALIASES = {
    "la clippers": "los angeles clippers",
    "los angeles clips": "los angeles clippers",
    "ny knicks": "new york knicks",
    "gsw": "golden state warriors",
    "new jersey devils": "new jersey devils",
    "ny rangers": "new york rangers",
    "arizona coyotes": "utah hockey club",
    "wsft": "washington commanders",
}


@dataclass(frozen=True)
class OddsPair:
    home_decimal: float
    away_decimal: float


def canonicalize_team_name(name: str) -> str:
    normalized = " ".join(name.strip().lower().split())
    return TEAM_ALIASES.get(normalized, normalized)


def american_to_decimal(american_odds: float) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero.")
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    return (100.0 / abs(american_odds)) + 1.0


def decimal_to_implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1.0 / decimal_odds


def remove_vig_two_way(home_decimal: float, away_decimal: float) -> tuple[float, float]:
    home_imp = decimal_to_implied_prob(home_decimal)
    away_imp = decimal_to_implied_prob(away_decimal)
    total = home_imp + away_imp
    if total <= 0:
        raise ValueError("Total implied probability must be positive.")
    return home_imp / total, away_imp / total


def bounded_probability(p: float, floor: float = 0.02, ceiling: float = 0.98) -> float:
    return float(np.clip(p, floor, ceiling))


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    if not vals:
        return default
    return float(np.mean(vals))
