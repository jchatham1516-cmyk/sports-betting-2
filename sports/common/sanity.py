# sports/common/sanity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SanityLimits:
    max_abs_model_spread: float
    max_abs_market_spread: float


LIMITS = {
    "nba": SanityLimits(max_abs_model_spread=15.0, max_abs_market_spread=20.0),
    "nfl": SanityLimits(max_abs_model_spread=17.0, max_abs_market_spread=24.0),
    "nhl": SanityLimits(max_abs_model_spread=3.5, max_abs_market_spread=4.0),
}


def sanity_check(
    *,
    sport: str,
    model_spread_home: Optional[float],
    market_spread_home: Optional[float],
) -> Tuple[bool, str]:
    """
    Returns (ok, reason).
    ok=False means "reject bet recommendation" (still keep row).
    """
    lim = LIMITS.get(sport, LIMITS["nba"])

    if model_spread_home is None:
        return True, ""

    if abs(float(model_spread_home)) > lim.max_abs_model_spread:
        return False, f"Reject: model spread absurd ({model_spread_home:.2f})"

    if market_spread_home is not None:
        try:
            ms = float(market_spread_home)
            if abs(ms) > lim.max_abs_market_spread:
                return False, f"Reject: market spread absurd ({ms:.2f})"
        except Exception:
            pass

    return True, ""
