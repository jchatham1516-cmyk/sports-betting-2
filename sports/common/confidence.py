# sports/common/confidence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SportScales:
    """
    Typical edge scale per sport (tunable).
    """
    ml_edge_1sigma: float   # probability points
    ats_edge_1sigma: float  # spread points


SPORT_SCALES = {
    "nba": SportScales(ml_edge_1sigma=0.06, ats_edge_1sigma=3.0),
    "nfl": SportScales(ml_edge_1sigma=0.055, ats_edge_1sigma=2.5),
    "nhl": SportScales(ml_edge_1sigma=0.055, ats_edge_1sigma=0.75),
}


def compute_confidence_score(
    *,
    sport: str,
    edge_home: Optional[float],
    spread_edge_home: Optional[float],
) -> float:
    """
    Returns a 0..1-ish confidence score. Higher = more confident.
    Uses the *stronger* of ML/ATS signals after normalization.
    """
    scales = SPORT_SCALES.get(sport, SPORT_SCALES["nba"])

    ml = abs(edge_home) if edge_home is not None else 0.0
    ats = abs(spread_edge_home) if spread_edge_home is not None else 0.0

    # normalize edges into "sigmas"
    ml_z = ml / max(scales.ml_edge_1sigma, 1e-9)
    ats_z = ats / max(scales.ats_edge_1sigma, 1e-9)

    z = max(ml_z, ats_z)

    # map z → score (soft saturation)
    # z=0 => 0, z~1 => 0.5, z~2 => 0.75, z~3 => 0.86
    score = z / (z + 1.0)
    return float(max(0.0, min(1.0, score)))


def confidence_bucket(score: float) -> str:
    if score >= 0.72:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"


def value_tier_from_edges(*, edge_home: float, spread_edge_home: float) -> str:
    """
    Rough value tier, consistent across sports.
    """
    ml = abs(edge_home)
    ats = abs(spread_edge_home)

    if max(ml, ats) >= 0.12 or ats >= 5.0:
        return "HIGH VALUE"
    if max(ml, ats) >= 0.07 or ats >= 3.0:
        return "MEDIUM VALUE"
    return "LOW VALUE"
