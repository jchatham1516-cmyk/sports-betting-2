from __future__ import annotations

from sports_betting.models.types import MarketPrediction
from sports_betting.sports.common.staking import StakingConfig, recommend_units


def apply_selection(
    prediction: MarketPrediction,
    thresholds: dict[str, float],
    decimal_odds: float,
    staking: StakingConfig,
) -> MarketPrediction:
    if prediction.edge < thresholds["min_edge"]:
        prediction.decision = "pass"
        prediction.flags.append("edge_below_threshold")
        return prediction
    if prediction.expected_value < thresholds["min_ev"]:
        prediction.decision = "pass"
        prediction.flags.append("ev_below_threshold")
        return prediction
    if prediction.confidence < thresholds["min_confidence"]:
        prediction.decision = "pass"
        prediction.flags.append("confidence_below_threshold")
        return prediction

    prediction.recommended_units = recommend_units(
        prediction.model_probability, decimal_odds, prediction.confidence, staking
    )
    if prediction.recommended_units <= 0:
        prediction.decision = "pass"
        prediction.flags.append("stake_below_minimum")
    else:
        prediction.decision = "bet"
    return prediction


def rank_predictions(predictions: list[MarketPrediction]) -> list[MarketPrediction]:
    return sorted(
        predictions,
        key=lambda p: ((p.expected_value * 0.45) + (p.edge * 0.3) + (p.confidence * 0.15) + (p.model_quality * 0.1)),
        reverse=True,
    )
