from sports_betting.models.types import MarketPrediction
from sports_betting.sports.common.ev import edge, expected_value
from sports_betting.sports.common.odds import (
    american_to_decimal,
    american_to_implied_probability,
    remove_vig_two_way,
)
from sports_betting.sports.common.selection import apply_selection
from sports_betting.sports.common.staking import StakingConfig


def test_odds_conversions():
    assert round(american_to_decimal(-110), 4) == 1.9091
    assert round(american_to_decimal(150), 2) == 2.5
    p = american_to_implied_probability(-110)
    assert 0.52 < p < 0.53


def test_no_vig_two_way():
    p1, p2 = remove_vig_two_way(0.55, 0.52)
    assert round(p1 + p2, 8) == 1.0


def test_ev_and_edge():
    ev = expected_value(0.56, 1.91)
    assert ev > 0
    assert edge(0.56, 0.5238) > 0


def test_staking_selection_thresholds():
    pred = MarketPrediction(
        event_id="1",
        date="2026-01-15",
        sport="NBA",
        game="A @ B",
        market="moneyline",
        side="B",
        line=None,
        sportsbook_odds=-110,
        model_probability=0.57,
        market_probability=0.5238,
        edge=0.0462,
        expected_value=0.0887,
        confidence=0.7,
        model_quality=0.7,
        explanation=["x"],
        flags=[],
    )
    cfg = StakingConfig()
    out = apply_selection(pred, {"min_edge": 0.02, "min_ev": 0.01, "min_confidence": 0.55}, 1.91, cfg)
    assert out.decision == "bet"
    assert out.recommended_units > 0
