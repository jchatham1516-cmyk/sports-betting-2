import pandas as pd

from sports.common.bet_rules import (
    breakeven_prob_from_american,
    implied_prob_american,
    confidence_tier_from_edge,
    primary_metrics_for_row,
    value_tier_from_edge,
)


def test_breakeven_prob_from_american():
    assert breakeven_prob_from_american(100) == 0.5
    assert round(breakeven_prob_from_american(-200), 4) == 0.6667


def test_implied_prob_american():
    assert implied_prob_american(100) == 0.5
    assert round(implied_prob_american(-200), 4) == 0.6667


def test_confidence_with_penalties():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML (strong)",
            "home_ml": 600,
            "away_ml": -800,
            "model_home_prob": 0.45,
            "market_home_prob": 0.3,
            "injury_confidence": "low",
        }
    )
    metrics = primary_metrics_for_row(row, sport="nba")
    confidence = metrics[7]
    reason = metrics[8]

    assert confidence == "LOW"
    assert "ML_CONFIDENCE_CAP>=400" in reason
    assert "INJURY_CONF_LOW" in reason


def test_confidence_disagreement_penalty():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML (strong)",
            "home_ml": -150,
            "away_ml": 130,
            "model_home_prob": 0.8,
            "market_home_prob": 0.5,
        }
    )
    metrics = primary_metrics_for_row(row, sport="nba")
    assert metrics[7] == "LOW"
    assert "DISAGREE_CAP" in metrics[8]


def test_abs_edge_flips_for_away():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: AWAY ML (strong)",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob": 0.4,
            "market_home_prob": 0.5,
        }
    )
    metrics = primary_metrics_for_row(row, sport="nba")
    assert metrics[2] == 0.6
    assert metrics[4] == 0.5
    assert round(metrics[5], 4) == 0.1


def test_value_tier_from_abs_edge_prob():
    assert value_tier_from_edge(0.065, 0.03) == "HIGH VALUE"
    assert value_tier_from_edge(0.04, 0.03) == "MED VALUE"
    assert value_tier_from_edge(0.02, 0.015) == "LOW VALUE"
    assert value_tier_from_edge(0.01, 0.03) == "NO BET"


def test_confidence_from_edge_prob():
    assert confidence_tier_from_edge(0.06, 0.03) == "HIGH"
    assert confidence_tier_from_edge(0.04, 0.03) == "MEDIUM"
    assert confidence_tier_from_edge(0.01, 0.03) == "LOW"
