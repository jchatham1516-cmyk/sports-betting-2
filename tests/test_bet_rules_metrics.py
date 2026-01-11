import pandas as pd

from sports.common.bet_rules import (
    breakeven_prob_from_american,
    decide_bet_from_row,
    implied_prob_american,
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
    metrics = primary_metrics_for_row(row)
    confidence = metrics[5]
    reason = metrics[6]

    assert confidence == "LOW"
    assert "LONGSHOT_ODDS>=+500" in reason
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
    metrics = primary_metrics_for_row(row)
    assert metrics[5] == "MEDIUM"
    assert "DISAGREE_PROB>0.20" in metrics[6]


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
    metrics = primary_metrics_for_row(row)
    assert metrics[2] == 0.6
    assert metrics[3] == 0.5
    assert round(metrics[4], 4) == 0.1


def test_value_tier_from_abs_edge_prob():
    assert value_tier_from_edge(0.065) == "HIGH VALUE"
    assert value_tier_from_edge(0.04) == "MED VALUE"
    assert value_tier_from_edge(0.02) == "LOW VALUE"
    assert value_tier_from_edge(0.01) == "NO BET"


def test_longshot_extreme_requires_edge():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML (strong)",
            "home_ml": 750,
            "away_ml": -1200,
            "model_home_prob": 0.18,
            "market_home_prob": 0.12,
        }
    )
    decision = decide_bet_from_row(row, unit_dollars=25.0)
    assert decision.play_pass == "PASS"
    assert "LONGSHOT_CAP" in decision.decision_flags


def test_longshot_cap_units():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML (strong)",
            "home_ml": 600,
            "away_ml": -800,
            "model_home_prob": 0.35,
            "market_home_prob": 0.2,
        }
    )
    decision = decide_bet_from_row(row, unit_dollars=25.0)
    assert decision.play_pass == "PLAY"
    assert decision.units == 0.25
    assert "LONGSHOT_CAP" in decision.decision_flags
