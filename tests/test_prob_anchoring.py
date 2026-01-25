import pandas as pd
import pytest

from sports.common.bet_rules import ml_probabilities_for_row, primary_metrics_for_row


def test_anchor_weight_applied():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML (strong)",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob_raw": 0.7,
            "market_home_prob": 0.5,
        }
    )

    ml_probs = ml_probabilities_for_row(row, sport="nba")
    assert ml_probs["model_home_prob_final"] == pytest.approx(0.59, abs=1e-3)


def test_underdog_cap_applied():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML (strong)",
            "home_ml": 300,
            "away_ml": -350,
            "model_home_prob_raw": 0.8,
            "market_home_prob": 0.2,
        }
    )

    metrics = primary_metrics_for_row(row, sport="nba")
    p_final = metrics[4]
    flags = metrics[15]

    assert p_final <= 0.3000001
    assert "UNDERDOG_CAP" in flags
