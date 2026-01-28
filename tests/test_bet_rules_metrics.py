import pandas as pd

from sports.common.bet_config import get_sport_bet_config
from sports.common import bet_rules
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
    confidence = metrics[9]
    reason = metrics[10]

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
    assert metrics[9] == "LOW"
    assert "DISAGREE_CAP" in metrics[10]


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
    assert metrics[5] == 0.5
    assert round(metrics[6], 4) == 0.1


def test_value_tier_from_abs_edge_prob():
    assert value_tier_from_edge(0.065, 0.03) == "HIGH VALUE"
    assert value_tier_from_edge(0.04, 0.03) == "MED VALUE"
    assert value_tier_from_edge(0.02, 0.015) == "LOW VALUE"
    assert value_tier_from_edge(0.01, 0.03) == "NO BET"


def test_confidence_from_edge_prob():
    assert confidence_tier_from_edge(0.06, 0.03) == "HIGH"
    assert confidence_tier_from_edge(0.04, 0.03) == "MEDIUM"
    assert confidence_tier_from_edge(0.01, 0.03) == "LOW"


def test_nhl_uncertainty_scaling_reduces_min_edge(monkeypatch):
    config = get_sport_bet_config("nhl")
    monkeypatch.setenv("NHL_DYNAMIC_MIN_EDGE_CAP", "0.2")
    sample = {"uncertainty": 0.12, "n": 20}

    def _mock_uncertainty(_sport, market=None):
        return dict(sample)

    monkeypatch.setattr(bet_rules, "load_uncertainty", _mock_uncertainty)
    bet_rules._UNCERTAINTY_CACHE.clear()
    min_edge_small = bet_rules._dynamic_min_edge("nhl", config.min_edge_cal, config)

    sample["n"] = 60
    bet_rules._UNCERTAINTY_CACHE.clear()
    min_edge_large = bet_rules._dynamic_min_edge("nhl", config.min_edge_cal, config)

    assert min_edge_small > min_edge_large


def test_goalie_unconfirmed_flag_added_to_decisions():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob": 0.55,
            "market_home_prob": 0.5,
            "model_home_prob_final": 0.55,
            "goalie_status": "PARTIAL",
            "goalie_home_status": "PROJECTED",
            "goalie_away_status": "CONFIRMED",
        }
    )
    metrics = primary_metrics_for_row(row, sport="nhl")
    flags = metrics[15]
    assert "GOALIE_UNCONFIRMED" in flags


def test_nhl_min_edge_cap_applied(monkeypatch):
    monkeypatch.setenv("NHL_MIN_EDGE_CAP", "0.02")
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob": 0.55,
            "model_home_prob_final": 0.55,
            "market_home_prob": 0.5,
            "goalie_confirmed": True,
            "goalie_status": "OK",
            "goalie_home_status": "CONFIRMED",
            "goalie_away_status": "CONFIRMED",
        }
    )
    metrics = primary_metrics_for_row(row, sport="nhl")
    assert metrics[17] <= 0.02
    assert metrics[18] <= 0.02


def test_nhl_min_edge_cap(monkeypatch):
    monkeypatch.setenv("NHL_MIN_EDGE_CAP", "0.055")
    monkeypatch.setenv("NHL_DYNAMIC_MIN_EDGE_CAP", "1.0")
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob": 0.6,
            "model_home_prob_final": 0.6,
            "market_home_prob": 0.5,
            "goalie_confirmed": True,
            "goalie_status": "OK",
            "goalie_home_status": "CONFIRMED",
            "goalie_away_status": "CONFIRMED",
        }
    )
    metrics = primary_metrics_for_row(row, sport="nhl")
    assert metrics[17] <= 0.055
    assert metrics[18] <= 0.055


def test_low_unc_reduces_anchor(monkeypatch):
    config = get_sport_bet_config("nhl")
    monkeypatch.setenv("NHL_ANCHOR_W", "0.50")
    monkeypatch.setenv("NHL_ANCHOR_W_UNCONFIRMED", "0.50")
    monkeypatch.setenv("NHL_LOW_UNC_THRESHOLD", "0.01")
    monkeypatch.setenv("NHL_LOW_UNC_ANCHOR_MULT", "0.85")

    def _low_unc(_config, *, use_floor, row=None, market=None):
        return 0.005, None, 0.005

    def _high_unc(_config, *, use_floor, row=None, market=None):
        return 0.02, None, 0.02

    row = pd.Series({"goalie_confirmed": True})

    monkeypatch.setattr(bet_rules, "_nhl_uncertainty_context", _low_unc)
    low_anchor = bet_rules.nhl_anchor_context_for_row(row, config).anchor_weight

    monkeypatch.setattr(bet_rules, "_nhl_uncertainty_context", _high_unc)
    high_anchor = bet_rules.nhl_anchor_context_for_row(row, config).anchor_weight

    assert low_anchor < high_anchor


def test_nhl_low_unc_anchor_and_min_edge_cap(monkeypatch):
    config = get_sport_bet_config("nhl")
    monkeypatch.setenv("NHL_MIN_EDGE_CAP", "0.04")
    monkeypatch.setenv("NHL_ANCHOR_W", "0.50")
    monkeypatch.setenv("NHL_ANCHOR_W_UNCONFIRMED", "0.50")
    monkeypatch.setenv("NHL_LOW_UNC_THRESHOLD", "0.01")
    monkeypatch.setenv("NHL_LOW_UNC_ANCHOR_MULT", "0.80")

    def _low_unc(_config, *, use_floor, row=None, market=None):
        return 0.005, None, 0.005

    def _high_unc(_config, *, use_floor, row=None, market=None):
        return 0.02, None, 0.02

    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob": 0.6,
            "model_home_prob_final": 0.6,
            "market_home_prob": 0.5,
            "goalie_status": "OK",
            "goalie_home_status": "CONFIRMED",
            "goalie_away_status": "CONFIRMED",
            "goalie_confirmed": True,
        }
    )

    monkeypatch.setattr(bet_rules, "_nhl_uncertainty_context", _high_unc)
    high_anchor = bet_rules.nhl_anchor_context_for_row(row, config).anchor_weight

    monkeypatch.setattr(bet_rules, "_nhl_uncertainty_context", _low_unc)
    low_anchor = bet_rules.nhl_anchor_context_for_row(row, config).anchor_weight
    metrics = primary_metrics_for_row(row, sport="nhl")

    assert low_anchor < high_anchor
    assert metrics[17] <= 0.04 + 1e-6
    assert metrics[18] <= 0.04 + 1e-6


def test_goalie_unconfirmed_single_penalty(monkeypatch):
    monkeypatch.setenv("NHL_GOALIE_UNCONF_EDGE_ADD", "0.01")
    monkeypatch.setenv("NHL_MIN_EDGE_CAP", "1.0")
    monkeypatch.setenv("NHL_ANCHOR_W", "0.50")
    monkeypatch.setenv("NHL_ANCHOR_W_UNCONFIRMED", "0.50")
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob": 0.6,
            "model_home_prob_final": 0.6,
            "market_home_prob": 0.5,
            "goalie_status": "PROJECTED",
            "goalie_home_status": "PROJECTED",
            "goalie_away_status": "CONFIRMED",
        }
    )
    confirmed = row.copy()
    confirmed["goalie_confirmed"] = True
    unconfirmed = row.copy()
    unconfirmed["goalie_confirmed"] = False

    metrics_confirmed = primary_metrics_for_row(confirmed, sport="nhl")
    metrics_unconfirmed = primary_metrics_for_row(unconfirmed, sport="nhl")
    assert "GOALIE_UNCONFIRMED" in metrics_unconfirmed[15]
    diff = float(metrics_unconfirmed[17]) - float(metrics_confirmed[17])
    assert diff >= 0.01


def test_primary_metrics_safe_for_minimal_nhl_ml_row():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML",
            "home_ml": -120,
            "away_ml": 110,
            "model_home_prob": 0.55,
            "market_home_prob": 0.5,
        }
    )
    metrics = primary_metrics_for_row(row, sport="nhl")
    assert str(metrics[11]).strip()
    assert isinstance(metrics[12], bool)


def test_primary_metrics_safe_for_missing_ats_fields():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ATS",
            "primary_market": "ATS",
            "primary_side": "HOME",
        }
    )
    metrics = primary_metrics_for_row(row, sport="nba")
    assert str(metrics[11]).strip()
    assert isinstance(metrics[12], bool)


def test_primary_metrics_safe_when_value_tier_inputs_missing():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: AWAY ML",
            "home_ml": -115,
            "away_ml": 105,
            "model_home_prob": 0.48,
            "market_home_prob": 0.52,
        }
    )
    metrics = primary_metrics_for_row(row, sport="nba")
    assert str(metrics[11]).strip()
    assert isinstance(metrics[12], bool)


def test_primary_metrics_handles_uncalibrated_ats():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK ATS: HOME",
            "primary_market": "ATS",
            "primary_side": "HOME",
            "home_spread": -4.5,
            "spread_price": -110,
            "ats_home_cover_prob": 0.58,
            "market_spread_prob": 0.52,
            "decision_flags": "ATS_UNCALIBRATED_MARGIN",
        }
    )
    metrics = primary_metrics_for_row(row, sport="nba")
    flags = metrics[15]
    assert "ATS_UNCALIBRATED" in flags
    assert "UNCALIBRATED_FALLBACK" in flags
