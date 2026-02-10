import pandas as pd

from sports.common import bet_rules
from sports.common.bet_rules import decide_bet_from_row


def _row(**kwargs):
    base = {
        "primary_recommendation": "Model PICK: HOME ML",
        "home_ml": -110,
        "away_ml": 100,
        "model_home_prob": 0.60,
        "market_home_prob": 0.50,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_uncalibrated_sizes_down_not_pass(monkeypatch):
    monkeypatch.setattr(bet_rules, "load_calibrator", lambda *args, **kwargs: None)
    bet_rules._UNCERTAINTY_CACHE.clear()
    d = decide_bet_from_row(_row(), unit_dollars=20.0, sport="nba")
    assert d.play_pass == "PLAY"
    assert d.calibration_multiplier == 0.55
    assert d.final_units > 0


def test_uncertainty_reduces_units_via_multiplier(monkeypatch):
    monkeypatch.setattr(bet_rules, "load_calibrator", lambda *args, **kwargs: {"a": 1.0, "b": 0.0, "n_samples": 500})
    monkeypatch.setattr(bet_rules, "load_uncertainty", lambda *_args, **_kwargs: {"uncertainty": 0.01, "n": 300})
    bet_rules._UNCERTAINTY_CACHE.clear()
    low = decide_bet_from_row(_row(), unit_dollars=20.0, sport="nba")

    monkeypatch.setattr(bet_rules, "load_uncertainty", lambda *_args, **_kwargs: {"uncertainty": 0.10, "n": 20})
    bet_rules._UNCERTAINTY_CACHE.clear()
    high = decide_bet_from_row(_row(), unit_dollars=20.0, sport="nba")

    assert high.final_units < low.final_units
    assert high.edge_prob_final == low.edge_prob_final


def test_nhl_goalie_multiplier_and_both_unknown_pass(monkeypatch):
    monkeypatch.setattr(bet_rules, "load_calibrator", lambda *args, **kwargs: {"a": 1.0, "b": 0.0, "n_samples": 500})
    projected = decide_bet_from_row(
        _row(goalie_home_status="PROJECTED", goalie_away_status="CONFIRMED"),
        unit_dollars=20.0,
        sport="nhl",
    )
    assert projected.play_pass == "PLAY"
    assert projected.goalie_multiplier in (0.65, 1.0)

    both_unknown = decide_bet_from_row(
        _row(goalie_home_status="UNKNOWN", goalie_away_status="UNKNOWN"),
        unit_dollars=20.0,
        sport="nhl",
    )
    assert both_unknown.play_pass == "PASS"
    assert "GOALIE_BOTH_UNKNOWN_PASS" in both_unknown.decision_reason


def test_odds_caps_and_longshot_rules():
    out_of_range = decide_bet_from_row(
        _row(home_ml=260, away_ml=-300, model_home_prob=0.7, market_home_prob=0.5),
        unit_dollars=20.0,
        sport="nba",
    )
    assert out_of_range.play_pass == "PASS"

    longshot_allowed = decide_bet_from_row(
        _row(home_ml=190, away_ml=-210, model_home_prob=0.66, market_home_prob=0.5),
        unit_dollars=20.0,
        sport="nba",
    )
    assert longshot_allowed.play_pass == "PLAY"
