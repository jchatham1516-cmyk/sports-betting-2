import pandas as pd

from sports.common.bet_rules import DecisionSettings, decide_bet_from_row, primary_metrics_for_row


def _base_row(**kwargs):
    base = {
        "primary_recommendation": "Model PICK: HOME ML (strong)",
        "value_tier": "HIGH VALUE",
        "confidence": "HIGH",
        "home_ml": 550,
        "away_ml": -800,
        "model_home_prob": 0.35,
        "market_home_prob": 0.2,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_longshot_allowed_but_capped():
    settings = DecisionSettings(max_units=1.0, longshot_max_units=0.25)
    decision = decide_bet_from_row(
        _base_row(),
        unit_dollars=40.0,
        settings=settings,
        max_abs_moneyline=400,
    )

    assert decision.play_pass == "PLAY"
    assert decision.final_units == 0.25
    assert "LONGSHOT_CAP" in decision.decision_flags


def test_disagreement_cap():
    row = _base_row(home_ml=150, away_ml=-170, model_home_prob=0.8, market_home_prob=0.5)
    settings = DecisionSettings(max_units=1.0, disagreement_max_units=0.25)

    decision = decide_bet_from_row(
        row,
        unit_dollars=40.0,
        settings=settings,
        max_abs_moneyline=None,
    )

    assert decision.play_pass == "PLAY"
    assert decision.final_units == 0.25
    assert "DISAGREE_CAP" in decision.decision_flags


def test_low_edge_passes():
    row = _base_row(model_home_prob=0.51, market_home_prob=0.5, home_ml=120, away_ml=-140)
    settings = DecisionSettings(min_play_edge_abs=0.02)
    decision = decide_bet_from_row(
        row,
        unit_dollars=40.0,
        settings=settings,
        max_abs_moneyline=None,
    )

    assert decision.play_pass == "PASS"
    assert decision.final_units == 0.0


def test_missing_odds_passes():
    row = _base_row(home_ml=None, away_ml=None)
    settings = DecisionSettings()
    decision = decide_bet_from_row(
        row,
        unit_dollars=40.0,
        settings=settings,
        max_abs_moneyline=None,
    )

    assert decision.play_pass == "PASS"
    assert "MISSING_DATA_PASS" in decision.decision_flags or decision.reason.startswith("missing moneyline")


def test_confidence_caps_for_moneyline_underdogs():
    row = _base_row(home_ml=300, away_ml=-350, model_home_prob=0.35, market_home_prob=0.2)
    metrics = primary_metrics_for_row(row)
    assert metrics[5] != "HIGH"

    row = _base_row(home_ml=450, away_ml=-500, model_home_prob=0.35, market_home_prob=0.2)
    metrics = primary_metrics_for_row(row)
    assert metrics[5] == "LOW"


def test_longshot_pass_threshold():
    row = _base_row(home_ml=650, away_ml=-800, model_home_prob=0.35, market_home_prob=0.3)
    settings = DecisionSettings(ml_pass_odds=600, ml_pass_min_edge=0.08)
    decision = decide_bet_from_row(
        row,
        unit_dollars=40.0,
        settings=settings,
        max_abs_moneyline=None,
    )

    assert decision.play_pass == "PASS"
