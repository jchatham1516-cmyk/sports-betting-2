import pandas as pd

from sports.common.bet_rules import DecisionSettings, decide_bet_from_row, primary_metrics_for_row


def _base_row(**kwargs):
    base = {
        "primary_recommendation": "Model PICK: HOME ML (strong)",
        "home_ml": 550,
        "away_ml": -800,
        "model_home_prob": 0.35,
        "market_home_prob": 0.2,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_longshot_allowed_but_capped():
    settings = DecisionSettings()
    decision = decide_bet_from_row(
        _base_row(model_home_prob=0.8, market_home_prob=0.3),
        unit_dollars=40.0,
        sport="nba",
        settings=settings,
        max_abs_moneyline=400,
    )

    assert decision.play_pass == "PLAY"
    assert decision.final_units == 0.25
    assert "LONGSHOT_CAP" in decision.decision_flags


def test_disagreement_cap():
    row = _base_row(home_ml=150, away_ml=-170, model_home_prob=0.72, market_home_prob=0.5)
    settings = DecisionSettings()

    decision = decide_bet_from_row(
        row,
        unit_dollars=40.0,
        sport="nba",
        settings=settings,
        max_abs_moneyline=None,
    )

    assert decision.play_pass == "PLAY"
    assert decision.final_units == 0.25
    assert "DISAGREE_CAP" in decision.decision_flags


def test_disagreement_pass():
    row = _base_row(home_ml=150, away_ml=-170, model_home_prob=0.7, market_home_prob=0.1)
    decision = decide_bet_from_row(row, unit_dollars=40.0, sport="nba")

    assert decision.play_pass == "PASS"
    assert "LOW_EDGE_PASS" in decision.decision_flags or "DISAGREE_PASS" in decision.decision_flags


def test_low_edge_passes():
    row = _base_row(model_home_prob=0.51, market_home_prob=0.5, home_ml=120, away_ml=-140)
    settings = DecisionSettings()
    decision = decide_bet_from_row(
        row,
        unit_dollars=40.0,
        sport="nhl",
        settings=settings,
        max_abs_moneyline=None,
    )

    assert decision.play_pass == "PASS"
    assert decision.final_units == 0.0


def test_min_edge_allows_play_for_nba():
    row = _base_row(model_home_prob=0.75, market_home_prob=0.5, home_ml=120, away_ml=-140)
    decision = decide_bet_from_row(row, unit_dollars=40.0, sport="nba")
    assert decision.play_pass == "PLAY"


def test_nhl_unconfirmed_goalie_play_with_edge(monkeypatch):
    monkeypatch.setenv("NHL_MIN_EDGE_CAP", "0.055")
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK: HOME ML",
            "home_ml": -110,
            "away_ml": 100,
            "model_home_prob": 0.56,
            "model_home_prob_final": 0.56,
            "market_home_prob": 0.5,
            "goalie_confirmed": False,
            "goalie_status": "PROJECTED",
            "goalie_home_status": "PROJECTED",
            "goalie_away_status": "CONFIRMED",
        }
    )
    decision = decide_bet_from_row(row, unit_dollars=40.0, sport="nhl")

    assert decision.play_pass == "PLAY"
    assert "GOALIE_UNCONFIRMED" in decision.decision_flags


def test_missing_odds_passes():
    row = _base_row(home_ml=None, away_ml=None)
    settings = DecisionSettings()
    decision = decide_bet_from_row(
        row,
        unit_dollars=40.0,
        sport="nba",
        settings=settings,
        max_abs_moneyline=None,
    )

    assert decision.play_pass == "PASS"
    assert "MISSING_DATA_PASS" in decision.decision_flags or decision.reason.startswith("missing moneyline")


def test_confidence_caps_for_moneyline_underdogs():
    row = _base_row(home_ml=300, away_ml=-350, model_home_prob=0.35, market_home_prob=0.2)
    metrics = primary_metrics_for_row(row, sport="nba")
    assert metrics[9] != "HIGH"

    row = _base_row(home_ml=450, away_ml=-500, model_home_prob=0.35, market_home_prob=0.2)
    metrics = primary_metrics_for_row(row, sport="nba")
    assert metrics[9] == "LOW"


def test_ats_uncalibrated_margin_forces_pass():
    row = pd.Series(
        {
            "primary_recommendation": "Model PICK ATS: HOME",
            "primary_market": "ATS",
            "primary_side": "HOME",
            "home_spread": -5.5,
            "spread_price": -110,
            "ats_home_cover_prob": 0.56,
            "market_spread_prob": 0.5238,
            "decision_flags": "ATS_UNCALIBRATED_MARGIN",
        }
    )

    decision = decide_bet_from_row(row, unit_dollars=40.0, sport="nba")

    assert decision.play_pass == "PASS"
    assert decision.decision_reason == "NO ATS: margin model uncalibrated"
