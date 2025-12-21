import pandas as pd
import pytest

from sports.common.elo import EloState
from sports.nba import model


class _DummyCal:
    def predict_spread(self, elo_diff):
        return float(elo_diff) / 30.0


def test_build_form_adjustments_uses_recent_net_rating():
    stats_df = pd.DataFrame({
        "TEAM_NAME": ["Alpha Team", "Beta Team"],
        "ORtg_RECENT": [112.0, 102.0],
        "DRtg_RECENT": [107.0, 105.0],
    })

    adjs = model._build_form_adjustments(stats_df)

    expected_alpha = (5.0 - 1.0) * model.FORM_ELO_PER_NET  # centered vs league avg net
    expected_beta = (-3.0 - 1.0) * model.FORM_ELO_PER_NET

    assert adjs["Alpha Team"] == pytest.approx(expected_alpha)
    assert adjs["Beta Team"] == pytest.approx(expected_beta)


def test_missing_team_falls_back_to_pure_elo(monkeypatch):
    stats_df = pd.DataFrame({
        "TEAM_NAME": ["Alpha Team", "Beta Team"],
        "ORtg_RECENT": [112.0, 102.0],
        "DRtg_RECENT": [107.0, 105.0],
    })
    dummy_state = EloState(ratings={}, processed_games={})

    monkeypatch.setattr(model, "update_elo_from_recent_scores", lambda days_from=3: dummy_state)
    monkeypatch.setattr(model, "load_nba_calibrator", lambda: _DummyCal())
    monkeypatch.setattr(model, "update_and_save_nba_calibration", lambda: _DummyCal())

    odds = {
        ("Alpha Team", "Beta Team"): {},
        ("Gamma Team", "Delta Team"): {},
    }

    results = model.run_daily_nba("01/01/2024", odds_dict=odds, stats_df=stats_df)
    adjs = model._build_form_adjustments(stats_df)

    known_row = results.loc[results["home"] == "Alpha Team"].iloc[0]
    expected_diff = (adjs["Alpha Team"] - adjs["Beta Team"]) + model.HOME_ADV
    assert known_row["elo_diff"] == pytest.approx(expected_diff)

    missing_row = results.loc[results["home"] == "Gamma Team"].iloc[0]
    assert missing_row["elo_diff"] == pytest.approx(model.HOME_ADV)
