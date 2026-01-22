import pandas as pd

from sports.common.elo import EloState
from sports.nba import model


def test_fallback_spread_used_when_margin_cal_untrained(monkeypatch):
    dummy_state = EloState(ratings={"Home Team": 1600.0, "Away Team": 1400.0}, processed_games=set())

    monkeypatch.setattr(model, "update_elo_from_recent_scores", lambda days_from=3, st=None: dummy_state)
    monkeypatch.setattr(model, "backfill_nba_elo_state", lambda **kwargs: dummy_state)
    monkeypatch.setattr(model, "_build_team_scoring_table", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(model, "build_team_historical_total_lines", lambda **kwargs: {})
    monkeypatch.setattr(model, "_fetch_from_official_nba", lambda: {})
    monkeypatch.setattr(model, "_fetch_from_espn", lambda: {})
    monkeypatch.setattr(model, "_load_calibrators", lambda: (None, None))

    odds = {
        ("Home Team", "Away Team"): {"home_ml": -110, "away_ml": 100, "home_spread": -6.0},
    }

    results = model.run_daily_nba("01/01/2024", odds_dict=odds, stats_df=None)
    row = results.iloc[0]

    assert row["model_spread_home"] != 0.0
    assert row["fallback_spread_home"] == row["model_spread_home"]
    assert not bool(row["margin_cal_used"])
