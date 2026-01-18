import pandas as pd

from sports.common.elo import EloState
from sports.nhl import goalies as goalies_module
from sports.nhl.model import run_daily_nhl


def test_goalie_lookup_normalizes_keys(monkeypatch):
    st = EloState(ratings={"New Jersey Devils": 1500.0, "Boston Bruins": 1500.0})
    monkeypatch.setattr("sports.nhl.model.update_elo_from_recent_scores", lambda days_from=120: st)
    monkeypatch.setattr("sports.nhl.model.build_team_historical_total_lines", lambda **_kwargs: {})
    monkeypatch.setattr("sports.nhl.model._build_team_scoring_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_UNKNOWN_PENALTY", 0.02)
    monkeypatch.setattr(
        "sports.nhl.model.get_goalie_rating_with_meta", lambda _name, _season: (1.0, True, "ok")
    )
    monkeypatch.setattr(
        "sports.nhl.model.get_starting_goalies",
        lambda _date: {
            "New Jersey": goalies_module.GoalieInfo(
                team="New Jersey",
                goalie_name="Test Goalie",
                status="CONFIRMED",
                source="test",
            )
        },
    )

    odds_dict = {
        ("New Jersey Devils", "Boston Bruins"): {
            "home_ml": -110,
            "away_ml": 100,
            "total_points": 5.5,
            "over_price": -110,
            "under_price": -110,
        }
    }

    df = run_daily_nhl("01/16/2026", odds_dict=odds_dict)

    assert df["goalie_home_name"].fillna("").astype(str).str.strip().eq("Test Goalie").any()
    assert df["goalie_adj"].fillna(0.0).astype(float).ne(0.0).any()
