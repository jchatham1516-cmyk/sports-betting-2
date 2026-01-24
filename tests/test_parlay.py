import pandas as pd

from sports.common.parlay import build_weekly_parlay


def _base_row(idx: int, *, home: str, away: str, edge: float = 0.06) -> dict:
    return {
        "sport": "nba",
        "play_pass": "PLAY",
        "edge_prob_final": edge,
        "decision_flags": "",
        "p_model_cal": 0.56,
        "p_market": 0.48,
        "p_model_final": 0.56,
        "primary_market": "ML",
        "primary_side": "HOME",
        "home": home,
        "away": away,
        "date": "2025-01-01",
        "home_ml": -110,
        "away_ml": 100,
        "confidence": "HIGH",
    }


def test_weekly_parlay_refuses_under_min_legs():
    rows = [_base_row(i, home=f"Home{i}", away=f"Away{i}") for i in range(5)]
    df = pd.DataFrame(rows)
    result = build_weekly_parlay(df, min_legs=6, max_legs=6)
    assert result["status"] == "NO_PARLAY_THIS_WEEK"


def test_parlay_builder_enforces_game_and_team_limits():
    rows = [
        _base_row(0, home="TeamA", away="TeamB", edge=0.09),
        _base_row(1, home="TeamC", away="TeamD", edge=0.08),
        _base_row(2, home="TeamE", away="TeamF", edge=0.07),
        _base_row(3, home="TeamG", away="TeamH", edge=0.065),
        _base_row(4, home="TeamI", away="TeamJ", edge=0.064),
        _base_row(5, home="TeamK", away="TeamL", edge=0.063),
        _base_row(6, home="TeamA", away="TeamM", edge=0.062),
        _base_row(7, home="TeamC", away="TeamD", edge=0.061),
    ]
    df = pd.DataFrame(rows)
    result = build_weekly_parlay(df, min_legs=6, max_legs=7)
    assert result["status"] == "PARLAY_READY"

    legs = result["legs"]
    game_keys = [leg["game_key"] for leg in legs]
    assert len(game_keys) == len(set(game_keys))

    teams = [leg["team"] for leg in legs]
    assert len(teams) == len(set(teams))
