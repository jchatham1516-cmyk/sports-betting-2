import pandas as pd

from sports.common.parlay import build_daily_parlay


def _base_row(
    *,
    event_id: str,
    home: str,
    away: str,
    edge: float = 0.08,
    p_win: float = 0.6,
    p_market: float = 0.52,
) -> dict:
    return {
        "sport": "nba",
        "play_pass": "PLAY",
        "edge_prob_final": edge,
        "decision_flags": "",
        "p_model_final": p_win,
        "p_model_used": p_win,
        "p_market_used": p_market,
        "p_market": p_market,
        "primary_market": "ML",
        "primary_side": "HOME",
        "home": home,
        "away": away,
        "event_id": event_id,
        "date": "2025-01-01",
        "home_ml": -110,
        "away_ml": 100,
    }


def test_daily_parlay_refuses_under_min_legs():
    rows = [
        _base_row(event_id="g1", home="Home1", away="Away1"),
        _base_row(event_id="g2", home="Home2", away="Away2"),
        _base_row(event_id="g3", home="Home3", away="Away3"),
    ]
    df = pd.DataFrame(rows)
    result = build_daily_parlay(df, min_legs=4, max_legs=7)
    assert result["status"] == "NO_PARLAY_TODAY"


def test_daily_parlay_enforces_unique_game_keys():
    rows = [
        _base_row(event_id="g1", home="TeamA", away="TeamB", edge=0.09),
        _base_row(event_id="g1", home="TeamA", away="TeamB", edge=0.085),
        _base_row(event_id="g2", home="TeamC", away="TeamD", edge=0.08),
        _base_row(event_id="g3", home="TeamE", away="TeamF", edge=0.075),
        _base_row(event_id="g4", home="TeamG", away="TeamH", edge=0.074),
    ]
    df = pd.DataFrame(rows)
    result = build_daily_parlay(df, min_legs=4, max_legs=5)
    assert result["status"] == "PARLAY_READY"
    game_keys = [leg["game_key"] for leg in result["legs"]]
    assert len(game_keys) == len(set(game_keys))


def test_daily_parlay_enforces_unique_teams():
    rows = [
        _base_row(event_id="g1", home="TeamA", away="TeamB", edge=0.09),
        _base_row(event_id="g2", home="TeamC", away="TeamD", edge=0.085),
        _base_row(event_id="g3", home="TeamE", away="TeamF", edge=0.08),
        _base_row(event_id="g4", home="TeamA", away="TeamG", edge=0.079),
        _base_row(event_id="g5", home="TeamH", away="TeamI", edge=0.078),
    ]
    df = pd.DataFrame(rows)
    result = build_daily_parlay(df, min_legs=4, max_legs=5)
    assert result["status"] == "PARLAY_READY"
    teams = [leg["team"] for leg in result["legs"]]
    assert len(teams) == len(set(teams))


def test_daily_parlay_is_deterministic():
    rows = [
        _base_row(event_id="g1", home="TeamA", away="TeamB", edge=0.09),
        _base_row(event_id="g2", home="TeamC", away="TeamD", edge=0.085),
        _base_row(event_id="g3", home="TeamE", away="TeamF", edge=0.08),
        _base_row(event_id="g4", home="TeamG", away="TeamH", edge=0.079),
        _base_row(event_id="g5", home="TeamI", away="TeamJ", edge=0.078),
    ]
    df = pd.DataFrame(rows)
    result1 = build_daily_parlay(df, min_legs=4, max_legs=5)
    result2 = build_daily_parlay(df, min_legs=4, max_legs=5)
    assert result1["legs"] == result2["legs"]
