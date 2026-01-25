import pandas as pd

from sports.common.parlay import build_smart_parlay


def _base_row(
    *,
    sport: str,
    game_key: str,
    home: str,
    away: str,
    market: str = "moneyline",
    side: str = "HOME",
    edge: float = 0.08,
    p_win: float = 0.62,
    p_market: float = 0.52,
    ev: float = 0.04,
    confidence: str = "HIGH",
) -> dict:
    return {
        "sport": sport,
        "play_pass": "PLAY",
        "primary_market": market,
        "primary_side": side,
        "p_model_final": p_win,
        "p_market_used": p_market,
        "edge_prob_final": edge,
        "abs_edge_prob": abs(edge),
        "primary_ev": ev,
        "confidence": confidence,
        "value_tier": "HIGH VALUE",
        "decision_flags": "",
        "reason_short": "Value edge",
        "primary_price": -110,
        "home": home,
        "away": away,
        "game_key": game_key,
    }


def test_smart_parlay_refuses_under_min_legs():
    rows = [
        _base_row(sport="nba", game_key="g1", home="Home1", away="Away1"),
        _base_row(sport="nba", game_key="g2", home="Home2", away="Away2"),
        _base_row(sport="nba", game_key="g3", home="Home3", away="Away3"),
    ]
    df = pd.DataFrame(rows)
    result = build_smart_parlay(df, min_legs=4, max_legs=7)
    assert result["status"] == "NO_PARLAY_TODAY"


def test_smart_parlay_enforces_unique_game_keys():
    rows = [
        _base_row(sport="nba", game_key="g1", home="TeamA", away="TeamB", edge=0.09),
        _base_row(sport="nba", game_key="g1", home="TeamA", away="TeamB", edge=0.085),
        _base_row(sport="nba", game_key="g2", home="TeamC", away="TeamD", edge=0.08),
        _base_row(sport="nba", game_key="g3", home="TeamE", away="TeamF", edge=0.075),
        _base_row(sport="nba", game_key="g4", home="TeamG", away="TeamH", edge=0.074),
    ]
    df = pd.DataFrame(rows)
    result = build_smart_parlay(df, min_legs=4, max_legs=5)
    assert result["status"] == "PARLAY_READY"
    game_keys = [leg["game_key"] for leg in result["legs"]]
    assert len(game_keys) == len(set(game_keys))


def test_smart_parlay_enforces_unique_teams():
    rows = [
        _base_row(sport="nba", game_key="g1", home="TeamA", away="TeamB", edge=0.09),
        _base_row(sport="nba", game_key="g2", home="TeamC", away="TeamD", edge=0.085),
        _base_row(sport="nba", game_key="g3", home="TeamE", away="TeamF", edge=0.08),
        _base_row(sport="nba", game_key="g4", home="TeamA", away="TeamG", edge=0.079),
        _base_row(sport="nba", game_key="g5", home="TeamH", away="TeamI", edge=0.078),
    ]
    df = pd.DataFrame(rows)
    result = build_smart_parlay(df, min_legs=4, max_legs=5)
    assert result["status"] == "PARLAY_READY"
    teams = [leg["home"] for leg in result["legs"]] + [leg["away"] for leg in result["legs"]]
    assert len(teams) == len(set(teams))


def test_smart_parlay_respects_sport_and_market_caps():
    rows = [
        _base_row(sport="nba", game_key="g1", home="TeamA", away="TeamB", market="moneyline", edge=0.09),
        _base_row(sport="nba", game_key="g2", home="TeamC", away="TeamD", market="moneyline", edge=0.088),
        _base_row(sport="nba", game_key="g3", home="TeamE", away="TeamF", market="moneyline", edge=0.087),
        _base_row(sport="nba", game_key="g4", home="TeamG", away="TeamH", market="spread", edge=0.086),
        _base_row(sport="nhl", game_key="g5", home="TeamI", away="TeamJ", market="total", edge=0.085),
        _base_row(sport="nhl", game_key="g6", home="TeamK", away="TeamL", market="total", edge=0.084),
    ]
    df = pd.DataFrame(rows)
    result = build_smart_parlay(df, min_legs=4, max_legs=6)
    assert result["status"] == "PARLAY_READY"
    legs = result["legs"]
    sport_counts = {}
    market_counts = {}
    for leg in legs:
        sport_counts[leg["sport"]] = sport_counts.get(leg["sport"], 0) + 1
        market_counts[leg["market"]] = market_counts.get(leg["market"], 0) + 1
    total = len(legs)
    assert all(count / total <= 0.60 for count in sport_counts.values())
    assert all(count / total <= 0.60 for count in market_counts.values())


def test_smart_parlay_is_deterministic():
    rows = [
        _base_row(sport="nba", game_key="g1", home="TeamA", away="TeamB", edge=0.09),
        _base_row(sport="nba", game_key="g2", home="TeamC", away="TeamD", edge=0.085),
        _base_row(sport="nhl", game_key="g3", home="TeamE", away="TeamF", edge=0.08),
        _base_row(sport="nhl", game_key="g4", home="TeamG", away="TeamH", edge=0.079),
        _base_row(sport="nba", game_key="g5", home="TeamI", away="TeamJ", edge=0.078),
    ]
    df = pd.DataFrame(rows)
    result1 = build_smart_parlay(df, min_legs=4, max_legs=5)
    result2 = build_smart_parlay(df, min_legs=4, max_legs=5)
    assert result1["legs"] == result2["legs"]
