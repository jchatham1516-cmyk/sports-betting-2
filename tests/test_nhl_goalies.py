import json
from pathlib import Path

import pandas as pd

from sports.nhl import goalies as goalies_module
from sports.nhl import goalie_ratings
from sports.common.elo import EloState
from sports.nhl.model import _compute_goalie_adjustment, run_daily_nhl


def test_goalie_provider_parses_daily_faceoff_table(monkeypatch, tmp_path):
    html = """
    <html>
      <body>
        <table>
          <tr><th>Team</th><th>Goalie</th><th>Status</th></tr>
          <tr><td>Toronto Maple Leafs</td><td>Joseph Woll</td><td>Projected</td></tr>
          <tr><td>NY Rangers</td><td>Igor Shesterkin</td><td>Confirmed</td></tr>
        </table>
      </body>
    </html>
    """
    monkeypatch.setattr(goalies_module, "GOALIE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(goalies_module, "_get_with_retry", lambda *_args, **_kwargs: (html, 200))

    results = goalies_module.get_starting_goalies("2024-10-10")

    assert results["Toronto Maple Leafs"].goalie_name == "Joseph Woll"
    assert results["Toronto Maple Leafs"].status == "PROJECTED"
    assert results["New York Rangers"].goalie_name == "Igor Shesterkin"
    assert results["New York Rangers"].status == "CONFIRMED"

    cache_path = tmp_path / "nhl_goalies_2024-10-10.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert "goalies" in payload
    assert payload["source"] == "puckpedia"
    assert payload["date_key"] == "2024-10-10"


def test_goalie_provider_parses_daily_faceoff_fixture():
    html = Path("tests/fixtures/dailyfaceoff_goalies.html").read_text(encoding="utf-8")
    results = goalies_module._parse_daily_faceoff(html)

    assert len(results) >= 5
    assert "Boston Bruins" in results
    assert "Toronto Maple Leafs" in results


def test_goalie_provider_parses_puckpedia_fixture():
    html = Path("tests/fixtures/puckpedia_starting_goalies.html").read_text(encoding="utf-8")
    results = goalies_module._parse_puckpedia(html)

    assert len(results) >= 3
    assert results["Boston Bruins"].goalie_name == "Jeremy Swayman"
    assert results["Boston Bruins"].status == "CONFIRMED"
    assert results["Toronto Maple Leafs"].status == "PROJECTED"


def test_goalie_rating_known_and_unknown(monkeypatch):
    payload = {
        "data": [
            {
                "goalieFullName": "Igor Shesterkin",
                "savePct": 0.92,
                "gamesPlayed": 12,
            },
            {
                "goalieFullName": "Connor Hellebuyck",
                "savePct": 0.88,
                "gamesPlayed": 12,
            }
        ]
    }
    monkeypatch.setattr(goalie_ratings, "_fetch_goalie_stats", lambda _season: payload)

    rating = goalie_ratings.get_goalie_rating("Igor Shesterkin", "2024")
    assert isinstance(rating, float)
    assert rating > 0

    missing = goalie_ratings.get_goalie_rating("Unknown Goalie", "2024")
    assert missing == 0.0


def test_goalie_adj_non_zero_when_both_goalies_found(monkeypatch):
    def _rating(name, _season):
        return (1.8, True) if "Home" in name else (-1.2, True)

    monkeypatch.setattr("sports.nhl.model.get_goalie_rating_with_meta", _rating)
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_WEIGHT", 0.5)
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_MAX_PROB_SHIFT", 0.08)
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_UNKNOWN_PENALTY", 0.01)

    adj, _, _, _, _, _, reason = _compute_goalie_adjustment(
        goalie_home_name="Home Goalie",
        goalie_away_name="Away Goalie",
        goalie_home_status="CONFIRMED",
        goalie_away_status="CONFIRMED",
        season_label="2024",
    )

    assert adj != 0.0
    assert reason == "goalie_rating_applied"


def test_goalie_adj_limited_for_elite_vs_weak(monkeypatch):
    def _rating(name, _season):
        return (3.0, True) if "Home" in name else (-3.0, True)

    monkeypatch.setattr("sports.nhl.model.get_goalie_rating_with_meta", _rating)
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_WEIGHT", 0.35)
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_MAX_PROB_SHIFT", 0.06)

    adj, _, _, _, _, _, _ = _compute_goalie_adjustment(
        goalie_home_name="Home Goalie",
        goalie_away_name="Away Goalie",
        goalie_home_status="CONFIRMED",
        goalie_away_status="CONFIRMED",
        season_label="2024",
    )

    assert abs(adj) <= 0.06


def test_goalie_adj_zero_for_equal_goalies(monkeypatch):
    monkeypatch.setattr("sports.nhl.model.get_goalie_rating_with_meta", lambda _name, _season: (1.2, True))
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_WEIGHT", 0.35)
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_MAX_PROB_SHIFT", 0.06)

    adj, _, _, _, _, _, _ = _compute_goalie_adjustment(
        goalie_home_name="Home Goalie",
        goalie_away_name="Away Goalie",
        goalie_home_status="CONFIRMED",
        goalie_away_status="CONFIRMED",
        season_label="2024",
    )

    assert abs(adj) < 1e-6


def test_goalie_adj_small_when_goalie_missing(monkeypatch):
    monkeypatch.setattr("sports.nhl.model.get_goalie_rating_with_meta", lambda _name, _season: (1.2, True))
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_UNKNOWN_PENALTY", 0.01)
    monkeypatch.setattr("sports.nhl.model.NHL_GOALIE_MAX_PROB_SHIFT", 0.06)

    adj, _, _, _, _, _, reason = _compute_goalie_adjustment(
        goalie_home_name="Home Goalie",
        goalie_away_name="",
        goalie_home_status="CONFIRMED",
        goalie_away_status="UNKNOWN",
        season_label="2024",
    )

    assert abs(adj) <= 0.01
    assert reason == "goalie_missing_opponent"


def test_run_daily_nhl_uses_goalies_when_available(monkeypatch):
    st = EloState(ratings={"Boston Bruins": 1500.0, "Toronto Maple Leafs": 1500.0})
    monkeypatch.setattr("sports.nhl.model.update_elo_from_recent_scores", lambda days_from=120: st)
    monkeypatch.setattr("sports.nhl.model.build_team_historical_total_lines", lambda **_kwargs: {})
    monkeypatch.setattr("sports.nhl.model._build_team_scoring_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(
        "sports.nhl.model.get_goalie_rating_with_meta",
        lambda name, _season: ((15.0, True) if "Home" in name else (-10.0, True)),
    )
    monkeypatch.setattr(
        "sports.nhl.model.get_starting_goalies",
        lambda _date: {
            "Boston Bruins": goalies_module.GoalieInfo(
                team="Boston Bruins",
                goalie_name="Home Starter",
                status="CONFIRMED",
                source="test",
            ),
            "Toronto Maple Leafs": goalies_module.GoalieInfo(
                team="Toronto Maple Leafs",
                goalie_name="Away Starter",
                status="CONFIRMED",
                source="test",
            ),
        },
    )

    odds_dict = {
        ("Boston Bruins", "Toronto Maple Leafs"): {
            "home_ml": -110,
            "away_ml": 100,
            "total_points": 5.5,
            "over_price": -110,
            "under_price": -110,
        }
    }
    df = run_daily_nhl("01/16/2026", odds_dict=odds_dict)

    assert df["goalie_home_name"].fillna("").astype(str).str.strip().ne("").any()
    assert df["goalie_adj"].fillna(0.0).astype(float).ne(0.0).any()
