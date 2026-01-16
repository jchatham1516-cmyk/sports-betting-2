import json
from pathlib import Path

from sports.nhl import goalies as goalies_module
from sports.nhl import goalie_ratings
from sports.nhl.model import GOALIE_PROB_PER_RATING, _goalie_adjustment


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
            }
        ]
    }
    monkeypatch.setattr(goalie_ratings, "_fetch_goalie_stats", lambda _season: payload)

    rating = goalie_ratings.get_goalie_rating("Igor Shesterkin", "2024")
    assert isinstance(rating, float)
    assert rating > 0

    missing = goalie_ratings.get_goalie_rating("Unknown Goalie", "2024")
    assert missing == 0.0


def test_goalie_adjustment_returns_raw_direction():
    diff = 30.0 - (-30.0)
    expected = diff * GOALIE_PROB_PER_RATING
    assert _goalie_adjustment(30.0, -30.0) == expected
    assert _goalie_adjustment(-30.0, 30.0) == -expected
