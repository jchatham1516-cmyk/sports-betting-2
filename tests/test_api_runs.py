from fastapi.testclient import TestClient


def test_create_run(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")

    from app.core import runner
    from app.main import app

    def fake_run_model(sport, game_date, settings=None):
        return {
            "predictions_path": "results/predictions_nba_2024-01-01.csv",
            "tracked_bets_path": "results/tracked_bets_nba_2024-01-01.csv",
            "predictions_rows": [
                {
                    "date": "2024-01-01",
                    "home": "Team A",
                    "away": "Team B",
                    "play_pass": "PLAY",
                }
            ],
            "tracked_bets_rows": [
                {
                    "bet_date": "2024-01-01",
                    "sport": "nba",
                    "market": "ml",
                    "home": "Team A",
                    "away": "Team B",
                    "pick": "Team A",
                    "price": -110,
                    "units": 1.0,
                    "result": "PENDING",
                }
            ],
            "log": "ok",
        }

    monkeypatch.setattr(runner, "run_model", fake_run_model)

    client = TestClient(app)
    response = client.post(
        "/api/runs",
        json={"sport": "nba", "game_date": "2024-01-01"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "done"
    assert payload["predictions_count"] == 1
    assert payload["tracked_bets_count"] == 1
