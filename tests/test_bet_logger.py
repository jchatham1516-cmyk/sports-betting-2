import pandas as pd

from sports.common.bet_logger import append_plays_to_bet_log


def _sample_predictions():
    return pd.DataFrame(
        [
            {
                "date": "01/01/2024",
                "home": "Lakers",
                "away": "Bulls",
                "play_pass": "PLAY",
                "units": 1.5,
                "primary_market": "ML",
                "primary_side": "HOME",
                "home_ml": -120,
                "away_ml": 110,
                "model_home_prob": 0.55,
                "market_home_prob": 0.52,
                "confidence": "HIGH",
                "value_tier": "HIGH VALUE",
                "primary_ev": 0.05,
                "unit_dollars": 12,
            },
            {
                "date": "01/01/2024",
                "home": "Lakers",
                "away": "Bulls",
                "play_pass": "PASS",
                "units": 1.0,
                "primary_market": "ML",
                "primary_side": "AWAY",
                "home_ml": -120,
                "away_ml": 110,
            },
        ]
    )


def test_append_plays_filters_and_logs(tmp_path):
    preds = _sample_predictions()
    log_path = tmp_path / "tracking" / "bet_log.csv"

    added = append_plays_to_bet_log(preds, "nba", bet_log_path=str(log_path))
    assert added == 1
    assert log_path.exists()

    logged = pd.read_csv(log_path)
    assert len(logged) == 1
    row = logged.iloc[0]
    assert row["market_type"] == "moneyline"
    assert row["side"] == "HOME"
    assert row["price_at_bet"] == -120
    assert row["units"] == 1.5
    assert row["stake_dollars"] == 18

    # Calling again should not duplicate entries
    added_again = append_plays_to_bet_log(preds, "nba", bet_log_path=str(log_path))
    logged_again = pd.read_csv(log_path)
    assert added_again == 0
    assert len(logged_again) == 1


def test_append_plays_flips_probs_for_away(tmp_path):
    preds = _sample_predictions().copy()
    preds.loc[0, "primary_side"] = "AWAY"

    log_path = tmp_path / "bet_log.csv"
    append_plays_to_bet_log(preds, "nba", bet_log_path=str(log_path))

    logged = pd.read_csv(log_path)
    row = logged.iloc[0]
    # model prob should flip to the away side (1 - 0.55)
    assert round(row["model_prob"], 3) == 0.45
