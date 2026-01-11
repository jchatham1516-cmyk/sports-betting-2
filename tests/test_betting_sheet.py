import pandas as pd

from betting_sheet import convert_to_betting_sheet


def _build_sample_df():
    return pd.DataFrame(
        [
            {
                "Date": "12/29/2025",
                "Home": "Charlotte Hornets",
                "Away": "Milwaukee Bucks",
                "model_home_prob": 0.581870426,
                "market_home_prob": 0.421010425,
                "home_ml": 128,
                "away_ml": -152,
                "home_spread": 3.0,
                "spread_price": -110,
                "model_spread_home": -1.512184375,
            }
        ]
    )


def test_convert_to_betting_sheet_creates_output(tmp_path):
    df = _build_sample_df()
    src = tmp_path / "raw.csv"
    df.to_csv(src, index=False)

    result_df = convert_to_betting_sheet(src, sport="nba", unit_dollars=20.0, min_play_edge_abs=0.05)

    output_path = tmp_path / "raw_betting_sheet.csv"
    assert output_path.exists()

    expected_cols = {
        "primary_recommendation",
        "play_pass",
        "bet_size",
        "unit_dollars",
        "units",
        "decision_flags",
        "decision_reason",
        "raw_units",
        "final_units",
        "p_model_used",
        "p_market_used",
        "abs_edge_used",
        "abs_edge_prob",
        "confidence_reason",
        "stake_dollars",
    }
    assert expected_cols.issubset(result_df.columns)
    assert result_df.loc[0, "unit_dollars"] == 20.0
    assert result_df.loc[0, "play_pass"] in {"PLAY", "PASS"}


def test_convert_handles_case_insensitive_columns(tmp_path):
    df = pd.DataFrame(
        [
            {
                "DATE": "12/29/2025",
                "Home": "Team A",
                "Away": "Team B",
                "HOME_ML": 110,
                "AWAY_ML": -130,
                "MODEL_HOME_PROB": 0.55,
                "MARKET_HOME_PROB": 0.45,
            }
        ]
    )
    src = tmp_path / "case.csv"
    df.to_csv(src, index=False)

    result_df = convert_to_betting_sheet(src, sport="nba", min_play_edge_abs=0.02)

    assert "home_ml" in result_df.columns
    assert "market_home_prob" in result_df.columns
    assert pd.notna(result_df.loc[0, "home_ml"])
