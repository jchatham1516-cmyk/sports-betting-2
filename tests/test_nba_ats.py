import pandas as pd

import recommendations
from sports.nba import model as nba_model


def test_spread_anchor_blend():
    anchored, anchored_flag = nba_model._anchor_spread(-2.0, -6.0, 0.55)
    assert anchored_flag is True
    assert anchored != -2.0
    assert round(anchored, 4) == round(0.55 * -6.0 + 0.45 * -2.0, 4)


def test_ats_pass_without_calibrator(monkeypatch):
    monkeypatch.setattr(recommendations, "load_ats_calibrator", lambda sport: None)
    df = pd.DataFrame(
        [
            {
                "model_spread_home": -2.0,
                "home_spread": -6.0,
                "spread_price": -110,
                "home_ml": -120,
                "away_ml": 110,
                "model_home_prob": 0.55,
                "market_home_prob": 0.5,
            }
        ]
    )
    out, _ = recommendations.add_recommendations_to_df(df, sport="nba")
    assert out.loc[0, "spread_recommendation"].startswith("No ATS bet")


def test_ats_calibrated_prob_clipped():
    assert nba_model._clip_prob(0.0, 0.01, 0.99) == 0.01
    assert nba_model._clip_prob(1.0, 0.01, 0.99) == 0.99
