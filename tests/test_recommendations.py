import pandas as pd
from recommendations import (
    Thresholds,
    ml_recommendation,
    ats_recommendation,
    choose_primary,
    add_recommendations_to_df,
)

def test_ml_recommendation_sign_never_flips():
    t = Thresholds(ml_edge_strong=0.06, ml_edge_lean=0.035)

    # If edge_home is negative, HOME ML should never be recommended
    rec = ml_recommendation(-0.08, t)
    assert "HOME ML" not in rec
    assert "AWAY ML" in rec or "No ML bet" in rec

    # If edge_home is positive, AWAY ML should never be recommended
    rec = ml_recommendation(+0.08, t)
    assert "AWAY ML" not in rec
    assert "HOME ML" in rec or "No ML bet" in rec

def test_ats_recommendation_sign():
    t = Thresholds(ats_edge_strong_pts=3.0, ats_edge_lean_pts=1.5)

    # Positive spread edge => HOME ATS
    rec = ats_recommendation(+4.0, t)
    assert "HOME" in rec

    # Negative spread edge => AWAY ATS
    rec = ats_recommendation(-4.0, t)
    assert "AWAY" in rec

def test_primary_priority_order():
    # Strong ATS beats strong ML
    p = choose_primary("Model PICK: HOME ML (strong)", "Model PICK ATS: AWAY (strong)")
    assert "PICK ATS" in p

    # Strong ML beats lean ATS? (per our rule, lean ATS comes before lean ML; strong ML comes before lean ATS)
    p = choose_primary("Model PICK: AWAY ML (strong)", "Model lean ATS: HOME")
    assert "ML" in p and "(strong)" in p

def test_end_to_end_no_ml_inherits_ats_direction():
    # Construct a game where:
    # - ML edge says AWAY has value (edge_home negative)
    # - ATS edge says HOME has value (spread_edge_home positive)
    df = pd.DataFrame([{
        "date": "12/12/2025",
        "home": "HomeTeam",
        "away": "AwayTeam",
        "model_home_prob": 0.52,
        "home_ml": -250,  # implied ~0.714
        "away_ml": 200,
        "home_spread": +8.0,  # market gives home +8
        "model_spread_home": +3.0,  # model says home should be +3
    }])

    out, debug = add_recommendations_to_df(df, model_spread_home_col="model_spread_home")

    # ML: model prob 0.52 vs market 0.714 => edge_home negative => must be AWAY ML or no bet
    assert "HOME ML" not in out.loc[0, "ml_recommendation"]

    # ATS: +8 - +3 = +5 => should recommend HOME ATS
    assert "HOME" in out.loc[0, "spread_recommendation"]

    # Ensure they can differ without contaminating each other
    assert out.loc[0, "ml_recommendation"] != out.loc[0, "spread_recommendation"]
