import pandas as pd
from recommendations import Thresholds, ml_recommendation, ats_recommendation, choose_primary, add_recommendations_to_df


def test_ml_recommendation_sign_never_flips():
    t = Thresholds(ml_edge_strong=0.06, ml_edge_lean=0.035)

    rec = ml_recommendation(-0.08, t)
    assert "HOME ML" not in rec
    assert ("AWAY ML" in rec) or ("No ML bet" in rec)

    rec = ml_recommendation(+0.08, t)
    assert "AWAY ML" not in rec
    assert ("HOME ML" in rec) or ("No ML bet" in rec)


def test_ats_recommendation_sign():
    t = Thresholds(ats_edge_strong_pts=3.0, ats_edge_lean_pts=1.5)
    assert "HOME" in ats_recommendation(+4.0, t)
    assert "AWAY" in ats_recommendation(-4.0, t)


def test_primary_priority_order():
    p = choose_primary("Model PICK: HOME ML (strong)", "Model PICK ATS: AWAY (strong)")
    assert "PICK ATS" in p

    p = choose_primary("Model PICK: AWAY ML (strong)", "Model lean ATS: HOME")
    assert "ML" in p and "(strong)" in p


def test_end_to_end_ml_does_not_inherit_ats():
    df = pd.DataFrame([{
        "date": "12/12/2025",
        "home": "HomeTeam",
        "away": "AwayTeam",
        "model_home_prob": 0.52,
        "home_ml": -250,   # implied ~0.714
        "away_ml": 200,
        "home_spread": +8.0,
        "model_spread_home": +3.0,
    }])

    out, debug = add_recommendations_to_df(df, model_spread_home_col="model_spread_home")

    # ML edge is negative => must NOT recommend HOME ML
    assert "HOME ML" not in out.loc[0, "ml_recommendation"]

    # ATS edge is +5 => should recommend HOME ATS
    assert "HOME" in out.loc[0, "spread_recommendation"]
