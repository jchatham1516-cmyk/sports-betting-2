import math

import pandas as pd

from sports.common.bet_rules import ml_probabilities_for_row


def test_goalie_shift_scaling_and_cap_applied():
    row = pd.Series(
        {
            "model_home_prob_raw": 0.55,
            "market_home_prob": 0.50,
            "goalie_prob_shift": 0.05,
            "goalie_status": "PROBABLE",
            "home": "TeamA",
            "away": "TeamB",
        }
    )

    probs = ml_probabilities_for_row(row, sport="nhl")

    assert math.isclose(probs["goalie_shift_mult"], 0.75, rel_tol=1e-6)
    assert math.isclose(probs["goalie_shift_used"], 0.02, rel_tol=1e-6)
    assert probs["goalie_shift_cap_hit"] is True
