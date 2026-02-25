from sports.common.betting import BettingConfig, kelly_fraction, size_bet_units


def test_kelly_fraction_positive_edge():
    frac = kelly_fraction(0.55, 2.0)
    assert round(frac, 4) == 0.1


def test_kelly_fraction_negative_edge():
    assert kelly_fraction(0.45, 2.0) == 0.0


def test_size_bet_units_respects_caps():
    config = BettingConfig(bankroll=100, fractional_kelly=1.0, max_bet_units=2.0)
    units, reason = size_bet_units(prob_win=0.7, decimal_odds=2.0, edge=0.2, config=config)
    assert reason == "play"
    assert units == 2.0
