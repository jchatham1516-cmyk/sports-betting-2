from sports.common.normalization import american_to_decimal, decimal_to_implied_prob


def test_american_to_decimal_positive():
    assert american_to_decimal(150) == 2.5


def test_american_to_decimal_negative():
    assert round(american_to_decimal(-200), 4) == 1.5


def test_decimal_to_implied_prob():
    assert round(decimal_to_implied_prob(2.0), 4) == 0.5
