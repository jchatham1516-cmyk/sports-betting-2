import math

from sports.common.util import implied_prob_from_american, remove_vig_two_way


def test_implied_prob_from_american_basic():
    assert math.isclose(implied_prob_from_american(-110), 110 / (110 + 100))
    assert math.isclose(implied_prob_from_american(150), 100 / (150 + 100))


def test_remove_vig_two_way():
    ph = implied_prob_from_american(-110)
    pa = implied_prob_from_american(-110)
    no_vig = remove_vig_two_way(ph, pa)
    assert no_vig is not None
    h, a = no_vig
    assert math.isclose(h + a, 1.0)
    assert math.isclose(h, 0.5)
