import math

from sports.common import tracker


def test_grade_moneyline_win_loss():
    assert tracker.grade_moneyline("HOME", 105, 99) == "WIN"
    assert tracker.grade_moneyline("AWAY", 95, 100) == "WIN"
    assert tracker.grade_moneyline("HOME", 90, 100) == "LOSS"


def test_grade_spread_push_and_win():
    # Home -3 wins by exactly 3 => push
    assert tracker.grade_spread("HOME", -3.0, 100, 97) == "PUSH"
    # Away +3 covers when loses by 2
    assert tracker.grade_spread("AWAY", -3.0, 101, 99) == "WIN"


def test_grade_total_push_and_under():
    assert tracker.grade_total("OVER", 200.0, 100, 100) == "PUSH"
    assert tracker.grade_total("UNDER", 210.5, 100, 105) == "WIN"


def test_profit_from_american_odds():
    stake_win, profit_win, payout_win = tracker._calc_profit_and_payout(5, 10, 120, "WIN")
    assert math.isclose(stake_win, 50.0, rel_tol=1e-6)
    assert math.isclose(profit_win, 60.0, rel_tol=1e-6)
    assert math.isclose(payout_win, 110.0, rel_tol=1e-6)

    stake_loss, profit_loss, payout_loss = tracker._calc_profit_and_payout(5, 10, -110, "LOSS")
    assert math.isclose(stake_loss, 50.0, rel_tol=1e-6)
    assert math.isclose(profit_loss, -50.0, rel_tol=1e-6)
    assert payout_loss == 0

    stake_push, profit_push, payout_push = tracker._calc_profit_and_payout(5, 10, -110, "PUSH")
    assert math.isclose(stake_push, 50.0, rel_tol=1e-6)
    assert profit_push == 0
    assert payout_push == 50


def test_profit_with_partial_units():
    stake_win, profit_win, payout_win = tracker._calc_profit_and_payout(0.25, 10, 200, "WIN")
    assert math.isclose(stake_win, 2.5, rel_tol=1e-6)
    assert math.isclose(profit_win, 5.0, rel_tol=1e-6)
    assert math.isclose(payout_win, 7.5, rel_tol=1e-6)

    stake_loss, profit_loss, payout_loss = tracker._calc_profit_and_payout(0.25, 10, 200, "LOSS")
    assert math.isclose(stake_loss, 2.5, rel_tol=1e-6)
    assert math.isclose(profit_loss, -2.5, rel_tol=1e-6)
    assert payout_loss == 0

    stake_push, profit_push, payout_push = tracker._calc_profit_and_payout(0.25, 10, 200, "PUSH")
    assert math.isclose(stake_push, 2.5, rel_tol=1e-6)
    assert profit_push == 0
    assert payout_push == 2.5


def test_profit_from_fractional_units_loss_and_win():
    stake_loss, profit_loss, payout_loss = tracker._calc_profit_and_payout(0.25, 10, -106, "LOSS")
    assert math.isclose(stake_loss, 2.5, rel_tol=1e-6)
    assert math.isclose(profit_loss, -2.5, rel_tol=1e-6)
    assert payout_loss == 0

    stake_win, profit_win, payout_win = tracker._calc_profit_and_payout(0.25, 10, 200, "WIN")
    assert math.isclose(stake_win, 2.5, rel_tol=1e-6)
    assert math.isclose(profit_win, 5.0, rel_tol=1e-6)
    assert math.isclose(payout_win, 7.5, rel_tol=1e-6)


def test_american_to_decimal_conversion():
    assert math.isclose(tracker._american_to_decimal(150), 2.5, rel_tol=1e-6)
    assert math.isclose(tracker._american_to_decimal(-200), 1.5, rel_tol=1e-6)
