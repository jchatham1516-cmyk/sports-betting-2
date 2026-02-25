from sports.common.normalization import remove_vig_two_way


def test_remove_vig_two_way_sums_to_one():
    home, away = remove_vig_two_way(1.91, 1.91)
    assert round(home + away, 10) == 1.0


def test_remove_vig_two_way_directional():
    home, away = remove_vig_two_way(1.8, 2.1)
    assert home > away
