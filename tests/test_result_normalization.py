from sports.common.util import normalize_result_label


def test_normalize_result_label():
    assert normalize_result_label("w") == "WIN"
    assert normalize_result_label("W") == "WIN"
    assert normalize_result_label("l") == "LOSS"
    assert normalize_result_label("L") == "LOSS"
    assert normalize_result_label("p") == "PUSH"
    assert normalize_result_label("push") == "PUSH"
