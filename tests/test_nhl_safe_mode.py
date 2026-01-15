from sports.nhl.model import _shrink_prob_toward_half


def test_nhl_shrink_toward_half():
    raw = 0.8
    shrunk = _shrink_prob_toward_half(raw, 0.75)
    assert shrunk < raw
    assert round(shrunk, 3) == 0.725
