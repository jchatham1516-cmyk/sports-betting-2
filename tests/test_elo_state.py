from sports.common.elo import EloState


def test_elo_state_persists_ratings(tmp_path):
    path = tmp_path / "elo_state.json"
    st = EloState()
    st.set("Boston Bruins", 1510.0)
    st.set("Toronto Maple Leafs", 1495.5)
    st.save(str(path))

    loaded = EloState.load(str(path))

    assert loaded.ratings["Boston Bruins"] == 1510.0
    assert loaded.ratings["Toronto Maple Leafs"] == 1495.5
