import pytest

from sports.nba import model


def test_anchor_model_total():
    anchored = model._anchor_model_total(220.0, 230.0, 0.70)
    assert anchored == pytest.approx(0.70 * 230.0 + 0.30 * 220.0)


def test_total_pick_from_edge_over():
    side, reco, flags = model._total_pick_from_edge(
        5.0,
        min_edge=4.0,
        sanity_fail=False,
        anchored=True,
    )
    assert side == "OVER"
    assert reco == "Model PICK TOTAL: OVER"
    assert "TOTAL_EDGE_OK" in flags
    assert "TOTAL_ANCHORED" in flags


def test_total_pick_from_edge_pass_small():
    side, reco, flags = model._total_pick_from_edge(
        2.0,
        min_edge=4.0,
        sanity_fail=False,
        anchored=False,
    )
    assert side == "NONE"
    assert reco == "No total bet (edge too small)"
    assert "TOTAL_EDGE_TOO_SMALL" in flags


def test_total_pick_from_edge_sanity_fail():
    side, reco, flags = model._total_pick_from_edge(
        10.0,
        min_edge=4.0,
        sanity_fail=True,
        anchored=True,
    )
    assert side == "NONE"
    assert "sanity fail" in reco
    assert "TOTAL_SANITY_FAIL_PASS" in flags


def test_injury_total_adjustment_applies():
    home_inj = [
        {"player": "Guard One", "pos": "PG", "role": "starter", "status_mult": 1.0, "impact": 2.0},
    ]
    away_inj = []
    adj, reason = model._injury_total_adjustment(home_inj, away_inj)
    assert adj != 0.0
    assert "scorer" in reason or "ball-handler" in reason or "rim-protector" in reason
