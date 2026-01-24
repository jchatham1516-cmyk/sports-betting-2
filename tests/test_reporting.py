import pandas as pd

from sports.common.reporting import build_display_columns, build_rankings, render_console_report


def test_rank_accuracy_orders_by_prob():
    df = pd.DataFrame(
        [
            {
                "primary_market": "ML",
                "primary_side": "HOME",
                "primary_price": -120,
                "p_model_final": 0.62,
                "edge_prob_final": 0.04,
                "confidence": "MEDIUM",
            },
            {
                "primary_market": "ML",
                "primary_side": "AWAY",
                "primary_price": 140,
                "p_model_final": 0.55,
                "edge_prob_final": 0.02,
                "confidence": "HIGH",
            },
            {
                "primary_market": "ML",
                "primary_side": "HOME",
                "primary_price": -180,
                "p_model_final": 0.71,
                "edge_prob_final": 0.06,
                "confidence": "LOW",
            },
        ]
    )

    ranked = build_rankings(build_display_columns(df))
    ordered = ranked.sort_values("rank_accuracy").index.tolist()
    assert ordered == [2, 0, 1]


def test_top_plays_only_includes_play_rows(capsys):
    df = pd.DataFrame(
        [
            {
                "primary_market": "ML",
                "primary_side": "HOME",
                "primary_price": -110,
                "p_model_final": 0.6,
                "edge_prob_final": 0.03,
                "primary_ev": 0.02,
                "abs_edge_prob": 0.03,
                "confidence": "MEDIUM",
                "play_pass": "PLAY",
                "decision_reason": "PLAY_REASON_ONE",
            },
            {
                "primary_market": "ML",
                "primary_side": "AWAY",
                "primary_price": 150,
                "p_model_final": 0.58,
                "edge_prob_final": 0.05,
                "primary_ev": 0.08,
                "abs_edge_prob": 0.05,
                "confidence": "HIGH",
                "play_pass": "PASS",
                "decision_reason": "PASS_REASON_SHOULD_NOT_APPEAR",
            },
            {
                "primary_market": "ML",
                "primary_side": "AWAY",
                "primary_price": 105,
                "p_model_final": 0.57,
                "edge_prob_final": 0.04,
                "primary_ev": 0.01,
                "abs_edge_prob": 0.04,
                "confidence": "LOW",
                "play_pass": "PLAY",
                "decision_reason": "PLAY_REASON_TWO",
            },
        ]
    )

    render_console_report(df, sport="nba", date="01/01/2026", debug=False)
    output = capsys.readouterr().out
    top_section = output.split("=== Top Plays ===")[1].split("=== Accuracy Ranking ===")[0]
    assert "PASS_REASON_SHOULD_NOT_APPEAR" not in top_section
