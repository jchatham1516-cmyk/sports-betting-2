# recommendations.py
from __future__ import annotations

import math
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ==========================
# Threshold configuration
# ==========================
@dataclass
class Thresholds:
    # Moneyline (no-vig prob edge)
    ml_edge_strong: float = 0.06
    ml_edge_lean: float = 0.035

    # ATS (points)
    ats_edge_strong_pts: float = 3.0
    ats_edge_lean_pts: float = 1.5

    # Totals
    total_edge_pts: float = 3.0
    total_edge_vs_be: float = 0.025

    # Confidence bands
    conf_high: float = 0.18
    conf_med: float = 0.10


# ==========================
# Helper functions
# ==========================
def _is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return True


def _abs(x: Optional[float]) -> float:
    return abs(x) if x is not None and not _is_nan(x) else 0.0


# ==========================
# Core recommendation logic
# ==========================
def add_recommendations_to_df(
    df: pd.DataFrame,
    *,
    thresholds: Thresholds,
    model_spread_home_col: Optional[str] = None,
    model_margin_home_col: Optional[str] = None,
):
    """
    Adds:
      - ml_recommendation
      - spread_recommendation
      - total_recommendation
      - primary_recommendation
      - confidence
      - why_bet
      - value_tier

    Returns:
      (results_df, debug_df)
    """

    rows = []
    debug_rows = []

    for _, r in df.iterrows():
        # ----------------------
        # Moneyline evaluation
        # ----------------------
        edge_home = r.get("edge_home")
        ml_rec = "No ML bet (edge too small)"

        if not _is_nan(edge_home):
            if edge_home >= thresholds.ml_edge_strong:
                ml_rec = "Model PICK: HOME ML (strong)"
            elif edge_home >= thresholds.ml_edge_lean:
                ml_rec = "Model lean: HOME ML"
            elif edge_home <= -thresholds.ml_edge_strong:
                ml_rec = "Model PICK: AWAY ML (strong)"
            elif edge_home <= -thresholds.ml_edge_lean:
                ml_rec = "Model lean: AWAY ML"

        # ----------------------
        # ATS evaluation
        # ----------------------
        spread_rec = "No ATS bet (missing spread/model)"
        spread_edge = r.get("spread_edge_home")

        if not _is_nan(spread_edge):
            if abs(spread_edge) >= thresholds.ats_edge_strong_pts:
                spread_rec = "Model PICK ATS: HOME (strong)" if spread_edge > 0 else "Model PICK ATS: AWAY (strong)"
            elif abs(spread_edge) >= thresholds.ats_edge_lean_pts:
                spread_rec = "Model lean ATS: HOME" if spread_edge > 0 else "Model lean ATS: AWAY"
            else:
                spread_rec = "No ATS bet (edge too small)"

        # ----------------------
        # Totals evaluation
        # ----------------------
        total_rec = "No total bet (missing total/model)"
        total_edge_pts = r.get("total_edge_points")
        total_edge_vs_be = r.get("total_edge_vs_be")

        if not _is_nan(total_edge_pts) and not _is_nan(total_edge_vs_be):
            if abs(total_edge_pts) >= thresholds.total_edge_pts and total_edge_vs_be >= thresholds.total_edge_vs_be:
                side = r.get("total_pick_side", "")
                total_rec = f"Model PICK TOTAL: {side}"
            else:
                total_rec = "No total bet (edge too small)"

        # ----------------------
        # Confidence scoring
        # ----------------------
        conf_score = max(
            _abs(edge_home),
            _abs(spread_edge) / 10.0,
            _abs(total_edge_vs_be),
        )

        if conf_score >= thresholds.conf_high:
            confidence = "HIGH"
        elif conf_score >= thresholds.conf_med:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # ----------------------
        # Value tier
        # ----------------------
        if conf_score >= 0.08:
            value_tier = "HIGH VALUE"
        elif conf_score >= 0.04:
            value_tier = "MEDIUM VALUE"
        elif conf_score >= 0.02:
            value_tier = "LOW VALUE"
        else:
            value_tier = "NO EDGE"

        # ----------------------
        # Primary selection
        # ----------------------
        primary = ml_rec
        why = f"Primary=ML (edge={edge_home:+.3f})" if not _is_nan(edge_home) else "Primary=ML"

        if spread_rec.startswith("Model PICK ATS") and abs(spread_edge) / 10.0 > _abs(edge_home):
            primary = spread_rec
            why = f"Primary=ATS (spread_edge={spread_edge:+.2f})"

        if total_rec.startswith("Model PICK TOTAL") and _abs(total_edge_vs_be) > max(_abs(edge_home), abs(spread_edge) / 10.0):
            primary = total_rec
            why = f"Primary=TOTAL (edge_vs_be={total_edge_vs_be:+.3f})"

        # ----------------------
        # Final PASS / PLAY gate
        # ----------------------
        play_pass = "PASS"
        if confidence != "LOW" and (
            ml_rec.startswith("Model PICK")
            or spread_rec.startswith("Model PICK")
            or total_rec.startswith("Model PICK")
        ):
            play_pass = "PLAY"

        # ----------------------
        # Save row
        # ----------------------
        out = dict(r)
        out.update(
            {
                "ml_recommendation": ml_rec,
                "spread_recommendation": spread_rec,
                "total_recommendation": total_rec,
                "primary_recommendation": primary,
                "confidence": confidence,
                "why_bet": why,
                "value_tier": value_tier,
                "play_pass": play_pass,
            }
        )

        rows.append(out)

        debug_rows.append(
            {
                "home": r.get("home"),
                "away": r.get("away"),
                "edge_home": edge_home,
                "spread_edge": spread_edge,
                "total_edge_vs_be": total_edge_vs_be,
                "confidence": confidence,
                "play_pass": play_pass,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(debug_rows)
