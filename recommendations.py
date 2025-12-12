from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math
import pandas as pd


# -------------------------
# Odds / probability helpers
# -------------------------

def american_to_implied_prob(ml: float) -> float:
    """
    Convert American moneyline odds to implied probability (no vig removal).
    """
    ml = float(ml)
    if ml == 0:
        return 0.5
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -------------------------
# Model convention helpers
# -------------------------

def model_margin_to_home_spread(model_margin_home: float) -> float:
    """
    Convert model expected margin (HOME - AWAY) into Vegas-style home spread:
      margin +5 (home by 5) -> spread -5
      margin -5 (home loses by 5) -> spread +5
    """
    return -float(model_margin_home)


# -------------------------
# Recommendations
# -------------------------

@dataclass(frozen=True)
class Thresholds:
    # ML edge thresholds (probability points)
    ml_edge_strong: float = 0.06
    ml_edge_lean: float = 0.035

    # ATS edge thresholds (points)
    ats_edge_strong_pts: float = 3.0
    ats_edge_lean_pts: float = 1.5

    # Confidence (certainty) thresholds based on abs(p - 0.5)
    conf_high: float = 0.18
    conf_med: float = 0.10


def confidence_from_prob(model_home_prob: float, t: Thresholds) -> str:
    certainty = abs(float(model_home_prob) - 0.5)
    if certainty >= t.conf_high:
        return "HIGH"
    if certainty >= t.conf_med:
        return "MEDIUM"
    return "LOW"


def value_tier_from_ml_edge(abs_edge_prob: float) -> str:
    e = abs(float(abs_edge_prob))
    if e >= 0.07:
        return "HIGH VALUE"
    if e >= 0.035:
        return "MEDIUM VALUE"
    if e >= 0.015:
        return "LOW VALUE"
    return "NO VALUE"


def ml_recommendation(edge_home: float, t: Thresholds) -> str:
    """
    ONLY uses probability edge. Never uses favorite status or spreads.
    edge_home = model_home_prob - market_home_prob
    """
    e = float(edge_home)
    if e >= t.ml_edge_strong:
        return "Model PICK: HOME ML (strong)"
    if e <= -t.ml_edge_strong:
        return "Model PICK: AWAY ML (strong)"
    if e >= t.ml_edge_lean:
        return "Model lean: HOME ML"
    if e <= -t.ml_edge_lean:
        return "Model lean: AWAY ML"
    return "No ML bet (edge too small)"


def ats_recommendation(spread_edge_home_pts: float, t: Thresholds) -> str:
    """
    spread_edge_home_pts = market_home_spread - model_home_spread
      positive => HOME ATS value
      negative => AWAY ATS value
    """
    se = float(spread_edge_home_pts)
    if se >= t.ats_edge_strong_pts:
        return "Model PICK ATS: HOME (strong)"
    if se <= -t.ats_edge_strong_pts:
        return "Model PICK ATS: AWAY (strong)"
    if se >= t.ats_edge_lean_pts:
        return "Model lean ATS: HOME"
    if se <= -t.ats_edge_lean_pts:
        return "Model lean ATS: AWAY"
    return "Too close to call ATS (edge too small)"


def choose_primary(ml_rec: str, ats_rec: str) -> str:
    """
    Clean priority rule:
      1) Strong ATS
      2) Strong ML
      3) Lean ATS
      4) Lean ML
      5) No bet
    Adjust if you want different behavior.
    """
    strong_ats = "PICK ATS" in ats_rec and "(strong)" in ats_rec
    strong_ml = "PICK:" in ml_rec and "(strong)" in ml_rec
    lean_ats = "lean ATS" in ats_rec
    lean_ml = "lean:" in ml_rec

    if strong_ats:
        return ats_rec
    if strong_ml:
        return ml_rec
    if lean_ats:
        return ats_rec
    if lean_ml:
        return ml_rec
    return "NO BET — edges too small"


def explain_ml_ats(
    model_home_prob: float,
    market_home_prob: float,
    home_spread_market: float,
    home_spread_model: float,
    edge_home: float,
    spread_edge_home_pts: float,
) -> str:
    """
    One-line explanation showing why ML and ATS can diverge.
    """
    return (
        f"ML edge={edge_home:+.3f} (model {model_home_prob:.3f} vs mkt {market_home_prob:.3f}) | "
        f"ATS edge={spread_edge_home_pts:+.1f}pts (mkt {home_spread_market:+.1f} vs model {home_spread_model:+.1f})"
    )


def add_recommendations_to_df(
    df: pd.DataFrame,
    thresholds: Thresholds = Thresholds(),
    *,
    # Choose ONE:
    model_spread_home_col: str | None = "model_spread_home",   # if already vegas-style home spread
    model_margin_home_col: str | None = None,                  # if you store HOME-AWAY margin
    home_ml_col: str = "home_ml",
    away_ml_col: str = "away_ml",
    home_spread_col: str = "home_spread",
    model_home_prob_col: str = "model_home_prob",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (df_with_recos, debug_table)

    Requirements in df:
      model_home_prob, home_ml, away_ml, home_spread, and either:
        - model_spread_home (vegas-style home line), OR
        - model_margin_home (expected HOME-AWAY margin)
    """

    out = df.copy()

    # --- Market home prob from odds (or keep your existing market_home_prob if you prefer)
    out["market_home_prob"] = out[home_ml_col].apply(american_to_implied_prob)

    # --- ML edges
    out["edge_home"] = out[model_home_prob_col].astype(float) - out["market_home_prob"].astype(float)
    out["edge_away"] = -out["edge_home"]

    # --- Normalize model spread to vegas-style home spread
    if model_margin_home_col is not None and model_margin_home_col in out.columns:
        out["model_spread_home_norm"] = out[model_margin_home_col].astype(float).apply(model_margin_to_home_spread)
    elif model_spread_home_col is not None and model_spread_home_col in out.columns:
        out["model_spread_home_norm"] = out[model_spread_home_col].astype(float)
    else:
        raise ValueError("Need either model_spread_home_col or model_margin_home_col present in df")

    # --- ATS spread edge in points (HOME ATS value)
    out["spread_edge_home"] = out[home_spread_col].astype(float) - out["model_spread_home_norm"].astype(float)

    # --- Recommendations
    out["ml_recommendation"] = out["edge_home"].apply(lambda e: ml_recommendation(e, thresholds))
    out["spread_recommendation"] = out["spread_edge_home"].apply(lambda se: ats_recommendation(se, thresholds))
    out["primary_recommendation"] = [
        choose_primary(mr, sr) for mr, sr in zip(out["ml_recommendation"], out["spread_recommendation"])
    ]

    # --- Confidence vs Value
    out["confidence"] = out[model_home_prob_col].apply(lambda p: confidence_from_prob(p, thresholds))
    out["value_tier"] = out["edge_home"].abs().apply(value_tier_from_ml_edge)

    # --- Why column
    out["why_bet"] = [
        explain_ml_ats(
            model_home_prob=float(p),
            market_home_prob=float(mp),
            home_spread_market=float(hs),
            home_spread_model=float(ms),
            edge_home=float(eh),
            spread_edge_home_pts=float(se),
        )
        for p, mp, hs, ms, eh, se in zip(
            out[model_home_prob_col],
            out["market_home_prob"],
            out[home_spread_col],
            out["model_spread_home_norm"],
            out["edge_home"],
            out["spread_edge_home"],
        )
    ]

    # --- Debug table: "why ML ≠ ATS"
    debug = out[[
        "date", "home", "away",
        model_home_prob_col, "market_home_prob", "edge_home",
        home_spread_col, "model_spread_home_norm", "spread_edge_home",
        "ml_recommendation", "spread_recommendation", "primary_recommendation",
        "confidence", "value_tier", "why_bet"
    ]].copy()

    return out, debug
