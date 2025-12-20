# recommendations.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd


def american_to_implied_prob(ml: float) -> float:
    ml = float(ml)
    if ml == 0:
        return 0.5
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)


@dataclass(frozen=True)
class Thresholds:
    ml_edge_strong: float = 0.06
    ml_edge_lean: float = 0.035

    # Legacy ATS thresholds (points-based, used when NFL ATS gating fields are NOT present)
    ats_edge_strong_pts: float = 3.0
    ats_edge_lean_pts: float = 1.5

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


def ats_recommendation_points_based(spread_edge_home_pts: float, t: Thresholds) -> str:
    """
    Legacy ATS rec (NBA-style): purely points edge.
    Positive spread_edge_home => HOME ATS value
    Negative spread_edge_home => AWAY ATS value
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


def choose_primary_legacy(ml_rec: str, ats_rec: str) -> str:
    """
    Legacy chooser (NBA/NHL style): ATS can win if strong, then ML, then leans.
    """
    strong_ats = ("PICK ATS" in ats_rec) and ("(strong)" in ats_rec)
    strong_ml = ("PICK:" in ml_rec) and ("(strong)" in ml_rec)
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
    return (
        f"ML edge={edge_home:+.3f} (model {model_home_prob:.3f} vs mkt {market_home_prob:.3f}) | "
        f"ATS edge={spread_edge_home_pts:+.1f}pts (mkt {home_spread_market:+.1f} vs model {home_spread_model:+.1f})"
    )


def add_recommendations_to_df(
    df: pd.DataFrame,
    thresholds: Thresholds = Thresholds(),
    *,
    model_spread_home_col: str | None = "model_spread_home",
    model_margin_home_col: str | None = None,
    home_ml_col: str = "home_ml",
    away_ml_col: str = "away_ml",
    home_spread_col: str = "home_spread",
    model_home_prob_col: str = "model_home_prob",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds:
      - market_home_prob (no-vig if both sides exist)
      - edge_home/edge_away
      - model_spread_home_norm
      - spread_edge_home (market - model; + means HOME ATS value)
      - ml_recommendation
      - spread_recommendation
      - primary_recommendation
      - confidence
      - value_tier
      - why_bet

    IMPORTANT NFL FIX:
      If NFL gating fields exist (ats_strength / ats_pass_reason), we DO NOT recompute ATS
      from points edge. We format ATS only if the NFL model says it's a play.
    """
    out = df.copy()

    # -----------------------
    # Market home prob (no-vig)
    # -----------------------
    def market_prob_row(r):
        h = r.get(home_ml_col)
        a = r.get(away_ml_col)
        if pd.notna(h) and pd.notna(a):
            ph = american_to_implied_prob(h)
            pa = american_to_implied_prob(a)
            tot = ph + pa
            return ph / tot if tot > 0 else 0.5
        if pd.notna(h):
            return american_to_implied_prob(h)
        if pd.notna(a):
            return 1.0 - american_to_implied_prob(a)
        return 0.5

    out["market_home_prob"] = out.apply(market_prob_row, axis=1).astype(float)

    out["edge_home"] = out[model_home_prob_col].astype(float) - out["market_home_prob"].astype(float)
    out["edge_away"] = -out["edge_home"]

    # -----------------------
    # Normalize model spread
    # -----------------------
    if model_margin_home_col is not None and model_margin_home_col in out.columns:
        out["model_spread_home_norm"] = (-out[model_margin_home_col].astype(float))
    elif model_spread_home_col is not None and model_spread_home_col in out.columns:
        out["model_spread_home_norm"] = out[model_spread_home_col].astype(float)
    else:
        raise ValueError("Need either model_spread_home_col or model_margin_home_col present in df")

    # ATS edge in points (market - model)
    out["spread_edge_home"] = out[home_spread_col].astype(float) - out["model_spread_home_norm"].astype(float)

    # -----------------------
    # ML rec
    # -----------------------
    out["ml_recommendation"] = out["edge_home"].apply(lambda e: ml_recommendation(e, thresholds))

    # -----------------------
    # ATS rec (NFL-aware)
    # -----------------------
    has_nfl_ats_fields = ("ats_strength" in out.columns) and ("ats_pass_reason" in out.columns)

    if has_nfl_ats_fields:
        # Format ATS from NFL fields ONLY; never recompute from points edge here.
        def nfl_spread_rec_row(r):
            strength = str(r.get("ats_strength", "")).strip().lower()
            pass_reason = str(r.get("ats_pass_reason", "")).strip()
            side = str(r.get("ats_pick_side", "")).strip().upper()

            # Only these strengths mean ATS is actually a bet
            if strength not in {"strong", "medium", "lean"}:
                return f"No ATS bet (gated){': ' + pass_reason if pass_reason else ''}"

            if side not in {"HOME", "AWAY"}:
                return "No ATS bet (missing side)"

            if strength in {"strong", "medium"}:
                return f"Model PICK ATS: {side} ({strength})"
            return f"Model lean ATS: {side}"

        out["spread_recommendation"] = out.apply(nfl_spread_rec_row, axis=1)

        def choose_primary_nfl(r):
            ml_rec = str(r.get("ml_recommendation", ""))
            ats_rec = str(r.get("spread_recommendation", ""))
            strength = str(r.get("ats_strength", "")).strip().lower()

            ats_is_play = strength in {"strong", "medium", "lean"} and not ats_rec.startswith("No ATS bet")

            # If ATS isn't eligible, ATS can never be primary
            if not ats_is_play:
                if "PICK:" in ml_rec or "lean:" in ml_rec:
                    return ml_rec
                return "NO BET — edges too small"

            # Prefer ATS only if ATS is strong; otherwise let strong ML win
            strong_ml = ("PICK:" in ml_rec) and ("(strong)" in ml_rec)

            if strength == "strong":
                return ats_rec
            if strong_ml:
                return ml_rec
            return ats_rec

        out["primary_recommendation"] = out.apply(choose_primary_nfl, axis=1)

    else:
        # Legacy behavior (NBA / NHL / old NFL): compute ATS from points edge
        out["spread_recommendation"] = out["spread_edge_home"].apply(
            lambda se: ats_recommendation_points_based(se, thresholds)
        )
        out["primary_recommendation"] = [
            choose_primary_legacy(mr, sr) for mr, sr in zip(out["ml_recommendation"], out["spread_recommendation"])
        ]

    # -----------------------
    # Confidence / value tier / why
    # -----------------------
    out["confidence"] = out[model_home_prob_col].apply(lambda p: confidence_from_prob(p, thresholds))
    out["value_tier"] = out["edge_home"].abs().apply(value_tier_from_ml_edge)

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

    # Debug view (keep NFL columns if present)
    debug_cols = [
        "date", "home", "away",
        model_home_prob_col, "market_home_prob", "edge_home",
        home_spread_col, "model_spread_home_norm", "spread_edge_home",
        "ml_recommendation", "spread_recommendation", "primary_recommendation",
        "confidence", "value_tier", "why_bet",
    ]
    for c in ["ats_strength", "ats_pass_reason", "ats_edge_vs_be", "ats_pick_side", "ats_pick_prob"]:
        if c in out.columns and c not in debug_cols:
            debug_cols.append(c)

    debug = out[debug_cols].copy()

    return out, debug
