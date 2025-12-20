# recommendations.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import pandas as pd


# -----------------------
# Odds helpers
# -----------------------
def american_to_implied_prob(ml: float) -> float:
    ml = float(ml)
    if ml == 0:
        return 0.5
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)


def breakeven_prob_from_american(price: float) -> float:
    """
    Probability needed to break even at American odds.
    -110 => 0.523809...
    """
    try:
        price = float(price)
        if price == 0:
            return 0.5
        if price < 0:
            return (-price) / ((-price) + 100.0)
        return 100.0 / (price + 100.0)
    except Exception:
        return 0.5238095238


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cover_prob_from_edge(spread_edge_pts: float, sd_pts: float) -> float:
    """
    Approx P(home covers) given spread edge in points, using Normal(sd_pts).
    spread_edge_pts = market_home_spread - model_home_spread
      + => home ATS value
      - => away ATS value
    """
    try:
        se = float(spread_edge_pts)
        sd = float(sd_pts)
        if sd <= 0:
            return float("nan")
        z = se / sd
        p = _phi(z)
        return max(0.001, min(0.999, float(p)))
    except Exception:
        return float("nan")


# -----------------------
# Tunables
# -----------------------
@dataclass(frozen=True)
class Thresholds:
    # ML thresholds (prob edge)
    ml_edge_strong: float = 0.06
    ml_edge_lean: float = 0.035

    # NEW: ATS gating for NON-NFL sports (probability edge vs breakeven)
    # This is the key change that stops "ATS every game".
    ats_min_edge_vs_be: float = 0.035     # must clear breakeven by 3.5%
    ats_strong_edge_vs_be: float = 0.060  # "strong" if 6% above breakeven

    # SD for cover-prob approximation (NBA ~ 11-13)
    ats_sd_pts_default: float = 12.5

    # Optional: cap ATS picks per slate for non-NFL too
    max_non_nfl_ats_plays: int | None = 3

    # Confidence
    conf_high: float = 0.18
    conf_med: float = 0.10


# -----------------------
# Labels
# -----------------------
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


def choose_primary_legacy(ml_rec: str, ats_rec: str) -> str:
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
    return "NO BET - edges too small"


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


# -----------------------
# Main
# -----------------------
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

    out["spread_edge_home"] = out[home_spread_col].astype(float) - out["model_spread_home_norm"].astype(float)

    # -----------------------
    # ML recommendation
    # -----------------------
    out["ml_recommendation"] = out["edge_home"].apply(lambda e: ml_recommendation(e, thresholds))

    # -----------------------
    # ATS recommendation
    # NFL: keep nfl model's gating if columns exist.
    # Non-NFL: use probability vs breakeven gating.
    # -----------------------
    has_nfl_ats_fields = ("ats_strength" in out.columns) and ("ats_pass_reason" in out.columns)

    DEFAULT_SPREAD_PRICE = -110.0

    if has_nfl_ats_fields:
        # Trust nfl model's decision; don't overwrite.
        def nfl_spread_rec_row(r):
            strength = str(r.get("ats_strength", "")).strip().lower()
            pass_reason = str(r.get("ats_pass_reason", "")).strip()
            side = str(r.get("ats_pick_side", "")).strip().upper()

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
            if not ats_is_play:
                if "PICK:" in ml_rec or "lean:" in ml_rec:
                    return ml_rec
                return "NO BET - edges too small"

            if strength == "strong":
                return ats_rec

            strong_ml = ("PICK:" in ml_rec) and ("(strong)" in ml_rec)
            return ml_rec if strong_ml else ats_rec

        out["primary_recommendation"] = out.apply(choose_primary_nfl, axis=1)

    else:
        # Non-NFL ATS gating (NBA / NHL)
        def non_nfl_ats_row(r):
            se = r.get("spread_edge_home")
            if pd.isna(se):
                return ("Too close to call ATS (edge too small)", float("nan"))

            # Use spread_price if present; else default -110
            price = r.get("spread_price", DEFAULT_SPREAD_PRICE)
            try:
                price = float(price)
            except Exception:
                price = DEFAULT_SPREAD_PRICE

            be = breakeven_prob_from_american(price)
            p_home_cover = cover_prob_from_edge(float(se), sd_pts=thresholds.ats_sd_pts_default)
            if pd.isna(p_home_cover):
                return ("Too close to call ATS (edge too small)", float("nan"))

            p_away_cover = 1.0 - p_home_cover
            if p_home_cover >= p_away_cover:
                side = "HOME"
                p_win = p_home_cover
            else:
                side = "AWAY"
                p_win = p_away_cover

            edge_vs_be = float(p_win - be)

            # Gate
            if edge_vs_be < thresholds.ats_min_edge_vs_be:
                return ("Too close to call ATS (edge too small)", edge_vs_be)

            strength = "strong" if edge_vs_be >= thresholds.ats_strong_edge_vs_be else "medium"
            return (f"Model PICK ATS: {side} ({strength})", edge_vs_be)

        tmp = out.apply(non_nfl_ats_row, axis=1, result_type="expand")
        out["spread_recommendation"] = tmp[0]
        out["ats_edge_vs_be"] = tmp[1]

        # Optional: keep only top N ATS plays per slate
        if thresholds.max_non_nfl_ats_plays is not None and thresholds.max_non_nfl_ats_plays > 0:
            is_pick = out["spread_recommendation"].astype(str).str.contains("Model PICK ATS:", na=False)
            out["__ats_rank"] = out["ats_edge_vs_be"].where(is_pick, -999.0)

            top_idx = out.sort_values("__ats_rank", ascending=False).head(thresholds.max_non_nfl_ats_plays).index
            keep = set(top_idx.tolist())

            for i in out.index:
                if is_pick.loc[i] and i not in keep:
                    out.loc[i, "spread_recommendation"] = "No ATS bet (top-N filter)"
            out.drop(columns=["__ats_rank"], inplace=True, errors="ignore")

        out["primary_recommendation"] = [
            choose_primary_legacy(mr, sr) for mr, sr in zip(out["ml_recommendation"], out["spread_recommendation"])
        ]

    # -----------------------
    # Confidence / Value / Why
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

    debug_cols = [
        "date", "home", "away",
        model_home_prob_col, "market_home_prob", "edge_home",
        home_spread_col, "model_spread_home_norm", "spread_edge_home",
        "ml_recommendation", "spread_recommendation", "primary_recommendation",
        "confidence", "value_tier", "why_bet",
    ]
    for c in ["ats_edge_vs_be", "ats_strength", "ats_pass_reason", "ats_pick_side", "ats_pick_prob"]:
        if c in out.columns and c not in debug_cols:
            debug_cols.append(c)

    debug = out[debug_cols].copy()
    return out, debug
