# recommendations.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


@dataclass
class Thresholds:
    # ML
    ml_edge_strong: float = 0.06
    ml_edge_lean: float = 0.035

    # ATS (points/goals edge — used only if your model stores spread_edge_home)
    ats_edge_strong_pts: float = 3.0
    ats_edge_lean_pts: float = 1.5

    # Confidence labels from abs_edge_home
    conf_high: float = 0.18
    conf_med: float = 0.10

    # Totals gating (if totals columns exist)
    total_min_edge_vs_be: float = 0.02
    total_min_edge_pts: float = 3.0


# -----------------------------
# SPORT BET-TYPE PRIORITIES (Priority 1 -> 3)
# Change these if your tracking proves otherwise.
# -----------------------------
SPORT_BET_PRIORITY: Dict[str, List[str]] = {
    # Many NBA models end up better on totals/ATS than ML, but this MUST be validated with your CLV/ROI.
    "nba": ["TOTAL", "ATS", "ML"],
    # NFL often strongest on spreads/totals vs ML for many public models (again: validate).
    "nfl": ["ATS", "TOTAL", "ML"],
    # NHL markets can be tighter; many people end up preferring ML or totals; pick one and let data decide.
    "nhl": ["ML", "TOTAL", "ATS"],
}


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float("nan")


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def american_to_prob(ml: float) -> float:
    ml = float(ml)
    if ml == 0:
        return float("nan")
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)


def no_vig_probs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    hp = american_to_prob(home_ml)
    ap = american_to_prob(away_ml)
    if np.isnan(hp) or np.isnan(ap) or (hp + ap) <= 0:
        return (float("nan"), float("nan"))
    s = hp + ap
    return (hp / s, ap / s)


def _confidence_from_abs_edge(abs_edge: float, th: Thresholds) -> str:
    if np.isnan(abs_edge):
        return "UNKNOWN"
    if abs_edge >= th.conf_high:
        return "HIGH"
    if abs_edge >= th.conf_med:
        return "MEDIUM"
    return "LOW"


def _value_tier(abs_edge: float) -> str:
    if np.isnan(abs_edge):
        return "UNKNOWN"
    if abs_edge >= 0.08:
        return "HIGH VALUE"
    if abs_edge >= 0.04:
        return "MED VALUE"
    if abs_edge >= 0.02:
        return "LOW VALUE"
    return "NO EDGE"


def _ml_pick(model_p: float, market_p: float, th: Thresholds) -> str:
    if np.isnan(model_p) or np.isnan(market_p):
        return "No ML bet (missing market prob)"
    edge = float(model_p - market_p)
    if edge >= th.ml_edge_strong:
        return "Model PICK: HOME ML (strong)"
    if edge >= th.ml_edge_lean:
        return "Model lean: HOME ML"
    if edge <= -th.ml_edge_strong:
        return "Model PICK: AWAY ML (strong)"
    if edge <= -th.ml_edge_lean:
        return "Model lean: AWAY ML"
    return "No ML bet (edge too small)"


def _ats_pick_from_edge_pts(spread_edge_home: float, th: Thresholds) -> str:
    if spread_edge_home is None or np.isnan(spread_edge_home):
        return "No ATS bet (missing spread)"

    if spread_edge_home >= th.ats_edge_strong_pts:
        return "Model PICK ATS: HOME (strong)"
    if spread_edge_home >= th.ats_edge_lean_pts:
        return "Model PICK ATS: HOME (lean)"
    if spread_edge_home <= -th.ats_edge_strong_pts:
        return "Model PICK ATS: AWAY (strong)"
    if spread_edge_home <= -th.ats_edge_lean_pts:
        return "Model PICK ATS: AWAY (lean)"
    return "Too close to call ATS (edge too small)"


def _is_real_pick(s: str) -> bool:
    s = str(s or "")
    return s.startswith("Model PICK")


def _totals_is_real_pick(s: str) -> bool:
    s = str(s or "")
    return s.startswith("Model PICK TOTAL:")


def _primary_with_priority(
    *,
    sport: str,
    ml_reco: str,
    ats_reco: str,
    total_reco: str,
    ml_score: float,
    ats_score: float,
    total_score: float,
) -> Tuple[str, str, float]:
    """
    Choose primary by:
      1) consider ONLY bets that are real picks
      2) within those, follow SPORT_BET_PRIORITY ordering
      3) but require the candidate score to be > 0 (so we don't promote junk)
      4) if nothing qualifies, fall back to ML reco
    Returns: (primary, why, pick_score)
    """
    pr = SPORT_BET_PRIORITY.get(str(sport).lower(), ["TOTAL", "ATS", "ML"])

    # build candidates
    candidates = {
        "ML": (ml_reco, ml_score, _is_real_pick(ml_reco)),
        "ATS": (ats_reco, ats_score, _is_real_pick(ats_reco) and str(ats_reco).startswith("Model PICK ATS:")),
        "TOTAL": (total_reco, total_score, _totals_is_real_pick(total_reco)),
    }

    # pick best by priority first, but only if it's a real pick and score positive
    for k in pr:
        reco, sc, ok = candidates.get(k, ("", -999.0, False))
        if ok and sc is not None and not np.isnan(sc) and float(sc) > 0:
            return (str(reco), f"Primary={k} (score={float(sc):+.3f})", float(sc))

    # fallback: pick whichever real pick has highest score
    best_k = None
    best_sc = -999.0
    best_reco = str(ml_reco)
    for k, (reco, sc, ok) in candidates.items():
        if not ok:
            continue
        if sc is None or np.isnan(sc):
            continue
        if float(sc) > best_sc:
            best_sc = float(sc)
            best_k = k
            best_reco = str(reco)

    if best_k is not None and best_sc > 0:
        return (best_reco, f"Primary={best_k} (score={best_sc:+.3f})", float(best_sc))

    # final fallback
    return (str(ml_reco), f"Primary=ML (score={float(ml_score):+.3f})", float(ml_score) if not np.isnan(ml_score) else -999.0)


def _fmt(x) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.3f}"
    except Exception:
        return "nan"


def add_recommendations_to_df(
    df: pd.DataFrame,
    thresholds: Thresholds = Thresholds(),
    *,
    model_spread_home_col: Optional[str] = "model_spread_home",
    model_margin_home_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds/ensures:
      market_home_prob, edge_home, edge_away
      ml_recommendation, spread_recommendation (if spread edge exists), total_recommendation pass-through
      abs_edge_home, confidence, value_tier
      primary_recommendation, why_primary
      pick_score (numeric, used to filter to 1-3 bets/day)
      why_bet
    """
    out = df.copy()

    # Ensure sport column exists (runner should set it; fallback safe)
    if "sport" not in out.columns:
        out["sport"] = ""

    # Ensure market_home_prob exists if MLs exist
    if "market_home_prob" not in out.columns:
        out["market_home_prob"] = np.nan
    if "home_ml" in out.columns and "away_ml" in out.columns:
        for i in out.index:
            hml = out.loc[i, "home_ml"]
            aml = out.loc[i, "away_ml"]
            try:
                if not pd.isna(hml) and not pd.isna(aml):
                    mh, _ = no_vig_probs(float(hml), float(aml))
                    out.loc[i, "market_home_prob"] = float(mh)
            except Exception:
                continue

    # edge columns
    if "edge_home" not in out.columns:
        out["edge_home"] = np.nan
    if "edge_away" not in out.columns:
        out["edge_away"] = np.nan

    if "model_home_prob" in out.columns and "market_home_prob" in out.columns:
        for i in out.index:
            mp = out.loc[i, "model_home_prob"]
            mk = out.loc[i, "market_home_prob"]
            if pd.isna(mp) or pd.isna(mk):
                continue
            out.loc[i, "edge_home"] = float(mp - mk)
            out.loc[i, "edge_away"] = float(-(mp - mk))

    # ML recommendation
    out["ml_recommendation"] = out.get("ml_recommendation", "")
    if "model_home_prob" in out.columns:
        for i in out.index:
            mp = _safe_float(out.loc[i, "model_home_prob"])
            mk = _safe_float(out.loc[i, "market_home_prob"])
            out.loc[i, "ml_recommendation"] = _ml_pick(mp, mk, thresholds)

    # Spread edge + recommendation (prefer your model's spread_edge_home if present)
    if "spread_edge_home" not in out.columns:
        out["spread_edge_home"] = np.nan

    # If we can compute spread_edge_home, do it; otherwise leave existing
    if model_spread_home_col and model_spread_home_col in out.columns and "home_spread" in out.columns:
        for i in out.index:
            ms = _safe_float(out.loc[i, model_spread_home_col])
            hs = _safe_float(out.loc[i, "home_spread"])
            if np.isnan(ms) or np.isnan(hs):
                continue
            out.loc[i, "spread_edge_home"] = float(hs - ms)

    out["spread_recommendation"] = out.get("spread_recommendation", "")
    for i in out.index:
        se = _safe_float(out.loc[i, "spread_edge_home"])
        out.loc[i, "spread_recommendation"] = _ats_pick_from_edge_pts(se, thresholds)

    # Totals recommendation:
    # - If per-sport models already compute total_recommendation, keep it
    if "total_recommendation" not in out.columns:
        out["total_recommendation"] = ""
    if "total_edge_vs_be" not in out.columns:
        out["total_edge_vs_be"] = np.nan
    if "total_edge_points" not in out.columns:
        out["total_edge_points"] = np.nan

    # Confidence & value tier from abs edge (ML edge)
    if "abs_edge_home" not in out.columns:
        out["abs_edge_home"] = np.nan
    for i in out.index:
        eh = out.loc[i, "edge_home"]
        if not pd.isna(eh):
            out.loc[i, "abs_edge_home"] = float(abs(float(eh)))

    out["confidence"] = out.get("confidence", "UNKNOWN")
    out["value_tier"] = out.get("value_tier", "UNKNOWN")
    for i in out.index:
        ae = _safe_float(out.loc[i, "abs_edge_home"])
        out.loc[i, "confidence"] = _confidence_from_abs_edge(ae, thresholds)
        out.loc[i, "value_tier"] = _value_tier(ae)

    # Scores for filtering & primary:
    # ML score: abs(edge_home)
    # ATS score: abs(spread_edge_home)/10 (puts points on ~0-2 scale)
    # TOTAL score: total_edge_vs_be (already probability-edge vs breakeven)
    if "pick_score" not in out.columns:
        out["pick_score"] = np.nan

    out["primary_recommendation"] = out.get("primary_recommendation", "")
    out["why_primary"] = out.get("why_primary", "")

    for i in out.index:
        sport = str(out.loc[i, "sport"] or "").lower()

        mlr = str(out.loc[i, "ml_recommendation"])
        atr = str(out.loc[i, "spread_recommendation"])
        tor = str(out.loc[i, "total_recommendation"])

        ml_score = _safe_float(out.loc[i, "abs_edge_home"], default=-999.0)

        se = _safe_float(out.loc[i, "spread_edge_home"])
        ats_score = float(abs(se)) / 10.0 if not np.isnan(se) else -999.0

        tev = _safe_float(out.loc[i, "total_edge_vs_be"])
        tep = _safe_float(out.loc[i, "total_edge_points"])
        # If totals edge vs BE missing, don't accidentally promote totals
        total_score = float(tev) if not np.isnan(tev) else -999.0

        primary, why, pscore = _primary_with_priority(
            sport=sport,
            ml_reco=mlr,
            ats_reco=atr,
            total_reco=tor,
            ml_score=ml_score,
            ats_score=ats_score,
            total_score=total_score,
        )

        out.loc[i, "primary_recommendation"] = primary
        out.loc[i, "why_primary"] = why
        out.loc[i, "pick_score"] = float(pscore)

    # why_bet: quick explainer
    out["why_bet"] = out.get("why_bet", "")
    for i in out.index:
        mp = out.loc[i, "model_home_prob"] if "model_home_prob" in out.columns else np.nan
        mk = out.loc[i, "market_home_prob"] if "market_home_prob" in out.columns else np.nan
        eh = out.loc[i, "edge_home"] if "edge_home" in out.columns else np.nan
        se = out.loc[i, "spread_edge_home"] if "spread_edge_home" in out.columns else np.nan
        tor = str(out.loc[i, "total_recommendation"] or "")
        tev = out.loc[i, "total_edge_vs_be"] if "total_edge_vs_be" in out.columns else np.nan
        out.loc[i, "why_bet"] = (
            f"ML edge={_fmt(eh)} (model {_fmt(mp)} vs mkt {_fmt(mk)})"
            + (f" | ATS edge={_fmt(se)}pts" if not pd.isna(se) else "")
            + (f" | TOTAL edge_vs_be={_fmt(tev)}" if tor.startswith("Model PICK TOTAL:") and not pd.isna(tev) else "")
        )

    debug_df = pd.DataFrame()
    return out, debug_df
