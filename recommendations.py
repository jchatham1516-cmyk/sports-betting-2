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

    # ATS (points/goals edge, not prob edge)
    ats_edge_strong_pts: float = 3.0
    ats_edge_lean_pts: float = 1.5

    # Confidence labels from abs_edge_home (or similar)
    conf_high: float = 0.18
    conf_med: float = 0.10


# Sport-specific "what to prefer FIRST if value is good"
# Your requested change: NFL totals first.
SPORT_PRIMARY_ORDER: Dict[str, List[str]] = {
    "nfl": ["TOTAL", "ATS", "ML"],
    "nba": ["ATS", "TOTAL", "ML"],   # reasonable default; you can change
    "nhl": ["ML", "TOTAL", "ATS"],   # reasonable default; you can change
}


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float("nan")


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


def _ats_pick(model_spread_home: float, market_home_spread: float, th: Thresholds) -> str:
    """
    Convention:
      - model_spread_home: negative means home favored by that many
      - market_home_spread: sportsbook home spread (e.g., -6.5)
    """
    if np.isnan(model_spread_home) or np.isnan(market_home_spread):
        return "No ATS bet (missing spread)"

    spread_edge_home = float(market_home_spread - model_spread_home)

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
    return isinstance(s, str) and s.startswith("Model PICK")


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
    sport: str = "nba",
    model_spread_home_col: Optional[str] = "model_spread_home",
    model_margin_home_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds/ensures:
      - market_home_prob (no-vig from MLs if possible)
      - edge_home/edge_away if possible
      - ml_recommendation, spread_recommendation (totals passes through)
      - confidence, value_tier
      - pick_score (unified scoring)
      - primary_recommendation (sport-aware preference)
    Returns: (df, debug_df)
    """
    out = df.copy()
    sport = str(sport or "nba").lower().strip()
    primary_order = SPORT_PRIMARY_ORDER.get(sport, ["TOTAL", "ATS", "ML"])

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

    # Ensure edge columns if possible
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
            mp = float(out.loc[i, "model_home_prob"]) if not pd.isna(out.loc[i, "model_home_prob"]) else float("nan")
            mk = float(out.loc[i, "market_home_prob"]) if not pd.isna(out.loc[i, "market_home_prob"]) else float("nan")
            out.loc[i, "ml_recommendation"] = _ml_pick(mp, mk, thresholds)

    # Spread recommendation (if we have spread + model spread)
    out["spread_recommendation"] = out.get("spread_recommendation", "")
    if model_spread_home_col and model_spread_home_col in out.columns and "home_spread" in out.columns:
        for i in out.index:
            ms = float(out.loc[i, model_spread_home_col]) if not pd.isna(out.loc[i, model_spread_home_col]) else float("nan")
            hs = float(out.loc[i, "home_spread"]) if not pd.isna(out.loc[i, "home_spread"]) else float("nan")
            out.loc[i, "spread_recommendation"] = _ats_pick(ms, hs, thresholds)

        # Helpful numeric edge (pts) for debugging
        if "spread_edge_home" not in out.columns:
            out["spread_edge_home"] = np.nan
        for i in out.index:
            ms = out.loc[i, model_spread_home_col]
            hs = out.loc[i, "home_spread"]
            if pd.isna(ms) or pd.isna(hs):
                continue
            out.loc[i, "spread_edge_home"] = float(hs - ms)

    # Totals recommendation: assume your sport model populates it.
    if "total_recommendation" not in out.columns:
        out["total_recommendation"] = out.get("total_recommendation", "")

    # abs_edge_home / confidence / value_tier
    if "abs_edge_home" not in out.columns:
        out["abs_edge_home"] = np.nan
    for i in out.index:
        eh = out.loc[i, "edge_home"]
        if not pd.isna(eh):
            out.loc[i, "abs_edge_home"] = float(abs(float(eh)))

    out["confidence"] = out.get("confidence", "UNKNOWN")
    out["value_tier"] = out.get("value_tier", "UNKNOWN")
    for i in out.index:
        ae = float(out.loc[i, "abs_edge_home"]) if not pd.isna(out.loc[i, "abs_edge_home"]) else float("nan")
        out.loc[i, "confidence"] = _confidence_from_abs_edge(ae, thresholds)
        out.loc[i, "value_tier"] = _value_tier(ae)

    # --------
    # pick_score: unified numeric scoring used for top-1..3 filtering & primary
    # --------
    if "pick_score" not in out.columns:
        out["pick_score"] = np.nan

    for i in out.index:
        mlr = str(out.loc[i, "ml_recommendation"])
        atr = str(out.loc[i, "spread_recommendation"])
        tor = str(out.loc[i, "total_recommendation"])

        # ML score: abs no-vig prob edge (only if "Model PICK")
        ml_score = -999.0
        if _is_real_pick(mlr):
            ae = out.loc[i, "abs_edge_home"]
            if not pd.isna(ae):
                ml_score = float(ae)

        # ATS score: prefer ats_edge_vs_be if present; else fallback to |spread_edge_home|/10
        ats_score = -999.0
        if _is_real_pick(atr):
            if "ats_edge_vs_be" in out.columns and not pd.isna(out.loc[i, "ats_edge_vs_be"]):
                ats_score = float(out.loc[i, "ats_edge_vs_be"])
            elif "spread_edge_home" in out.columns and not pd.isna(out.loc[i, "spread_edge_home"]):
                ats_score = float(abs(float(out.loc[i, "spread_edge_home"]))) / 10.0

        # TOTAL score: total_edge_vs_be if present
        tot_score = -999.0
        if _is_real_pick(tor):
            if "total_edge_vs_be" in out.columns and not pd.isna(out.loc[i, "total_edge_vs_be"]):
                tot_score = float(out.loc[i, "total_edge_vs_be"])

        # Save best score (used for filtering)
        out.loc[i, "pick_score"] = float(max(ml_score, ats_score, tot_score))

    # --------
    # Primary recommendation (sport-aware preference)
    # Rule:
    #   1) among REAL picks, choose highest score
    #   2) if scores are close/tied, break ties using sport preference order
    # --------
    out["primary_recommendation"] = out.get("primary_recommendation", "")
    out["why_primary"] = out.get("why_primary", "")

    def _score_for(kind: str, row: pd.Series) -> float:
        if kind == "ML":
            if _is_real_pick(str(row.get("ml_recommendation", ""))):
                v = row.get("abs_edge_home", np.nan)
                return float(v) if not pd.isna(v) else -999.0
            return -999.0
        if kind == "ATS":
            if _is_real_pick(str(row.get("spread_recommendation", ""))):
                v = row.get("ats_edge_vs_be", np.nan)
                if not pd.isna(v):
                    return float(v)
                se = row.get("spread_edge_home", np.nan)
                return float(abs(float(se))) / 10.0 if not pd.isna(se) else -999.0
            return -999.0
        if kind == "TOTAL":
            if _is_real_pick(str(row.get("total_recommendation", ""))):
                v = row.get("total_edge_vs_be", np.nan)
                return float(v) if not pd.isna(v) else -999.0
            return -999.0
        return -999.0

    for i in out.index:
        row = out.loc[i]
        scores = {k: _score_for(k, row) for k in ["ML", "ATS", "TOTAL"]}
        best_score = max(scores.values())

        if best_score <= -900:
            # nothing is a real pick; fall back to ML reco string
            out.loc[i, "primary_recommendation"] = str(row.get("ml_recommendation", ""))
            out.loc[i, "why_primary"] = "Primary=NONE (no real pick)"
            continue

        # candidates within epsilon of best_score
        eps = 1e-9
        cands = [k for k, v in scores.items() if v >= best_score - eps]

        # tie-break by sport preference order
        chosen = None
        for pref in primary_order:
            if pref in cands:
                chosen = pref
                break
        if chosen is None:
            chosen = cands[0]

        if chosen == "TOTAL":
            out.loc[i, "primary_recommendation"] = str(row.get("total_recommendation", ""))
            out.loc[i, "why_primary"] = f"Primary=TOTAL (score={scores['TOTAL']:+.3f})"
        elif chosen == "ATS":
            out.loc[i, "primary_recommendation"] = str(row.get("spread_recommendation", ""))
            out.loc[i, "why_primary"] = f"Primary=ATS (score={scores['ATS']:+.3f})"
        else:
            out.loc[i, "primary_recommendation"] = str(row.get("ml_recommendation", ""))
            out.loc[i, "why_primary"] = f"Primary=ML (score={scores['ML']:+.3f})"

    # why_bet quick explainer
    out["why_bet"] = out.get("why_bet", "")
    for i in out.index:
        mp = out.loc[i, "model_home_prob"] if "model_home_prob" in out.columns else np.nan
        mk = out.loc[i, "market_home_prob"] if "market_home_prob" in out.columns else np.nan
        eh = out.loc[i, "edge_home"] if "edge_home" in out.columns else np.nan
        se = out.loc[i, "spread_edge_home"] if "spread_edge_home" in out.columns else np.nan
        tev = out.loc[i, "total_edge_vs_be"] if "total_edge_vs_be" in out.columns else np.nan
        out.loc[i, "why_bet"] = (
            f"ML edge={_fmt(eh)} (model {_fmt(mp)} vs mkt {_fmt(mk)})"
            + (f" | ATS edge={_fmt(se)}pts" if not pd.isna(se) else "")
            + (f" | TOTAL edge_vs_be={_fmt(tev)}" if not pd.isna(tev) else "")
        )

    debug_df = pd.DataFrame()
    return out, debug_df


# OPTIONAL CLV attach (left as-is; requires clv_log.csv and bet_id)
def attach_clv_from_log(
    preds_df: pd.DataFrame,
    *,
    clv_log_path: str = "results/clv_log.csv",
) -> pd.DataFrame:
    if preds_df is None or preds_df.empty:
        return preds_df
    if not os.path.exists(clv_log_path):
        return preds_df
    try:
        clv = pd.read_csv(clv_log_path)
    except Exception:
        return preds_df
    if clv.empty or "bet_id" not in clv.columns:
        return preds_df
    if "bet_id" not in preds_df.columns:
        return preds_df
    keep_cols = [c for c in ["bet_id", "close_price", "close_line", "clv_prob_no_vig", "clv_price_american"] if c in clv.columns]
    if not keep_cols:
        return preds_df
    return preds_df.merge(clv[keep_cols].drop_duplicates("bet_id"), on="bet_id", how="left")
