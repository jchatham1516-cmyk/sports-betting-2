# recommendations.py
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Thresholds:
    # ML
    ml_edge_strong: float = 0.06
    ml_edge_lean: float = 0.035

    # ATS
    ats_edge_strong_pts: float = 3.0
    ats_edge_lean_pts: float = 1.5

    # Confidence labels from abs_edge_home (or similar)
    conf_high: float = 0.18
    conf_med: float = 0.10


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
    out = df.copy()

    # market_home_prob from MLs if needed
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

    # edges
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

    # Spread recommendation
    out["spread_recommendation"] = out.get("spread_recommendation", "")
    if model_spread_home_col and model_spread_home_col in out.columns and "home_spread" in out.columns:
        for i in out.index:
            ms = float(out.loc[i, model_spread_home_col]) if not pd.isna(out.loc[i, model_spread_home_col]) else float("nan")
            hs = float(out.loc[i, "home_spread"]) if not pd.isna(out.loc[i, "home_spread"]) else float("nan")
            out.loc[i, "spread_recommendation"] = _ats_pick(ms, hs, thresholds)

        if "spread_edge_home" not in out.columns:
            out["spread_edge_home"] = np.nan
        for i in out.index:
            ms = out.loc[i, model_spread_home_col]
            hs = out.loc[i, "home_spread"]
            if pd.isna(ms) or pd.isna(hs):
                continue
            out.loc[i, "spread_edge_home"] = float(hs - ms)

    # Totals recommendation: pass-through (models fill total_* fields)
    if "total_recommendation" not in out.columns:
        out["total_recommendation"] = out.get("total_recommendation", "")

    # Confidence & value tier
    out["abs_edge_home"] = out.get("abs_edge_home", np.nan)
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

    # Primary
    out["primary_recommendation"] = out.get("primary_recommendation", "")
    out["why_primary"] = out.get("why_primary", "")

    for i in out.index:
        mlr = str(out.loc[i, "ml_recommendation"])
        atr = str(out.loc[i, "spread_recommendation"])
        tor = str(out.loc[i, "total_recommendation"])

        ml_score = float(out.loc[i, "abs_edge_home"]) if not pd.isna(out.loc[i, "abs_edge_home"]) else -999.0

        ats_score = -999.0
        if atr.startswith("Model PICK ATS:"):
            se = out.loc[i, "spread_edge_home"] if "spread_edge_home" in out.columns else np.nan
            if not pd.isna(se):
                ats_score = float(abs(float(se))) / 10.0

        tot_score = -999.0
        if tor.startswith("Model PICK TOTAL:") and "total_edge_vs_be" in out.columns:
            tev = out.loc[i, "total_edge_vs_be"]
            if not pd.isna(tev):
                tot_score = float(tev)

        primary = mlr
        why = f"Primary=ML (score={ml_score:+.3f})"
        best = ml_score

        if ats_score > best:
            best = ats_score
            primary = atr
            why = f"Primary=ATS (score={ats_score:+.3f})"

        if tot_score > best:
            best = tot_score
            primary = tor
            why = f"Primary=TOTAL (edge_vs_be={tot_score:+.3f})"

        out.loc[i, "primary_recommendation"] = primary
        out.loc[i, "why_primary"] = why

    # why_bet
    out["why_bet"] = out.get("why_bet", "")
    for i in out.index:
        mp = out.loc[i, "model_home_prob"] if "model_home_prob" in out.columns else np.nan
        mk = out.loc[i, "market_home_prob"] if "market_home_prob" in out.columns else np.nan
        eh = out.loc[i, "edge_home"] if "edge_home" in out.columns else np.nan
        se = out.loc[i, "spread_edge_home"] if "spread_edge_home" in out.columns else np.nan
        out.loc[i, "why_bet"] = (
            f"ML edge={_fmt(eh)} (model {_fmt(mp)} vs mkt {_fmt(mk)})"
            + (f" | ATS edge={_fmt(se)}pts" if not pd.isna(se) else "")
        )

    # A unified score column the runner can use for top-N
    # (prefer primary type score; safe if missing)
    out["pick_score"] = -999.0
    for i in out.index:
        prim = str(out.loc[i, "primary_recommendation"])
        if prim.startswith("Model PICK TOTAL:"):
            tev = out.loc[i, "total_edge_vs_be"] if "total_edge_vs_be" in out.columns else np.nan
            if not pd.isna(tev):
                out.loc[i, "pick_score"] = float(tev)
        elif prim.startswith("Model PICK ATS:"):
            se = out.loc[i, "spread_edge_home"] if "spread_edge_home" in out.columns else np.nan
            if not pd.isna(se):
                out.loc[i, "pick_score"] = float(abs(float(se))) / 10.0
        else:
            ae = out.loc[i, "abs_edge_home"]
            if not pd.isna(ae):
                out.loc[i, "pick_score"] = float(ae)

    debug_df = pd.DataFrame()
    return out, debug_df


def attach_clv_from_log(
    preds_df: pd.DataFrame,
    *,
    clv_log_path: str = "results/clv_log.csv",
) -> pd.DataFrame:
    """
    Left-join predictions with CLV log if present.
    Adds: close_price, close_line, clv_prob_no_vig, clv_price_american
    """
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

    out = preds_df.merge(clv[keep_cols].drop_duplicates("bet_id"), on="bet_id", how="left")
    return out
