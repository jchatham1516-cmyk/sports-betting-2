# recommendations.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sports.common.bet_rules import _to_float, breakeven_prob_from_american, ev_per_dollar
from sports.common.bet_config import get_sport_bet_config


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
        return "MEDIUM VALUE"
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


def _norm_cdf(x: float) -> float:
    try:
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
    except Exception:
        return float("nan")


def p_over_total(model_total: float, total_line: float, total_sd: float) -> float:
    """Probability that actual total goes over given line using normal model."""

    try:
        z = (float(total_line) - float(model_total)) / float(total_sd)
    except Exception:
        return float("nan")

    return 1.0 - _norm_cdf(z)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for compatibility with older tests/usage)
# ---------------------------------------------------------------------------
def ml_recommendation(edge_home: float, th: Thresholds = Thresholds()) -> str:
    """Return a moneyline recommendation based on the probability edge.

    This is a thin wrapper around the legacy API that operated on the
    probability edge instead of absolute probabilities.
    """

    if edge_home >= th.ml_edge_strong:
        return "Model PICK: HOME ML (strong)"
    if edge_home >= th.ml_edge_lean:
        return "Model lean: HOME ML"
    if edge_home <= -th.ml_edge_strong:
        return "Model PICK: AWAY ML (strong)"
    if edge_home <= -th.ml_edge_lean:
        return "Model lean: AWAY ML"
    return "No ML bet (edge too small)"


def ats_recommendation(spread_edge_home: float, th: Thresholds = Thresholds()) -> str:
    """Return an ATS recommendation using the point-edge perspective."""

    if spread_edge_home >= th.ats_edge_strong_pts:
        return "Model PICK ATS: HOME (strong)"
    if spread_edge_home >= th.ats_edge_lean_pts:
        return "Model PICK ATS: HOME (lean)"
    if spread_edge_home <= -th.ats_edge_strong_pts:
        return "Model PICK ATS: AWAY (strong)"
    if spread_edge_home <= -th.ats_edge_lean_pts:
        return "Model PICK ATS: AWAY (lean)"
    return "Too close to call ATS (edge too small)"


def choose_primary(ml_reco: str, ats_reco: str) -> str:
    """Select which recommendation to prioritize (ATS wins only if strong)."""

    if isinstance(ats_reco, str) and ("Model PICK ATS" in ats_reco) and ("(strong)" in ats_reco):
        return ats_reco
    return ml_reco


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
    config = get_sport_bet_config(sport)

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

    if "model_home_prob_final" in out.columns and "market_home_prob" in out.columns:
        for i in out.index:
            mp = out.loc[i, "model_home_prob_final"]
            mk = out.loc[i, "market_home_prob"]
            if pd.isna(mp) or pd.isna(mk):
                continue
            out.loc[i, "edge_home"] = float(mp - mk)
            out.loc[i, "edge_away"] = float(-(mp - mk))

    # Standardize probability columns
    for col in ("model_home_prob_raw", "model_home_prob_cal", "model_home_prob_final"):
        if col not in out.columns:
            out[col] = np.nan
    fallback_mask = out["model_home_prob_cal"].isna() | out["model_home_prob_final"].isna()
    if "model_home_prob" in out.columns:
        for i in out.index:
            base = out.loc[i, "model_home_prob"]
            if pd.isna(out.loc[i, "model_home_prob_raw"]) and not pd.isna(base):
                out.loc[i, "model_home_prob_raw"] = float(base)
            if pd.isna(out.loc[i, "model_home_prob_cal"]) and not pd.isna(out.loc[i, "model_home_prob_raw"]):
                out.loc[i, "model_home_prob_cal"] = float(out.loc[i, "model_home_prob_raw"])
            if pd.isna(out.loc[i, "model_home_prob_final"]) and not pd.isna(out.loc[i, "model_home_prob_cal"]):
                out.loc[i, "model_home_prob_final"] = float(out.loc[i, "model_home_prob_cal"])

    if "decision_flags" not in out.columns:
        out["decision_flags"] = ""
    if "decision_reason" not in out.columns:
        out["decision_reason"] = ""

    # ML recommendation
    out["ml_recommendation"] = out.get("ml_recommendation", "")
    if "model_home_prob_final" in out.columns:
        for i in out.index:
            mp = (
                float(out.loc[i, "model_home_prob_final"])
                if not pd.isna(out.loc[i, "model_home_prob_final"])
                else float("nan")
            )
            mk = float(out.loc[i, "market_home_prob"]) if not pd.isna(out.loc[i, "market_home_prob"]) else float("nan")
            out.loc[i, "ml_recommendation"] = _ml_pick(mp, mk, thresholds)

    # Spread recommendation (if we have spread + model spread)
    out["spread_recommendation"] = out.get("spread_recommendation", "")
    if model_spread_home_col and model_spread_home_col in out.columns and "home_spread" in out.columns:
        for i in out.index:
            if bool(out.get("ats_gated", pd.Series(False, index=out.index)).loc[i]):
                out.loc[i, "spread_recommendation"] = "No ATS bet (gated): invalid spread model"
                continue
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
        row = out.loc[i]
        abs_edge_home = float(row.get("abs_edge_home")) if not pd.isna(row.get("abs_edge_home")) else float("nan")
        total_edge_vs_be = _to_float(row.get("total_edge_vs_be", np.nan))
        total_edge_goals = _to_float(row.get("total_edge_goals", np.nan))
        spread_edge_home = _to_float(row.get("spread_edge_home", np.nan))
        total_reco = str(row.get("total_recommendation", ""))
        spread_reco = str(row.get("spread_recommendation", ""))

        if total_reco.startswith("Model PICK TOTAL") and np.isfinite(total_edge_vs_be):
            conf_edge = abs(float(total_edge_vs_be))
        elif total_reco.startswith("Model PICK TOTAL") and np.isfinite(total_edge_goals):
            conf_edge = abs(float(total_edge_goals))
        elif spread_reco.startswith("Model PICK ATS") and np.isfinite(spread_edge_home):
            conf_edge = abs(float(spread_edge_home))
        else:
            conf_edge = abs_edge_home

        out.loc[i, "confidence"] = _confidence_from_abs_edge(conf_edge, thresholds)
        out.loc[i, "value_tier"] = _value_tier(conf_edge)

    # --------
    # pick_score + EV scoring: unified numeric scoring used for top-1..3 filtering & primary
    # --------
    if "pick_score" not in out.columns:
        out["pick_score"] = np.nan

    numeric_ev_cols = [
        "ml_ev_best",
        "ats_ev_best",
        "total_ev_best",
        "primary_ev",
    ]
    string_ev_cols = [
        "ml_ev_side",
        "ats_ev_side",
        "total_ev_side",
        "primary_market",
        "primary_side",
    ]

    for c in numeric_ev_cols:
        if c not in out.columns:
            out[c] = np.nan
    for c in string_ev_cols:
        if c not in out.columns:
            out[c] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)

    def _safe_num(x):
        try:
            if x is None:
                return np.nan
            if isinstance(x, str) and x.strip() == "":
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _best_ev(ev_home: float, ev_away: float, home_label: str, away_label: str):
        cands = [(ev_home, home_label), (ev_away, away_label)]
        cands = [(v, s) for v, s in cands if np.isfinite(v)]
        if not cands:
            return np.nan, ""
        return max(cands, key=lambda x: x[0])

    for i in out.index:
        row = out.loc[i]
        flags: List[str] = [f for f in str(row.get("decision_flags", "")).split(",") if f]

        # ML EV (NFL/NBA/NHL): use model_home_prob + odds
        p_home = _to_float(
            row.get(
                "model_home_prob_final",
                row.get("model_home_prob_cal", row.get("model_home_prob_raw", row.get("model_home_prob", np.nan))),
            )
        )
        p_away = 1.0 - p_home if np.isfinite(p_home) else np.nan
        ml_ev_home = ev_per_dollar(p_home, row.get("home_ml"))
        ml_ev_away = ev_per_dollar(p_away, row.get("away_ml"))
        ml_ev, ml_side = _best_ev(ml_ev_home, ml_ev_away, "HOME", "AWAY")

        if bool(fallback_mask.loc[i]) or not np.isfinite(_to_float(row.get("model_home_prob_final", np.nan))):
            if "UNCALIBRATED_FALLBACK" not in flags:
                flags.append("UNCALIBRATED_FALLBACK")
            if "UNCALIBRATED_FALLBACK" not in str(out.loc[i, "decision_reason"]):
                out.loc[i, "decision_reason"] = (
                    f"{out.loc[i, 'decision_reason']} | UNCALIBRATED_FALLBACK"
                    if str(out.loc[i, "decision_reason"]).strip()
                    else "UNCALIBRATED_FALLBACK"
                )

        # ATS EV: use ats_home_cover_prob + spread price
        p_home_cover = _to_float(row.get("ats_home_cover_prob", row.get("p_home_cover", np.nan)))
        p_away_cover = 1.0 - _to_float(p_home_cover) if np.isfinite(_to_float(p_home_cover)) else np.nan
        ats_price = row.get("spread_price")
        if bool(row.get("ats_gated", False)):
            ats_ev = np.nan
            ats_side = ""
            if "ATS_GATED_INVALID_SPREAD" not in flags:
                flags.append("ATS_GATED_INVALID_SPREAD")
            out.loc[i, "decision_reason"] = (
                f"{out.loc[i, 'decision_reason']} | ATS_GATED_INVALID_SPREAD"
                if str(out.loc[i, "decision_reason"]).strip()
                else "ATS_GATED_INVALID_SPREAD"
            )
        else:
            ats_ev_home = ev_per_dollar(p_home_cover, ats_price)
            ats_ev_away = ev_per_dollar(p_away_cover, ats_price)
            ats_ev, ats_side = _best_ev(ats_ev_home, ats_ev_away, "HOME", "AWAY")

        market_type = str(row.get("market_type") or row.get("primary_market") or row.get("market") or "").upper()
        is_spread_market = "SPREAD" in market_type or "ATS" in market_type
        margin_calibrated = row.get("margin_calibrated")
        margin_calibrated = bool(margin_calibrated) if not pd.isna(margin_calibrated) else False
        ats_edge_prob = np.nan
        be_prob = breakeven_prob_from_american(ats_price)
        if np.isfinite(p_home_cover) and np.isfinite(be_prob):
            ats_edge_prob = float(p_home_cover - be_prob)
        if not margin_calibrated:
            required_edge = float(config.min_edge_cal) + float(config.uncalibrated_edge_add)
            if not np.isfinite(ats_edge_prob) or abs(float(ats_edge_prob)) < required_edge:
                ats_ev = np.nan
                ats_side = ""
                if is_spread_market:
                    if "ATS_GATED_UNCALIBRATED_MARGIN" not in flags:
                        flags.append("ATS_GATED_UNCALIBRATED_MARGIN")
                    out.loc[i, "decision_reason"] = (
                        f"{out.loc[i, 'decision_reason']} | ATS_GATED_UNCALIBRATED_MARGIN"
                        if str(out.loc[i, "decision_reason"]).strip()
                        else "ATS_GATED_UNCALIBRATED_MARGIN"
                    )
            else:
                if is_spread_market:
                    if "ATS_UNCALIBRATED_MARGIN" not in flags:
                        flags.append("ATS_UNCALIBRATED_MARGIN")

        # TOTAL EV: compute probability of OVER/UNDER with fallbacks
        p_over = np.nan
        total_sd = _to_float(row.get("total_sd", np.nan))
        if np.isfinite(total_sd) and total_sd > 1e-6:
            model_total = _to_float(row.get("model_total_final", row.get("model_total")))
            z = (_to_float(row.get("total_points")) - model_total) / total_sd
            p_over = 1.0 - _norm_cdf(z)
        else:
            side = str(row.get("total_pick_side", "")).upper().strip()
            prob = _to_float(row.get("total_pick_prob", np.nan))
            if np.isfinite(prob):
                if side == "OVER":
                    p_over = prob
                elif side == "UNDER":
                    p_over = 1.0 - prob

        p_under = 1.0 - _to_float(p_over) if np.isfinite(_to_float(p_over)) else np.nan
        over_ev = ev_per_dollar(p_over, row.get("total_over_price"))
        under_ev = ev_per_dollar(p_under, row.get("total_under_price"))
        total_ev, total_side = _best_ev(over_ev, under_ev, "OVER", "UNDER")

        total_sd_ok = np.isfinite(total_sd) and float(total_sd) >= float(config.total_sd_min)
        model_total_ok = np.isfinite(_to_float(row.get("model_total_final", row.get("model_total"))))
        if not total_sd_ok or not model_total_ok:
            total_ev = np.nan
            total_side = ""
            total_flag = "TOTAL_GATED_LOW_QUALITY"
            if "total_decision_flags" in out.columns:
                existing_total_flags = str(out.loc[i, "total_decision_flags"] or "")
                if total_flag not in existing_total_flags:
                    out.loc[i, "total_decision_flags"] = (
                        f"{existing_total_flags},{total_flag}".strip(",")
                        if existing_total_flags.strip()
                        else total_flag
                    )
            if total_flag not in flags:
                flags.append(total_flag)
            out.loc[i, "decision_reason"] = (
                f"{out.loc[i, 'decision_reason']} | {total_flag}"
                if str(out.loc[i, "decision_reason"]).strip()
                else total_flag
            )

        out.loc[i, "ml_ev_best"] = ml_ev
        out.loc[i, "ml_ev_side"] = ml_side
        out.loc[i, "ats_ev_best"] = ats_ev
        out.loc[i, "ats_ev_side"] = ats_side
        out.loc[i, "total_ev_best"] = total_ev
        out.loc[i, "total_ev_side"] = total_side
        out.loc[i, "decision_flags"] = ",".join(flags)

        # Save best score (used for filtering)
        ev_options = [v for v in [ml_ev, ats_ev, total_ev] if np.isfinite(v)]
        out.loc[i, "pick_score"] = float(max(ev_options)) if ev_options else np.nan

    # --------
    # Primary recommendation: choose market with highest finite EV (NaN ignored)
    # --------
    out["primary_recommendation"] = out.get("primary_recommendation", "")
    out["why_primary"] = out.get("why_primary", "")

    def _fmt_ev(v: float) -> str:
        if v is None or np.isnan(v):
            return "nan"
        if not np.isfinite(v):
            return "nan"
        return f"{float(v):+.3f}"

    for i in out.index:
        row = out.loc[i]
        evs = {
            "ML": _safe_num(row.get("ml_ev_best")),
            "ATS": _safe_num(row.get("ats_ev_best")),
            "TOTAL": _safe_num(row.get("total_ev_best")),
        }

        choices = {k: v for k, v in evs.items() if np.isfinite(v)}
        if not choices:
            out.loc[i, "primary_recommendation"] = ""
            out.loc[i, "why_primary"] = (
                "Primary=NONE (no finite EV) "
                f"ML={_fmt_ev(evs['ML'])} ATS={_fmt_ev(evs['ATS'])} TOTAL={_fmt_ev(evs['TOTAL'])}"
            )
            out.loc[i, "primary_ev"] = np.nan
            out.loc[i, "primary_market"] = "NONE"
            out.loc[i, "primary_side"] = ""
            continue

        chosen = max(choices, key=lambda k: choices[k])
        out.loc[i, "primary_ev"] = choices[chosen]
        out.loc[i, "primary_market"] = chosen

        if chosen == "TOTAL":
            out.loc[i, "primary_recommendation"] = str(row.get("total_recommendation", ""))
            out.loc[i, "primary_side"] = str(row.get("total_ev_side", ""))
        elif chosen == "ATS":
            out.loc[i, "primary_recommendation"] = str(row.get("spread_recommendation", ""))
            out.loc[i, "primary_side"] = str(row.get("ats_ev_side", ""))
        else:
            out.loc[i, "primary_recommendation"] = str(row.get("ml_recommendation", ""))
            out.loc[i, "primary_side"] = str(row.get("ml_ev_side", ""))

        out.loc[i, "why_primary"] = (
            f"Primary={chosen} (EV={_fmt_ev(evs.get(chosen))} "
            f"ML={_fmt_ev(evs['ML'])} ATS={_fmt_ev(evs['ATS'])} TOTAL={_fmt_ev(evs['TOTAL'])})"
        )

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
