# recommendations.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sports.common.bet_rules import _to_float, breakeven_prob_from_american, ev_per_dollar


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

        # ML EV (NFL/NBA/NHL): use model_home_prob + odds
        p_home = _to_float(row.get("model_home_prob", np.nan))
        p_away = 1.0 - p_home if np.isfinite(p_home) else np.nan
        ml_ev_home = ev_per_dollar(p_home, row.get("home_ml"))
        ml_ev_away = ev_per_dollar(p_away, row.get("away_ml"))
        ml_ev, ml_side = _best_ev(ml_ev_home, ml_ev_away, "HOME", "AWAY")

        # ATS EV: use ats_home_cover_prob + spread price
        p_home_cover = _to_float(row.get("ats_home_cover_prob", row.get("p_home_cover", np.nan)))
        p_away_cover = 1.0 - _to_float(p_home_cover) if np.isfinite(_to_float(p_home_cover)) else np.nan
        ats_price = row.get("spread_price")
        ats_ev_home = ev_per_dollar(p_home_cover, ats_price)
        ats_ev_away = ev_per_dollar(p_away_cover, ats_price)
        ats_ev, ats_side = _best_ev(ats_ev_home, ats_ev_away, "HOME", "AWAY")

        # TOTAL EV: compute probability of OVER/UNDER with fallbacks
        p_over = np.nan
        total_sd = _to_float(row.get("total_sd", np.nan))
        if np.isfinite(total_sd) and total_sd > 1e-6:
            z = (_to_float(row.get("total_points")) - _to_float(row.get("model_total"))) / total_sd
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

        out.loc[i, "ml_ev_best"] = ml_ev
        out.loc[i, "ml_ev_side"] = ml_side
        out.loc[i, "ats_ev_best"] = ats_ev
        out.loc[i, "ats_ev_side"] = ats_side
        out.loc[i, "total_ev_best"] = total_ev
        out.loc[i, "total_ev_side"] = total_side

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
