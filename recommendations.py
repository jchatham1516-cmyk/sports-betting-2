# recommendations.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.clv_tracker import (
    american_to_prob,
    make_bet_id,
    markets_to_stop,
    MarketStopRule,
)


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


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def no_vig_probs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    """
    Classic no-vig from two american prices.
    """
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
    spread_edge_home = market_home_spread - model_spread_home
      Positive => HOME ATS value, Negative => AWAY ATS value
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


def _infer_bet_fields(row: pd.Series) -> Tuple[str, str, float, float]:
    """
    Return market, side, line, price based on primary_recommendation (fallbacks).
    market: "ML"|"ATS"|"TOTAL"
    side: "HOME"|"AWAY"|"OVER"|"UNDER"
    """
    primary = str(row.get("primary_recommendation", "") or "")
    mlr = str(row.get("ml_recommendation", "") or "")
    sr = str(row.get("spread_recommendation", "") or "")
    tr = str(row.get("total_recommendation", "") or "")

    s = primary
    if not s.startswith("Model"):
        if mlr.startswith("Model"):
            s = mlr
        elif sr.startswith("Model"):
            s = sr
        elif tr.startswith("Model"):
            s = tr

    # TOTAL
    if "TOTAL" in s:
        market = "TOTAL"
        line = _safe_float(row.get("total_points"))
        if "OVER" in s:
            return market, "OVER", line, _safe_float(row.get("total_over_price"))
        if "UNDER" in s:
            return market, "UNDER", line, _safe_float(row.get("total_under_price"))
        return market, "NONE", line, np.nan

    # ATS
    if "ATS" in s:
        market = "ATS"
        hs = _safe_float(row.get("home_spread"))
        price = _safe_float(row.get("spread_price"))
        if "HOME" in s:
            return market, "HOME", hs, price
        if "AWAY" in s:
            return market, "AWAY", (-hs if not np.isnan(hs) else np.nan), price
        return market, "NONE", np.nan, price

    # ML
    market = "ML"
    if "AWAY" in s:
        return market, "AWAY", np.nan, _safe_float(row.get("away_ml"))
    return market, "HOME", np.nan, _safe_float(row.get("home_ml"))


def add_recommendations_to_df(
    df: pd.DataFrame,
    thresholds: Thresholds = Thresholds(),
    *,
    sport: Optional[str] = None,                 # NEW (recommended for bet_id)
    game_date: Optional[str] = None,             # NEW (recommended for bet_id)
    model_spread_home_col: Optional[str] = "model_spread_home",
    model_margin_home_col: Optional[str] = None,
    clv_log_path: str = "results/clv_log.csv",
    use_clv_gating: bool = True,
    clv_stop_rule: MarketStopRule = MarketStopRule(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds:
      - market_home_prob (no-vig from MLs if possible)
      - edge_home/edge_away
      - ml_recommendation, spread_recommendation
      - bet_id + bet fields for CLV tracking
      - (optional) joins CLV from clv_log.csv and gates markets with consistently bad CLV

    Returns: (out_df, debug_df)
    """
    out = df.copy()

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

    # Ensure edge columns
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

    # Spread recommendation + numeric spread_edge_home
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

    # Totals recommendation: leave to sport models (pass-through)
    if "total_recommendation" not in out.columns:
        out["total_recommendation"] = out.get("total_recommendation", "")

    # Confidence & value tier from abs edge
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

    # Primary recommendation score logic (ML vs ATS vs TOTAL)
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

    # -----------------------------
    # CLV integration (NEW)
    # -----------------------------
    # add bet fields
    out["bet_market"] = out.get("bet_market", "")
    out["bet_side"] = out.get("bet_side", "")
    out["bet_line"] = out.get("bet_line", np.nan)
    out["bet_price"] = out.get("bet_price", np.nan)
    out["bet_id"] = out.get("bet_id", "")

    # market stop switches from CLV history
    stop_map = {"ML": False, "ATS": False, "TOTAL": False}
    if use_clv_gating and os.path.exists(clv_log_path):
        try:
            stop_map = markets_to_stop(clv_log_path=clv_log_path, rule=clv_stop_rule)
        except Exception:
            stop_map = {"ML": False, "ATS": False, "TOTAL": False}

    for i in out.index:
        market, side, line, price = _infer_bet_fields(out.loc[i])

        out.loc[i, "bet_market"] = market
        out.loc[i, "bet_side"] = side
        out.loc[i, "bet_line"] = (np.nan if np.isnan(_safe_float(line)) else float(line))
        out.loc[i, "bet_price"] = (np.nan if np.isnan(_safe_float(price)) else float(price))

        if sport and game_date and ("home" in out.columns) and ("away" in out.columns):
            out.loc[i, "bet_id"] = make_bet_id(
                sport=str(sport),
                game_date=str(game_date),
                home=str(out.loc[i, "home"]),
                away=str(out.loc[i, "away"]),
                market=str(market),
                side=str(side),
                line=None if np.isnan(_safe_float(line)) else float(line),
            )

        # Gate by CLV market health (only gates "Model PICK..." style recos)
        primary = str(out.loc[i, "primary_recommendation"])
        if use_clv_gating and primary.startswith("Model"):
            if stop_map.get(market, False):
                out.loc[i, "primary_recommendation"] = "NO BET (market failing CLV)"
                out.loc[i, "why_primary"] = f"CLV gate: STOP {market} (see results/clv_log.csv)"
                # don't lie about value tier; mark clearly
                out.loc[i, "value_tier"] = "CLV STOP"

    # Join CLV fields (if bet_id exists and log exists)
    if os.path.exists(clv_log_path) and "bet_id" in out.columns:
        try:
            clv = pd.read_csv(clv_log_path)
            if not clv.empty and "bet_id" in clv.columns:
                keep = [
                    c
                    for c in [
                        "bet_id",
                        "close_price",
                        "close_line",
                        "clv_imp_prob",
                        "clv_line",
                        "ts_open_utc",
                        "ts_close_utc",
                    ]
                    if c in clv.columns
                ]
                if keep:
                    out = out.merge(clv[keep].drop_duplicates("bet_id"), on="bet_id", how="left")
        except Exception:
            pass

    debug_df = pd.DataFrame()
    return out, debug_df


def _fmt(x) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "nan"
        return f"{float(x):.3f}"
    except Exception:
        return "nan"
