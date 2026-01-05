# sports/common/bet_rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Betting / sizing rules shared across sports
# -------------------------------------------------------------------

DEFAULT_UNIT_DOLLARS = 10.0

# Value tiers based on absolute edge (prob edge vs market)
TIER_HIGH = 0.08
TIER_MED = 0.04
TIER_LOW = 0.02

# If you want to be stricter/looser globally, change these:
MIN_EV_TO_PLAY = 0.015       # minimum EV per $1 to consider a "PLAY"
MIN_PLAY_EDGE_ABS = MIN_EV_TO_PLAY
MIN_PRIMARY_EDGE_ABS = 0.03   # not enforced here, but useful if you want later


def _to_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def american_to_decimal(odds: float) -> float:
    try:
        odds = float(odds)
    except Exception:
        return float("nan")
    if np.isnan(odds):
        return float("nan")
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return float("nan")


def breakeven_prob_from_american(odds: float) -> float:
    dec = american_to_decimal(odds)
    if np.isnan(dec) or dec <= 1e-12:
        return float("nan")
    return 1.0 / dec


def profit_per_dollar_from_american(odds: float) -> float:
    """Return the profit (not total return) for a $1 stake at given odds."""

    odds = _to_float(odds)
    if not np.isfinite(odds) or odds == 0:
        return float("nan")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def ev_per_dollar(p_win: float, american_odds: float) -> float:
    """Expected profit per $1 staked for a given win prob + price."""

    p_win = _to_float(p_win)
    if not np.isfinite(p_win):
        return float("nan")

    p_win = min(1.0, max(0.0, p_win))
    profit = profit_per_dollar_from_american(american_odds)
    if not np.isfinite(profit):
        return float("nan")

    # stake is $1; lose stake on loss
    return p_win * profit - (1.0 - p_win)


@dataclass
class BetDecision:
    play_pass: str  # "PLAY" or "PASS"
    bet_size: float
    unit_dollars: float
    units: float
    reason: str


def value_tier_from_edge(abs_edge: float) -> str:
    if abs_edge is None or (isinstance(abs_edge, float) and np.isnan(abs_edge)):
        return "UNKNOWN"
    abs_edge = float(abs_edge)
    if abs_edge >= TIER_HIGH:
        return "HIGH VALUE"
    if abs_edge >= TIER_MED:
        return "MEDIUM VALUE"
    if abs_edge >= TIER_LOW:
        return "LOW VALUE"
    return "NO EDGE"


def default_bet_units_from_tier(tier: str) -> float:
    t = (tier or "").upper()
    if "HIGH" in t:
        return 1.0
    if "MED" in t:
        return 0.5
    if "LOW" in t:
        return 0.25
    return 0.0


def decide_play_pass(
    abs_edge: float,
    *,
    min_edge: float = MIN_PLAY_EDGE_ABS,
    unit_dollars: float = DEFAULT_UNIT_DOLLARS,
    tier: Optional[str] = None,
    max_units: float = 1.0,
    reason_prefix: str = "",
) -> BetDecision:
    if abs_edge is None or (isinstance(abs_edge, float) and np.isnan(abs_edge)):
        return BetDecision("PASS", 0.0, float(unit_dollars), 0.0, f"{reason_prefix}missing edge")

    abs_edge = float(abs_edge)

    if abs_edge < float(min_edge):
        return BetDecision("PASS", 0.0, float(unit_dollars), 0.0, f"{reason_prefix}edge<{min_edge:.3f}")

    tier = tier or value_tier_from_edge(abs_edge)
    units = default_bet_units_from_tier(tier)
    units = float(min(units, float(max_units)))

    bet_size = float(units * float(unit_dollars))
    return BetDecision("PLAY", bet_size, float(unit_dollars), units, f"{reason_prefix}edge={abs_edge:.3f} tier={tier}")


def choose_primary_recommendation(
    *,
    ml_reco: str,
    spread_reco: str,
    total_reco: str,
    ml_ev: float,
    ats_ev: float,
    total_ev: float,
) -> Tuple[str, str]:
    """
    Pick the strongest allowed recommendation among ML/ATS/TOTAL using EV.

    Returns:
      (primary_recommendation, why_primary)
    """
    evs = {
        "ML": float(ml_ev) if ml_ev is not None and not np.isnan(ml_ev) else float("-inf"),
        "ATS": float(ats_ev) if ats_ev is not None and not np.isnan(ats_ev) else float("-inf"),
        "TOTAL": float(total_ev) if total_ev is not None and not np.isnan(total_ev) else float("-inf"),
    }

    def _fmt_ev(v: float) -> str:
        if v is None or np.isnan(v):
            return "nan"
        if not np.isfinite(v):
            return "nan"
        return f"{float(v):+.3f}"

    primary = "NONE"
    best_market = max(evs, key=lambda k: evs[k])
    best_ev = evs[best_market]

    if np.isfinite(best_ev):
        primary_map = {"ML": ml_reco, "ATS": spread_reco, "TOTAL": total_reco}
        primary = str(primary_map.get(best_market, ml_reco))
        why = (
            f"Primary={best_market} (EV={_fmt_ev(best_ev)} "
            f"ML={_fmt_ev(evs['ML'])} ATS={_fmt_ev(evs['ATS'])} TOTAL={_fmt_ev(evs['TOTAL'])})"
        )
    else:
        why = (
            "Primary=NONE (no finite EV) "
            f"ML={_fmt_ev(evs['ML'])} ATS={_fmt_ev(evs['ATS'])} TOTAL={_fmt_ev(evs['TOTAL'])}"
        )

    return primary, why


def add_betting_outputs(
    df: pd.DataFrame,
    *,
    unit_dollars: float = DEFAULT_UNIT_DOLLARS,
    min_play_edge_abs: float = MIN_PLAY_EDGE_ABS,
) -> pd.DataFrame:
    """
    Adds standardized columns:
      - primary_recommendation (recomputed so TOTAL can win)
      - why_primary
      - play_pass, bet_size, unit_dollars, units
      - why_bet

    Key fix:
      We recompute PRIMARY here using ML/ATS/TOTAL edges, so totals can actually
      become the recommended bet even if upstream code didn't set it right.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # --- 1) Recompute primary so totals can win ---
    def _safe_num(x):
        try:
            if x is None:
                return np.nan
            if isinstance(x, str) and x.strip() == "":
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _row_primary(r):
        ml_reco = str(r.get("ml_recommendation", ""))
        spread_reco = str(r.get("spread_recommendation", ""))
        total_reco = str(r.get("total_recommendation", ""))

        ml_ev = _safe_num(r.get("ml_ev_best"))
        ats_ev = _safe_num(r.get("ats_ev_best"))
        total_ev = _safe_num(r.get("total_ev_best"))

        primary, why = choose_primary_recommendation(
            ml_reco=ml_reco,
            spread_reco=spread_reco,
            total_reco=total_reco,
            ml_ev=ml_ev if not np.isnan(ml_ev) else np.nan,
            ats_ev=ats_ev,
            total_ev=total_ev,
        )
        return primary, why

    primaries = out.apply(_row_primary, axis=1, result_type="expand")
    out["primary_recommendation"] = primaries[0]
    out["why_primary"] = primaries[1]

    # --- 2) Size play/pass based on the recomputed primary ---
    def _row_abs_edge(r) -> float:
        try:
            primary = str(r.get("primary_recommendation", ""))

            if primary.startswith("Model PICK TOTAL:"):
                v = _safe_num(r.get("total_ev_best"))
                return float(v) if not np.isnan(v) else np.nan

            if primary.startswith("Model PICK ATS:"):
                v = _safe_num(r.get("ats_ev_best"))
                return float(v) if not np.isnan(v) else np.nan

            # ML fallback
            v = _safe_num(r.get("ml_ev_best"))
            return float(v) if not np.isnan(v) else np.nan
        except Exception:
            return np.nan

    abs_edges = out.apply(_row_abs_edge, axis=1)
    tiers = [value_tier_from_edge(x) for x in abs_edges]

    decisions = [
        decide_play_pass(
            x,
            min_edge=min_play_edge_abs,
            unit_dollars=unit_dollars,
            tier=t,
            max_units=1.0,
        )
        for x, t in zip(abs_edges, tiers)
    ]

    # Prefer the model's tier if it exists, but fill missing with our computed one
    if "value_tier" in out.columns:
        out["value_tier"] = out["value_tier"].fillna(pd.Series(tiers, index=out.index))
    else:
        out["value_tier"] = pd.Series(tiers, index=out.index)

    out["play_pass"] = [d.play_pass for d in decisions]
    out["bet_size"] = [d.bet_size for d in decisions]
    out["unit_dollars"] = [d.unit_dollars for d in decisions]
    out["units"] = [d.units for d in decisions]
    out["why_bet"] = [d.reason for d in decisions]

    return out

