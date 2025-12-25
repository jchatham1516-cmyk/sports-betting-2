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
MIN_PLAY_EDGE_ABS = 0.02      # minimum absolute edge to consider a "PLAY"
MIN_PRIMARY_EDGE_ABS = 0.04   # not enforced here, but useful if you want later


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
    ml_edge_abs: float,
    ats_edge_vs_be: float,
    total_edge_vs_be: float,
) -> Tuple[str, str]:
    """
    Pick the strongest allowed recommendation among ML/ATS/TOTAL.

    Returns:
      (primary_recommendation, why_primary)
    """
    best_edge = float(ml_edge_abs) if ml_edge_abs is not None and not np.isnan(ml_edge_abs) else -999.0
    primary = str(ml_reco)
    why = f"Primary=ML (abs_edge={best_edge:+.3f})" if best_edge > -900 else "Primary=ML (missing edge)"

    ats_ok = isinstance(spread_reco, str) and spread_reco.startswith("Model PICK ATS:")
    ats_val = float(ats_edge_vs_be) if ats_ok and ats_edge_vs_be is not None and not np.isnan(ats_edge_vs_be) else -999.0
    if ats_val > best_edge:
        best_edge = ats_val
        primary = str(spread_reco)
        why = f"Primary=ATS (edge_vs_be={ats_val:+.3f})"

    tot_ok = isinstance(total_reco, str) and total_reco.startswith("Model PICK TOTAL:")
    tot_val = float(total_edge_vs_be) if tot_ok and total_edge_vs_be is not None and not np.isnan(total_edge_vs_be) else -999.0
    if tot_val > best_edge:
        best_edge = tot_val
        primary = str(total_reco)
        why = f"Primary=TOTAL (edge_vs_be={tot_val:+.3f})"

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

        ml_edge_abs = abs(_safe_num(r.get("edge_home")))
        ats_edge_vs_be = _safe_num(r.get("ats_edge_vs_be"))
        total_edge_vs_be = _safe_num(r.get("total_edge_vs_be"))

        primary, why = choose_primary_recommendation(
            ml_reco=ml_reco,
            spread_reco=spread_reco,
            total_reco=total_reco,
            ml_edge_abs=ml_edge_abs if not np.isnan(ml_edge_abs) else np.nan,
            ats_edge_vs_be=ats_edge_vs_be,
            total_edge_vs_be=total_edge_vs_be,
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
                v = _safe_num(r.get("total_edge_vs_be"))
                if np.isnan(v):
                    # fallback: convert points edge -> soft probability edge proxy
                    pts = _safe_num(r.get("total_edge_points"))
                    if not np.isnan(pts):
                        v = abs(float(pts)) * 0.02
                return float(abs(v)) if not np.isnan(v) else np.nan

            if primary.startswith("Model PICK ATS:"):
                v = _safe_num(r.get("ats_edge_vs_be"))
                return float(abs(v)) if not np.isnan(v) else np.nan

            # ML fallback
            v = _safe_num(r.get("edge_home"))
            return float(abs(v)) if not np.isnan(v) else np.nan
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

