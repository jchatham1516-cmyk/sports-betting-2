# sports/common/bet_rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Betting / sizing rules shared across sports
# -------------------------------------------------------------------

# Default stake sizing assumptions
DEFAULT_UNIT_DOLLARS = 10.0

# Value tiers based on absolute edge (prob edge vs market)
TIER_HIGH = 0.08
TIER_MED = 0.04
TIER_LOW = 0.02

# If you want to be stricter/looser globally, change these:
MIN_PLAY_EDGE_ABS = 0.02  # minimum absolute edge to consider a "PLAY"
MIN_PRIMARY_EDGE_ABS = 0.04  # primary recommendation should usually exceed this


@dataclass
class BetDecision:
    play_pass: str  # "PLAY" or "PASS"
    bet_size: float  # dollars (or arbitrary)
    unit_dollars: float
    units: float
    reason: str


def value_tier_from_edge(abs_edge: float) -> str:
    """
    abs_edge is absolute probability edge (e.g., 0.06 = 6%).
    """
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
    """
    Map a tier label to default unit sizing.
    """
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
    """
    Generic play/pass + sizing.
    - abs_edge: absolute probability edge vs market (e.g., 0.05)
    - min_edge: minimum edge to PLAY
    - tier: optional precomputed tier; if None we compute from abs_edge
    - max_units: cap sizing
    """
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
    # Default = ML
    best_edge = float(ml_edge_abs) if ml_edge_abs is not None and not np.isnan(ml_edge_abs) else -999.0
    primary = str(ml_reco)
    why = f"Primary=ML (abs_edge={best_edge:+.3f})" if best_edge > -900 else "Primary=ML (missing edge)"

    # ATS only counts if it's an actual pick string
    ats_ok = isinstance(spread_reco, str) and spread_reco.startswith("Model PICK ATS:")
    ats_val = float(ats_edge_vs_be) if ats_ok and ats_edge_vs_be is not None and not np.isnan(ats_edge_vs_be) else -999.0
    if ats_val > best_edge:
        best_edge = ats_val
        primary = str(spread_reco)
        why = f"Primary=ATS (edge_vs_be={ats_val:+.3f})"

    # TOTAL only counts if it's an actual pick string
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
      - play_pass, bet_size, unit_dollars, units
    using the best available "edge" signal (prefers primary edge when present).

    This function is intentionally conservative: if the row doesn't have a clear
    edge metric, it will PASS.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Prefer the model's "why_primary" / "primary_recommendation" if present,
    # but size based on the most relevant edge:
    # - if primary is TOTAL -> use total_edge_vs_be (or total_edge_points fallback)
    # - if primary is ATS -> use ats_edge_vs_be
    # - else -> use abs(edge_home) (ML)
    def _row_abs_edge(r) -> float:
        try:
            # if we have primary strings, route based on them
            primary = str(r.get("primary_recommendation", ""))
            if primary.startswith("Model PICK TOTAL:"):
                v = r.get("total_edge_vs_be", np.nan)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = r.get("total_edge_points", np.nan)
                    # total_edge_points is in points; convert roughly to prob-edge
                    # using a soft scale (not perfect, but avoids always PASS)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        v = float(abs(v)) * 0.02
                return float(abs(v)) if v is not None else np.nan

            if primary.startswith("Model PICK ATS:"):
                v = r.get("ats_edge_vs_be", np.nan)
                return float(abs(v)) if v is not None else np.nan

            # ML fallback
            v = r.get("edge_home", np.nan)
            return float(abs(v)) if v is not None else np.nan
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

    out["value_tier"] = out.get("value_tier", pd.Series(tiers, index=out.index))
    out["play_pass"] = [d.play_pass for d in decisions]
    out["bet_size"] = [d.bet_size for d in decisions]
    out["unit_dollars"] = [d.unit_dollars for d in decisions]
    out["units"] = [d.units for d in decisions]
    out["why_bet"] = [d.reason for d in decisions]

    return out
