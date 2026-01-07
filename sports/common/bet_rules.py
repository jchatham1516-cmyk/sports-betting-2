# sports/common/bet_rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.bankroll import (
    bet_size_flat,
    bet_size_kelly_ml,
)
from sports.common.util import safe_float

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
MIN_SANITY_EDGE_ABS = 0.03
MIN_EV_OVERRIDE = 0.02
MIN_EV_OVERRIDE_EDGE = 0.02


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


@dataclass
class DecisionSettings:
    min_play_edge_abs: float = MIN_PLAY_EDGE_ABS
    min_primary_edge_abs: float = MIN_PRIMARY_EDGE_ABS
    min_sanity_edge_abs: float = MIN_SANITY_EDGE_ABS
    min_ev_override: float = MIN_EV_OVERRIDE
    min_ev_override_edge: float = MIN_EV_OVERRIDE_EDGE
    longshot_cutoff: float = 500.0
    favorite_extreme_cutoff: float = -800.0
    max_disagreement: float = 0.20
    max_units: float = 1.0
    max_units_sanity: float = 0.25
    min_play_units: float = 0.25
    flat_pct: float = 0.04
    sizing_mode: str = "flat"  # "flat" or "kelly"
    kelly_mult: float = 0.5
    kelly_max_pct: float = 0.03


@dataclass
class DecisionOutcome:
    play_pass: str
    bet_size: float
    unit_dollars: float
    units: float
    reason: str
    decision_flags: str
    decision_reason: str
    raw_units: float
    final_units: float
    p_model_used: float
    p_market_used: float
    abs_edge_used: float


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


def _probabilities_for_primary(row: pd.Series, primary: str) -> Tuple[float, float]:
    primary_upper = (primary or "").upper()
    model_home = safe_float(row.get("model_home_prob"))
    market_home = safe_float(row.get("market_home_prob"))

    if "AWAY ML" in primary_upper:
        model = None if model_home is None else 1.0 - float(model_home)
        market = None if market_home is None else 1.0 - float(market_home)
        return model, market

    return model_home, market_home


def _abs_edge_from_row(row: pd.Series, primary: str, p_model: float, p_market: float) -> float:
    try:
        primary_upper = (primary or "").upper()
        if "TOTAL" in primary_upper:
            total_edge_vs_be = safe_float(row.get("total_edge_vs_be"))
            if total_edge_vs_be is not None and not np.isnan(total_edge_vs_be):
                return abs(float(total_edge_vs_be))
            total_edge_goals = safe_float(row.get("total_edge_goals"))
            if total_edge_goals is not None and not np.isnan(total_edge_goals):
                return abs(float(total_edge_goals))
        if "ATS" in primary_upper:
            spread_edge_home = safe_float(row.get("spread_edge_home"))
            if spread_edge_home is not None and not np.isnan(spread_edge_home):
                return abs(float(spread_edge_home))
        if p_model is not None and p_market is not None:
            return abs(float(p_model) - float(p_market))
        if "abs_edge_home" in row:
            v = safe_float(row.get("abs_edge_home"))
            if v is not None and not np.isnan(v):
                return abs(float(v))
    except Exception:
        pass
    return float("nan")


def _ml_price_for_primary(row: pd.Series, primary: str) -> Tuple[float, float]:
    primary_upper = (primary or "").upper()
    home_ml = safe_float(row.get("home_ml"))
    away_ml = safe_float(row.get("away_ml"))

    if "HOME ML" in primary_upper:
        return home_ml, away_ml
    if "AWAY ML" in primary_upper:
        return away_ml, home_ml
    return None, None


def _tier_rank(tier: str) -> int:
    t = (tier or "").upper().replace("MED VALUE", "MEDIUM VALUE")
    if "HIGH" in t:
        return 3
    if "MEDIUM" in t:
        return 2
    if "LOW" in t:
        return 1
    if "NO EDGE" in t:
        return 0
    return -1


def decide_bet_from_row(
    row: pd.Series,
    *,
    unit_dollars: float,
    settings: DecisionSettings = DecisionSettings(),
    require_pick: bool = True,
    require_value_tier: str = "HIGH VALUE",
    min_confidence: str = "MEDIUM",
    max_abs_moneyline: Optional[float] = None,
) -> DecisionOutcome:
    primary = str(row.get("primary_recommendation", ""))
    value_tier = str(row.get("value_tier", ""))
    conf = str(row.get("confidence", ""))

    flags: List[str] = []
    reason_parts: List[str] = []

    if require_pick and ("PICK" not in primary):
        return DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            "missing primary pick",
            "NO_PICK",
            "missing primary pick",
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    conf_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if conf_rank.get(conf, 0) < conf_rank.get(min_confidence, 1):
        return DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            f"PASS: confidence {conf} < {min_confidence}",
            "CONF_FILTER",
            f"confidence {conf} < {min_confidence}",
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    ml_price, opp_price = _ml_price_for_primary(row, primary)

    if "ML" in primary.upper() and ml_price is None:
        return DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            "missing moneyline odds",
            "MISSING_ODDS",
            "missing moneyline odds",
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
        )

    p_model, p_market = _probabilities_for_primary(row, primary)
    abs_edge = _abs_edge_from_row(row, primary, p_model, p_market)

    if abs_edge is None or np.isnan(abs_edge):
        return DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            "PASS: missing edge",
            "MISSING_EDGE",
            "missing edge",
            0.0,
            0.0,
            p_model if p_model is not None else float("nan"),
            p_market if p_market is not None else float("nan"),
            float("nan"),
        )

    if abs_edge < float(settings.min_play_edge_abs):
        return DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            f"PASS: abs_edge<{settings.min_play_edge_abs:.3f}",
            "EDGE_FILTER",
            f"edge<{settings.min_play_edge_abs:.3f}",
            0.0,
            0.0,
            p_model if p_model is not None else float("nan"),
            p_market if p_market is not None else float("nan"),
            abs_edge,
        )

    tier = value_tier_from_edge(abs_edge)
    raw_units = default_bet_units_from_tier(tier)
    primary_ev = safe_float(row.get("primary_ev"))
    ev_override = (
        primary_ev is not None
        and np.isfinite(primary_ev)
        and abs_edge is not None
        and np.isfinite(abs_edge)
        and float(primary_ev) >= float(settings.min_ev_override)
        and float(abs_edge) >= float(settings.min_ev_override_edge)
    )

    if require_value_tier:
        current_rank = _tier_rank(value_tier)
        required_rank = _tier_rank(require_value_tier)
        if current_rank < required_rank and not ev_override:
            return DecisionOutcome(
                "PASS",
                0.0,
                float(unit_dollars),
                0.0,
                f"value_tier {value_tier} below {require_value_tier}",
                "TIER_FILTER",
                f"PASS: value_tier {value_tier} below {require_value_tier}",
                0.0,
                0.0,
                p_model if p_model is not None else float("nan"),
                p_market if p_market is not None else float("nan"),
                abs_edge,
            )
    if ev_override:
        if "EV_OVERRIDE" not in flags:
            flags.append("EV_OVERRIDE")
        reason_parts.append(
            f"PLAY: EV override primary_ev={float(primary_ev):.3f} abs_edge={abs_edge:.3f}"
        )
        if raw_units <= 0:
            raw_units = default_bet_units_from_tier("LOW VALUE")

    # Base bet sizing (flat pct or Kelly for ML)
    bet_size = float(raw_units * float(unit_dollars))
    primary_upper = primary.upper()
    if settings.sizing_mode == "kelly" and "ML" in primary_upper:
        ml_price, _ = _ml_price_for_primary(row, primary)
        if ml_price is None or np.isnan(ml_price) or p_model is None or np.isnan(p_model):
            bet_size = 0.0
        else:
            bet_size = bet_size_kelly_ml(
                float(unit_dollars) / settings.flat_pct,
                float(p_model),
                float(ml_price),
                kelly_mult=settings.kelly_mult,
                max_pct=settings.kelly_max_pct,
            )
    elif settings.sizing_mode == "flat":
        bet_size = bet_size_flat(float(unit_dollars) / settings.flat_pct, settings.flat_pct)

    raw_units = bet_size / float(unit_dollars) if unit_dollars > 0 else 0.0
    final_units = min(raw_units, float(settings.max_units))

    if max_abs_moneyline is not None and ml_price is not None and not np.isnan(ml_price):
        if abs(float(ml_price)) > float(max_abs_moneyline):
            if "LONGSHOT_CAP" not in flags:
                flags.append("LONGSHOT_CAP")
            reason_parts.append(f"moneyline {ml_price} over max_abs {max_abs_moneyline}")
            if abs_edge < float(settings.min_sanity_edge_abs):
                return DecisionOutcome(
                    "PASS",
                    0.0,
                    float(unit_dollars),
                    0.0,
                    "longshot edge too small",
                    ",".join(flags),
                    "longshot edge too small",
                    raw_units,
                    0.0,
                    p_model if p_model is not None else float("nan"),
                    p_market if p_market is not None else float("nan"),
                    abs_edge,
                )
            final_units = min(final_units, float(settings.max_units_sanity))

    # Additional longshot sanity for extreme favorite or underdog
    if ml_price is not None and not np.isnan(ml_price):
        longshot = False
        if ml_price >= float(settings.longshot_cutoff):
            longshot = True
            reason_parts.append(f"underdog {ml_price}>=+{settings.longshot_cutoff:.0f}")
        if opp_price is not None and not np.isnan(opp_price) and opp_price <= float(settings.favorite_extreme_cutoff):
            longshot = True
            reason_parts.append(f"favorite opp {opp_price}<={settings.favorite_extreme_cutoff:.0f}")
        if longshot:
            if "LONGSHOT_CAP" not in flags:
                flags.append("LONGSHOT_CAP")
            if abs_edge < float(settings.min_sanity_edge_abs):
                return DecisionOutcome(
                    "PASS",
                    0.0,
                    float(unit_dollars),
                    0.0,
                    "longshot edge too small",
                    ",".join(flags),
                    "longshot edge too small",
                    raw_units,
                    0.0,
                    p_model if p_model is not None else float("nan"),
                    p_market if p_market is not None else float("nan"),
                    abs_edge,
                )
            final_units = min(final_units, float(settings.max_units_sanity))

    if p_model is not None and p_market is not None and not np.isnan(p_model) and not np.isnan(p_market):
        disagreement = abs(float(p_model) - float(p_market))
        if disagreement > float(settings.max_disagreement):
            flags.append("DISAGREE_CAP")
            reason_parts.append(
                f"|model-market|={disagreement:.3f}>{float(settings.max_disagreement):.3f}"
            )
            if abs_edge < float(settings.min_sanity_edge_abs):
                return DecisionOutcome(
                    "PASS",
                    0.0,
                    float(unit_dollars),
                    0.0,
                    "market disagreement edge too small",
                    ",".join(flags),
                    "market disagreement edge too small",
                    raw_units,
                    0.0,
                    p_model,
                    p_market,
                    abs_edge,
                )
            final_units = min(final_units, float(settings.max_units_sanity))

    if final_units > 0 and final_units < float(settings.min_play_units):
        flags.append("MIN_UNIT_FLOOR")
        final_units = float(settings.min_play_units)

    bet_size = float(final_units * float(unit_dollars))
    play_pass = "PLAY" if final_units > 0 else "PASS"
    decision_reason = ", ".join(reason_parts) if reason_parts else ""
    if flags and not decision_reason:
        decision_reason = ",".join(flags)

    return DecisionOutcome(
        play_pass,
        bet_size,
        float(unit_dollars),
        final_units,
        f"edge={abs_edge:.3f} tier={tier}",
        ",".join(flags),
        decision_reason,
        raw_units,
        final_units,
        p_model if p_model is not None else float("nan"),
        p_market if p_market is not None else float("nan"),
        abs_edge,
    )


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
        decide_bet_from_row(
            r,
            unit_dollars=unit_dollars,
            settings=DecisionSettings(min_play_edge_abs=min_play_edge_abs),
            require_pick=True,
            require_value_tier="",  # respect whatever tier the sheet already uses
            min_confidence="LOW",
        )
        for _, r in out.iterrows()
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
    out["decision_flags"] = [d.decision_flags for d in decisions]
    out["decision_reason"] = [d.decision_reason for d in decisions]
    out["raw_units"] = [d.raw_units for d in decisions]
    out["final_units"] = [d.final_units for d in decisions]
    out["p_model_used"] = [d.p_model_used for d in decisions]
    out["p_market_used"] = [d.p_market_used for d in decisions]
    out["abs_edge_used"] = [d.abs_edge_used for d in decisions]

    return out


def format_decision_trace(row: pd.Series, decision: DecisionOutcome) -> str:
    """Human-readable trace for debugging play/pass decisions."""

    def _fmt(x):
        return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"

    parts = [
        f"{row.get('home', '')} vs {row.get('away', '')}",
        f"primary={row.get('primary_recommendation', '')}",
        f"model_p={_fmt(decision.p_model_used)} market_p={_fmt(decision.p_market_used)}",
        f"abs_edge={_fmt(decision.abs_edge_used)}",
        f"raw_units={decision.raw_units:.2f} -> final_units={decision.final_units:.2f}",
        f"flags={decision.decision_flags or 'NONE'}",
        f"reason={decision.decision_reason or decision.reason}",
        f"play_pass={decision.play_pass}",
    ]
    return " | ".join(parts)
