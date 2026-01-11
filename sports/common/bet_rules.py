# sports/common/bet_rules.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.bankroll import bet_size_kelly_ml
from sports.common.util import safe_float

# -------------------------------------------------------------------
# Betting / sizing rules shared across sports
# -------------------------------------------------------------------

DEFAULT_UNIT_DOLLARS = 10.0

# Value tiers based on absolute edge probability (prob edge vs market)
TIER_HIGH = 0.06
TIER_MED = 0.03
TIER_LOW = 0.015

# Confidence tiers based on absolute edge probability
CONF_HIGH = 0.05
CONF_MED = 0.02

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
    longshot_extreme_cutoff: float = 700.0
    favorite_extreme_cutoff: float = -800.0
    longshot_extreme_min_edge: float = 0.08
    longshot_max_units: float = 0.25
    longshot_extreme_units: float = 0.10
    max_disagreement: float = 0.20
    disagreement_max_units: float = 0.25
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


def confidence_tier_from_edge(abs_edge: float) -> str:
    if abs_edge is None or (isinstance(abs_edge, float) and np.isnan(abs_edge)):
        return "UNKNOWN"
    abs_edge = float(abs_edge)
    if abs_edge > CONF_HIGH:
        return "HIGH"
    if abs_edge >= CONF_MED:
        return "MEDIUM"
    return "LOW"


def value_tier_from_edge(abs_edge: float) -> str:
    if abs_edge is None or (isinstance(abs_edge, float) and np.isnan(abs_edge)):
        return "UNKNOWN"
    abs_edge = float(abs_edge)
    if abs_edge >= TIER_HIGH:
        return "HIGH VALUE"
    if abs_edge >= TIER_MED:
        return "MED VALUE"
    if abs_edge >= TIER_LOW:
        return "LOW VALUE"
    return "NO BET"


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


def _norm_cdf(x: float) -> float:
    try:
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
    except Exception:
        return float("nan")


def _primary_market_and_side(row: pd.Series) -> Tuple[str, str]:
    primary_market = str(row.get("primary_market", "")).upper().strip()
    primary_side = str(row.get("primary_side", "")).upper().strip()
    primary = str(row.get("primary_recommendation", "")).upper()

    if not primary_market or primary_market == "NONE":
        if "TOTAL" in primary:
            primary_market = "TOTAL"
        elif "ATS" in primary or "SPREAD" in primary:
            primary_market = "ATS"
        elif "ML" in primary:
            primary_market = "ML"

    if not primary_side:
        if "HOME" in primary:
            primary_side = "HOME"
        elif "AWAY" in primary:
            primary_side = "AWAY"
        elif "OVER" in primary:
            primary_side = "OVER"
        elif "UNDER" in primary:
            primary_side = "UNDER"

    return primary_market, primary_side


def _ml_price_for_primary(row: pd.Series, primary_side: str) -> Tuple[float, float]:
    home_ml = safe_float(row.get("home_ml"))
    away_ml = safe_float(row.get("away_ml"))
    side = (primary_side or "").upper()

    if side == "HOME":
        return home_ml, away_ml
    if side == "AWAY":
        return away_ml, home_ml
    return None, None


def _model_prob_for_total(row: pd.Series, primary_side: str) -> Optional[float]:
    side = (primary_side or "").upper()
    prob = safe_float(row.get("total_pick_prob"))
    if prob is not None and np.isfinite(prob):
        return float(prob) if side == "OVER" else 1.0 - float(prob)

    total_sd = safe_float(row.get("total_sd"))
    model_total = safe_float(row.get("model_total"))
    total_points = safe_float(row.get("total_points"))
    if total_sd is None or model_total is None or total_points is None:
        return None
    if not np.isfinite(total_sd) or not np.isfinite(model_total) or not np.isfinite(total_points):
        return None
    if float(total_sd) <= 1e-6:
        return None

    z = (float(total_points) - float(model_total)) / float(total_sd)
    p_over = 1.0 - _norm_cdf(z)
    if not np.isfinite(p_over):
        return None
    return float(p_over) if side == "OVER" else 1.0 - float(p_over)


def _injury_confidence_low(row: pd.Series) -> bool:
    for col in ("injury_confidence", "injury_data_confidence", "injury_conf"):
        val = str(row.get(col, "")).strip().lower()
        if val in {"low", "none", "unknown", "0"}:
            return True
    source = str(row.get("injury_source", "")).strip().lower()
    if source in {"none", "unknown"}:
        return True
    return False


def _apply_confidence_penalties(
    base_tier: str,
    *,
    primary_market: str,
    american_odds: Optional[float],
    abs_edge_prob: float,
    injury_low: bool,
) -> Tuple[str, List[str]]:
    penalties: List[str] = []
    tier_order = ["LOW", "MEDIUM", "HIGH"]
    tier = base_tier if base_tier in tier_order else "UNKNOWN"
    if tier == "UNKNOWN":
        return tier, penalties

    def _downgrade(t: str) -> str:
        idx = max(0, tier_order.index(t) - 1)
        return tier_order[idx]

    if primary_market == "ML" and american_odds is not None and np.isfinite(american_odds):
        if float(american_odds) >= 500:
            tier = _downgrade(tier)
            penalties.append("LONGSHOT_ODDS>=+500")
    if np.isfinite(abs_edge_prob) and float(abs_edge_prob) > 0.20:
        tier = _downgrade(tier)
        penalties.append("DISAGREE_PROB>0.20")
    if injury_low:
        tier = _downgrade(tier)
        penalties.append("INJURY_CONF_LOW")

    return tier, penalties


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


def _primary_probabilities(
    row: pd.Series, primary_market: str, primary_side: str
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str]]:
    market = (primary_market or "").upper()
    side = (primary_side or "").upper()

    if market == "ML":
        ml_price, opp_price = _ml_price_for_primary(row, side)
        if ml_price is None or not np.isfinite(ml_price):
            return None, None, ml_price, opp_price, "missing moneyline odds"

        model_home = safe_float(row.get("model_home_prob"))
        model_prob = None if model_home is None or np.isnan(model_home) else float(model_home)
        if model_prob is not None and side == "AWAY":
            model_prob = 1.0 - model_prob

        market_home = safe_float(row.get("market_home_prob"))
        market_prob = None if market_home is None or np.isnan(market_home) else float(market_home)
        if market_prob is not None and side == "AWAY":
            market_prob = 1.0 - market_prob
        if market_prob is None or np.isnan(market_prob):
            market_prob = breakeven_prob_from_american(ml_price)

        return model_prob, market_prob, ml_price, opp_price, None

    if market == "ATS":
        spread_line = safe_float(row.get("home_spread"))
        spread_price = safe_float(row.get("spread_price"))
        if (
            spread_line is None
            or spread_price is None
            or not np.isfinite(spread_line)
            or not np.isfinite(spread_price)
        ):
            return None, None, None, None, "missing spread line or price"

        model_home_cover = safe_float(row.get("ats_home_cover_prob", row.get("p_home_cover")))
        model_prob = None if model_home_cover is None or np.isnan(model_home_cover) else float(model_home_cover)
        if model_prob is not None and side == "AWAY":
            model_prob = 1.0 - model_prob

        market_prob = breakeven_prob_from_american(spread_price)
        return model_prob, market_prob, spread_price, None, None

    if market == "TOTAL":
        total_line = safe_float(row.get("total_points"))
        total_price = safe_float(
            row.get("total_over_price") if side == "OVER" else row.get("total_under_price")
        )
        if (
            total_line is None
            or total_price is None
            or not np.isfinite(total_line)
            or not np.isfinite(total_price)
        ):
            return None, None, None, None, "missing total line or price"

        model_prob = _model_prob_for_total(row, side)
        market_prob = breakeven_prob_from_american(total_price)
        return model_prob, market_prob, total_price, None, None

    return None, None, None, None, "missing primary market"


def primary_metrics_for_row(
    row: pd.Series,
) -> Tuple[str, str, float, float, float, str, str, str, Optional[float], Optional[str]]:
    primary_market, primary_side = _primary_market_and_side(row)
    p_model, p_market, primary_price, opp_price, data_reason = _primary_probabilities(
        row, primary_market, primary_side
    )

    p_model_used = float(p_model) if p_model is not None and np.isfinite(p_model) else float("nan")
    p_market_used = float(p_market) if p_market is not None and np.isfinite(p_market) else float("nan")
    abs_edge_prob = (
        abs(float(p_model_used) - float(p_market_used))
        if np.isfinite(p_model_used) and np.isfinite(p_market_used)
        else float("nan")
    )

    base_conf = confidence_tier_from_edge(abs_edge_prob)
    injury_low = _injury_confidence_low(row)
    conf, penalties = _apply_confidence_penalties(
        base_conf,
        primary_market=primary_market,
        american_odds=primary_price,
        abs_edge_prob=abs_edge_prob,
        injury_low=injury_low,
    )
    conf_reason = ", ".join(penalties) if penalties else ""
    value_tier = value_tier_from_edge(abs_edge_prob)

    return (
        primary_market,
        primary_side,
        p_model_used,
        p_market_used,
        abs_edge_prob,
        conf,
        conf_reason,
        value_tier,
        primary_price,
        data_reason,
    )


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
    flags: List[str] = []
    reason_parts: List[str] = []

    if require_pick and ("PICK" not in primary.upper()):
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

    (
        primary_market,
        primary_side,
        p_model_used,
        p_market_used,
        abs_edge_prob,
        _conf,
        _conf_reason,
        value_tier,
        primary_price,
        data_reason,
    ) = primary_metrics_for_row(row)

    if data_reason or not np.isfinite(p_model_used) or not np.isfinite(p_market_used):
        flags.append("MISSING_DATA_PASS")
        return DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            data_reason or "missing probabilities",
            ",".join(flags),
            data_reason or "missing probabilities",
            0.0,
            0.0,
            p_model_used,
            p_market_used,
            abs_edge_prob,
        )

    if abs_edge_prob < float(settings.min_play_edge_abs):
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
            p_model_used,
            p_market_used,
            abs_edge_prob,
        )

    raw_units = default_bet_units_from_tier(value_tier)
    bet_size = float(raw_units * float(unit_dollars))

    if settings.sizing_mode == "kelly" and primary_market == "ML":
        if primary_price is None or not np.isfinite(primary_price) or not np.isfinite(p_model_used):
            bet_size = float(raw_units * float(unit_dollars))
        else:
            bet_size = bet_size_kelly_ml(
                float(unit_dollars) / settings.flat_pct,
                float(p_model_used),
                float(primary_price),
                kelly_mult=settings.kelly_mult,
                max_pct=settings.kelly_max_pct,
            )

    raw_units = bet_size / float(unit_dollars) if unit_dollars > 0 else 0.0
    final_units = min(raw_units, float(settings.max_units))

    if primary_market == "ML" and primary_price is not None and np.isfinite(primary_price):
        if float(primary_price) >= float(settings.longshot_extreme_cutoff):
            if abs_edge_prob < float(settings.longshot_extreme_min_edge):
                flags.append("LONGSHOT_CAP")
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
                    p_model_used,
                    p_market_used,
                    abs_edge_prob,
                )
            flags.append("LONGSHOT_CAP")
            final_units = min(final_units, float(settings.longshot_extreme_units))
            reason_parts.append(f"longshot {primary_price}>=+{settings.longshot_extreme_cutoff:.0f}")
        elif float(primary_price) >= float(settings.longshot_cutoff):
            flags.append("LONGSHOT_CAP")
            final_units = min(final_units, float(settings.longshot_max_units))
            reason_parts.append(f"longshot {primary_price}>=+{settings.longshot_cutoff:.0f}")

    disagreement = abs(float(p_model_used) - float(p_market_used))
    if disagreement > float(settings.max_disagreement):
        flags.append("DISAGREE_CAP")
        reason_parts.append(
            f"|model-market|={disagreement:.3f}>{float(settings.max_disagreement):.3f}"
        )
        final_units = min(final_units, float(settings.disagreement_max_units))

    if final_units > 0 and final_units < float(settings.min_play_units):
        if not {"LONGSHOT_CAP", "DISAGREE_CAP"}.intersection(flags):
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
        f"edge={abs_edge_prob:.3f} tier={value_tier}",
        ",".join(flags),
        decision_reason,
        raw_units,
        final_units,
        p_model_used,
        p_market_used,
        abs_edge_prob,
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

    # --- 2) Compute primary probabilities/tiers/confidence ---
    metrics = [primary_metrics_for_row(r) for _, r in out.iterrows()]
    out["primary_market"] = [m[0] for m in metrics]
    out["primary_side"] = [m[1] for m in metrics]
    out["p_model_used"] = [m[2] for m in metrics]
    out["p_market_used"] = [m[3] for m in metrics]
    out["abs_edge_prob"] = [m[4] for m in metrics]
    out["confidence"] = [m[5] for m in metrics]
    out["confidence_reason"] = [m[6] for m in metrics]
    out["value_tier"] = [m[7] for m in metrics]

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

    out["play_pass"] = [d.play_pass for d in decisions]
    out["bet_size"] = [d.bet_size for d in decisions]
    out["unit_dollars"] = [d.unit_dollars for d in decisions]
    out["units"] = [d.units for d in decisions]
    out["why_bet"] = [d.reason for d in decisions]
    out["decision_flags"] = [d.decision_flags for d in decisions]
    out["decision_reason"] = [d.decision_reason for d in decisions]
    out["raw_units"] = [d.raw_units for d in decisions]
    out["final_units"] = [d.final_units for d in decisions]
    out["abs_edge_used"] = [d.abs_edge_used for d in decisions]
    out["abs_edge_prob"] = out["abs_edge_used"]
    out["stake_dollars"] = out["units"] * out["unit_dollars"]

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
