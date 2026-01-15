# sports/common/bet_rules.py
from __future__ import annotations

from dataclasses import dataclass
import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.bankroll import bet_size_kelly_ml
from sports.common.bet_config import SportBetConfig, get_sport_bet_config
from sports.common.prob_calibration import calibrate_prob
from sports.common.util import safe_float

# -------------------------------------------------------------------
# Betting / sizing rules shared across sports
# -------------------------------------------------------------------

DEFAULT_UNIT_DOLLARS = 10.0

# Value tiers based on calibrated edge probability (model vs market)
TIER_HIGH = 0.06
TIER_MED = 0.03
TIER_LOW = 0.03

# Confidence tiers based on calibrated edge probability
CONF_HIGH = 0.05
CONF_MED = 0.05

# Legacy defaults for call sites that still pass them explicitly
MIN_EV_TO_PLAY = 0.015
MIN_PLAY_EDGE_ABS = MIN_EV_TO_PLAY
MIN_PRIMARY_EDGE_ABS = 0.03
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


def implied_prob_american(odds: float) -> float:
    try:
        odds = float(odds)
    except Exception:
        return float("nan")
    if np.isnan(odds) or odds == 0:
        return float("nan")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


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
    p_model_raw: float
    p_model_cal: float
    p_model_final: float
    p_market: float
    edge_prob_raw: float
    edge_prob_cal: float
    edge_prob_final: float


def confidence_tier_from_edge(edge_prob: float, min_edge: float) -> str:
    if edge_prob is None or (isinstance(edge_prob, float) and np.isnan(edge_prob)):
        return "UNKNOWN"
    edge_prob = float(edge_prob)
    if edge_prob > CONF_HIGH:
        return "HIGH"
    if edge_prob >= float(min_edge):
        return "MEDIUM"
    return "LOW"


def value_tier_from_edge(edge_prob: float, min_edge: float) -> str:
    if edge_prob is None or (isinstance(edge_prob, float) and np.isnan(edge_prob)):
        return "UNKNOWN"
    edge_prob = float(edge_prob)
    if edge_prob >= TIER_HIGH:
        return "HIGH VALUE"
    if edge_prob >= TIER_MED:
        return "MED VALUE"
    if edge_prob >= float(min_edge):
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

    tier = tier or value_tier_from_edge(abs_edge, min_edge)
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
    model_total = safe_float(row.get("model_total_final", row.get("model_total")))
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
    edge_prob_cal: float,
    injury_low: bool,
    config: SportBetConfig,
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
        if float(american_odds) >= 400:
            tier = "LOW"
            penalties.append("ML_CONFIDENCE_CAP>=400")
        elif float(american_odds) >= 250 and tier == "HIGH":
            tier = "MEDIUM"
            penalties.append("ML_CONFIDENCE_CAP>=250")
    if np.isfinite(edge_prob_cal) and abs(float(edge_prob_cal)) > float(config.disagree_cap_edge):
        tier = "LOW"
        penalties.append("DISAGREE_CAP")
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

        model_home = safe_float(row.get("model_home_prob_raw", row.get("model_home_prob")))
        model_prob = None if model_home is None or np.isnan(model_home) else float(model_home)
        if model_prob is not None and side == "AWAY":
            model_prob = 1.0 - model_prob

        market_prob = None
        if side == "HOME":
            market_home = safe_float(row.get("market_home_prob"))
            market_prob = None if market_home is None or np.isnan(market_home) else float(market_home)
        elif side == "AWAY":
            market_away = safe_float(row.get("market_away_prob"))
            if market_away is not None and not np.isnan(market_away):
                market_prob = float(market_away)
            else:
                market_home = safe_float(row.get("market_home_prob"))
                market_prob = None if market_home is None or np.isnan(market_home) else float(1.0 - market_home)
        if market_prob is None or np.isnan(market_prob):
            market_prob = implied_prob_american(ml_price)

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

        market_prob = None
        if side == "HOME":
            market_prob = safe_float(row.get("market_home_cover_prob", row.get("market_spread_prob")))
        elif side == "AWAY":
            market_away_cover = safe_float(row.get("market_away_cover_prob"))
            if market_away_cover is not None and not np.isnan(market_away_cover):
                market_prob = market_away_cover
            else:
                market_home_cover = safe_float(row.get("market_home_cover_prob", row.get("market_spread_prob")))
                if market_home_cover is not None and not np.isnan(market_home_cover):
                    market_prob = 1.0 - float(market_home_cover)
        if market_prob is None or np.isnan(market_prob):
            market_prob = implied_prob_american(spread_price)
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
        market_prob = None
        if side == "OVER":
            market_prob = safe_float(row.get("market_over_prob", row.get("market_total_prob")))
        elif side == "UNDER":
            market_under = safe_float(row.get("market_under_prob"))
            if market_under is not None and not np.isnan(market_under):
                market_prob = market_under
            else:
                market_over = safe_float(row.get("market_over_prob", row.get("market_total_prob")))
                if market_over is not None and not np.isnan(market_over):
                    market_prob = 1.0 - float(market_over)
        if market_prob is None or np.isnan(market_prob):
            market_prob = implied_prob_american(total_price)
        return model_prob, market_prob, total_price, None, None

    return None, None, None, None, "missing primary market"


def _anchor_prob(p_cal: float, p_market: float, config: SportBetConfig) -> float:
    if not np.isfinite(p_cal):
        return float("nan")
    if not np.isfinite(p_market):
        return float(p_cal)
    w = float(max(0.0, min(1.0, float(config.anchor_weight))))
    return float(w * p_market + (1.0 - w) * p_cal)


def _apply_underdog_cap(p_final: float, p_market: float, config: SportBetConfig) -> Tuple[float, bool]:
    if not np.isfinite(p_final) or not np.isfinite(p_market):
        return p_final, False
    cap = None
    if p_market < float(config.underdog_cap_low_prob):
        cap = p_market + float(config.underdog_cap_low_add)
    elif p_market < float(config.underdog_cap_med_prob):
        cap = p_market + float(config.underdog_cap_med_add)
    if cap is None:
        return p_final, False
    capped = min(p_final, cap)
    return float(capped), capped < p_final - 1e-9


def _ml_probabilities_for_side(
    model_home: float,
    market_home: float,
    side: str,
    *,
    sport: str,
    config: SportBetConfig,
) -> Tuple[float, float, float, float, List[str]]:
    flags: List[str] = []
    side = (side or "").upper()
    p_raw = float(model_home) if np.isfinite(model_home) else float("nan")
    p_market = float(market_home) if np.isfinite(market_home) else float("nan")

    if side == "AWAY":
        if np.isfinite(p_raw):
            p_raw = 1.0 - p_raw
        if np.isfinite(p_market):
            p_market = 1.0 - p_market

    p_cal = calibrate_prob(sport, p_raw, market_type="ML")
    if not np.isfinite(p_cal):
        p_cal = p_raw

    p_final = _anchor_prob(p_cal, p_market, config)
    p_final, capped = _apply_underdog_cap(p_final, p_market, config)
    if capped:
        flags.append("UNDERDOG_CAP")

    return p_raw, p_cal, p_final, p_market, flags


def ml_probabilities_for_row(row: pd.Series, sport: str = "nba") -> Dict[str, float]:
    model_home = safe_float(row.get("model_home_prob_raw", row.get("model_home_prob")))
    market_home = safe_float(row.get("market_home_prob"))
    config = get_sport_bet_config(sport)

    home_raw, home_cal, home_final, home_market, _ = _ml_probabilities_for_side(
        float(model_home) if model_home is not None else float("nan"),
        float(market_home) if market_home is not None else float("nan"),
        "HOME",
        sport=sport,
        config=config,
    )
    away_raw, away_cal, away_final, away_market, _ = _ml_probabilities_for_side(
        float(model_home) if model_home is not None else float("nan"),
        float(market_home) if market_home is not None else float("nan"),
        "AWAY",
        sport=sport,
        config=config,
    )

    return {
        "model_home_prob_raw": home_raw,
        "model_home_prob_cal": home_cal,
        "model_home_prob_final": home_final,
        "model_away_prob_raw": away_raw,
        "model_away_prob_cal": away_cal,
        "model_away_prob_final": away_final,
        "market_home_prob": home_market,
        "market_away_prob": away_market,
    }


def primary_metrics_for_row(
    row: pd.Series,
    *,
    sport: str = "nba",
    settings: DecisionSettings = DecisionSettings(),
) -> Tuple[
    str,
    str,
    float,
    float,
    float,
    float,
    float,
    float,
    str,
    str,
    str,
    Optional[float],
    Optional[str],
    str,
]:
    primary_market, primary_side = _primary_market_and_side(row)
    p_model_raw, p_market, primary_price, opp_price, data_reason = _primary_probabilities(
        row, primary_market, primary_side
    )

    config = get_sport_bet_config(sport)
    p_model_raw_val = (
        float(p_model_raw) if p_model_raw is not None and np.isfinite(p_model_raw) else float("nan")
    )
    p_market_val = float(p_market) if p_market is not None and np.isfinite(p_market) else float("nan")

    p_model_cal = calibrate_prob(sport, p_model_raw_val, market_type=primary_market)
    if not np.isfinite(p_model_cal):
        p_model_cal = p_model_raw_val

    edge_prob_raw = (
        float(p_model_raw_val) - float(p_market_val)
        if np.isfinite(p_model_raw_val) and np.isfinite(p_market_val)
        else float("nan")
    )
    edge_prob_cal = (
        float(p_model_cal) - float(p_market_val)
        if np.isfinite(p_model_cal) and np.isfinite(p_market_val)
        else float("nan")
    )

    p_model_final = p_model_cal
    edge_prob_final = edge_prob_cal
    flags: List[str] = []

    if primary_market == "ML":
        p_model_final = _anchor_prob(p_model_cal, p_market_val, config)
        p_model_final, capped = _apply_underdog_cap(p_model_final, p_market_val, config)
        if capped:
            flags.append("UNDERDOG_CAP")
        edge_prob_final = (
            float(p_model_final) - float(p_market_val)
            if np.isfinite(p_model_final) and np.isfinite(p_market_val)
            else float("nan")
        )

    base_conf = confidence_tier_from_edge(edge_prob_final, config.min_edge_cal)
    injury_low = _injury_confidence_low(row)
    conf, penalties = _apply_confidence_penalties(
        base_conf,
        primary_market=primary_market,
        american_odds=primary_price,
        edge_prob_cal=edge_prob_cal,
        injury_low=injury_low,
        config=config,
    )
    conf_reason = ", ".join(penalties) if penalties else ""
    value_tier = value_tier_from_edge(edge_prob_final, config.min_edge_cal)

    return (
        primary_market,
        primary_side,
        p_model_raw_val,
        p_model_cal,
        p_model_final,
        p_market_val,
        edge_prob_raw,
        edge_prob_cal,
        edge_prob_final,
        conf,
        conf_reason,
        value_tier,
        primary_price,
        data_reason,
        ",".join(flags),
    )


def decide_bet_from_row(
    row: pd.Series,
    *,
    unit_dollars: float,
    sport: str = "nba",
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
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    (
        primary_market,
        primary_side,
        p_model_raw,
        p_model_cal,
        p_model_final,
        p_market,
        edge_prob_raw,
        edge_prob_cal,
        edge_prob_final,
        _conf,
        _conf_reason,
        value_tier,
        primary_price,
        data_reason,
        metric_flags,
    ) = primary_metrics_for_row(row, sport=sport, settings=settings)

    config = get_sport_bet_config(sport)
    min_edge_cal = float(config.min_edge_cal)
    if str(sport).lower() == "nhl":
        goalie_status = str(row.get("goalie_status", "UNKNOWN")).upper().strip()
        if goalie_status == "UNKNOWN":
            min_edge_cal = max(min_edge_cal, float(os.getenv("NHL_MIN_EDGE_CAL_UNKNOWN", "0.05")))

    if data_reason or not np.isfinite(p_model_cal) or not np.isfinite(p_market):
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
            p_model_raw,
            p_model_cal,
            p_model_final,
            p_market,
            edge_prob_raw,
            edge_prob_cal,
            edge_prob_final,
        )

    if edge_prob_final < float(min_edge_cal):
        return DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            f"PASS: edge_final<{min_edge_cal:.3f}",
            "LOW_EDGE_PASS",
            f"edge_cal<{min_edge_cal:.3f}",
            0.0,
            0.0,
            p_model_raw,
            p_model_cal,
            p_model_final,
            p_market,
            edge_prob_raw,
            edge_prob_cal,
            edge_prob_final,
        )

    raw_units = default_bet_units_from_tier(value_tier)
    bet_size = float(raw_units * float(unit_dollars))

    if settings.sizing_mode == "kelly" and primary_market == "ML":
        if primary_price is None or not np.isfinite(primary_price) or not np.isfinite(p_model_cal):
            bet_size = float(raw_units * float(unit_dollars))
        else:
            bet_size = bet_size_kelly_ml(
                float(unit_dollars) / settings.flat_pct,
                float(p_model_final),
                float(primary_price),
                kelly_mult=settings.kelly_mult,
                max_pct=settings.kelly_max_pct,
            )

    raw_units = bet_size / float(unit_dollars) if unit_dollars > 0 else 0.0
    final_units = min(raw_units, float(config.max_units))

    if primary_market == "ML" and primary_price is not None and np.isfinite(primary_price):
        if float(primary_price) >= float(config.longshot_odds):
            flags.append("LONGSHOT_CAP")
            final_units = min(final_units, float(config.longshot_cap_units))
            reason_parts.append(f"longshot {primary_price}>=+{config.longshot_odds:.0f}")

    if metric_flags:
        for flag in metric_flags.split(","):
            if flag:
                flags.append(flag)

    disagreement = abs(float(p_model_cal - p_market)) if np.isfinite(p_model_cal) and np.isfinite(p_market) else 0.0
    if disagreement > float(config.disagree_pass_edge):
        if edge_prob_final < float(config.disagree_pass_min_edge) or raw_units > float(config.disagree_pass_max_units):
            flags.append("DISAGREE_PASS")
            return DecisionOutcome(
                "PASS",
                0.0,
                float(unit_dollars),
                0.0,
                "DISAGREE_PASS",
                ",".join(flags),
                "DISAGREE_PASS",
                raw_units,
                0.0,
                p_model_raw,
                p_model_cal,
                p_model_final,
                p_market,
                edge_prob_raw,
                edge_prob_cal,
                edge_prob_final,
            )

    if disagreement > float(config.disagree_cap_edge):
        flags.append("DISAGREE_CAP")
        reason_parts.append(f"|model-market|={disagreement:.3f}>{float(config.disagree_cap_edge):.3f}")
        final_units = min(final_units, float(config.disagree_cap_units))

    if final_units > 0 and final_units < 0.25:
        if not {"LONGSHOT_CAP", "DISAGREE_CAP"}.intersection(flags):
            flags.append("MIN_UNIT_FLOOR")
            final_units = 0.25

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
        f"edge_final={edge_prob_final:.3f} tier={value_tier}",
        ",".join(flags),
        decision_reason,
        raw_units,
        final_units,
        p_model_raw,
        p_model_cal,
        p_model_final,
        p_market,
        edge_prob_raw,
        edge_prob_cal,
        edge_prob_final,
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
    sport: str = "nba",
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
    metrics = [primary_metrics_for_row(r, sport=sport) for _, r in out.iterrows()]
    out["primary_market"] = [m[0] for m in metrics]
    out["primary_side"] = [m[1] for m in metrics]
    out["p_model_raw"] = [m[2] for m in metrics]
    out["p_model_cal"] = [m[3] for m in metrics]
    out["p_model_final"] = [m[4] for m in metrics]
    out["p_market"] = [m[5] for m in metrics]
    out["p_model_used"] = out["p_model_final"]
    out["p_market_used"] = out["p_market"]
    out["edge_prob_raw"] = [m[6] for m in metrics]
    out["edge_prob_cal"] = [m[7] for m in metrics]
    out["edge_prob_final"] = [m[8] for m in metrics]
    out["abs_edge_prob"] = out["edge_prob_final"].abs()
    out["confidence"] = [m[9] for m in metrics]
    out["confidence_reason"] = [m[10] for m in metrics]
    out["value_tier"] = [m[11] for m in metrics]
    out["primary_price"] = [m[12] for m in metrics]

    decisions = [
        decide_bet_from_row(
            r,
            unit_dollars=unit_dollars,
            sport=sport,
            settings=DecisionSettings(),
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
    merged_flags: list[str] = []
    for i, d in enumerate(decisions):
        flags = [f for f in str(d.decision_flags or "").split(",") if f]
        total_flags = str(out.loc[i, "total_decision_flags"] or "").strip()
        total_flag_list = [f for f in total_flags.split(",") if f]

        apply_total_flags = False
        if str(out.loc[i, "primary_market"]).upper() == "TOTAL":
            apply_total_flags = True
        elif str(out.loc[i, "total_recommendation"]).startswith("Model PICK TOTAL") and d.play_pass == "PLAY":
            apply_total_flags = True

        if apply_total_flags:
            for flag in total_flag_list:
                if flag not in flags:
                    flags.append(flag)

        merged_flags.append(",".join(flags))

    out["decision_flags"] = merged_flags
    out["decision_reason"] = [d.decision_reason for d in decisions]
    out["raw_units"] = [d.raw_units for d in decisions]
    out["final_units"] = [d.final_units for d in decisions]
    out["edge_prob_raw"] = [d.edge_prob_raw for d in decisions]
    out["edge_prob_cal"] = [d.edge_prob_cal for d in decisions]
    out["edge_prob_final"] = [d.edge_prob_final for d in decisions]
    out["stake_dollars"] = out["units"] * out["unit_dollars"]

    return out


def format_decision_trace(row: pd.Series, decision: DecisionOutcome) -> str:
    """Human-readable trace for debugging play/pass decisions."""

    def _fmt(x):
        return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"

    parts = [
        f"{row.get('home', '')} vs {row.get('away', '')}",
        f"primary={row.get('primary_recommendation', '')}",
        f"model_raw={_fmt(decision.p_model_raw)} model_cal={_fmt(decision.p_model_cal)}",
        f"model_final={_fmt(decision.p_model_final)} market_p={_fmt(decision.p_market)} "
        f"edge_final={_fmt(decision.edge_prob_final)}",
        f"raw_units={decision.raw_units:.2f} -> final_units={decision.final_units:.2f}",
        f"flags={decision.decision_flags or 'NONE'}",
        f"reason={decision.decision_reason or decision.reason}",
        f"play_pass={decision.play_pass}",
    ]
    return " | ".join(parts)
