# sports/common/bet_rules.py
from __future__ import annotations

from dataclasses import dataclass, replace
import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.bankroll import bet_size_kelly_ml
from sports.common.bet_config import SportBetConfig, get_sport_bet_config
from sports.common.prob_calibration import calibrate_prob, load_calibrator, odds_bucket_from_price, MIN_SAMPLES
from sports.common.prob_uncertainty import load_uncertainty
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
MIN_PLAY_EDGE_ABS_NHL = 0.015
MIN_PRIMARY_EDGE_ABS_NHL = 0.025
MIN_SANITY_EDGE_ABS = 0.03
MIN_EV_OVERRIDE = 0.02
MIN_EV_OVERRIDE_EDGE = 0.02

EDGE_SHIFT_DOWNGRADE = float(os.getenv("EDGE_SHIFT_DOWNGRADE", "0.04"))
EXTREME_ML_POS_ODDS = float(os.getenv("EXTREME_ML_POS_ODDS", "350"))
EXTREME_ML_NEG_ODDS = float(os.getenv("EXTREME_ML_NEG_ODDS", "-450"))
EXTREME_EDGE_REQUIREMENT = float(os.getenv("EXTREME_ML_EDGE_REQUIREMENT", "0.12"))
ATS_UNCALIBRATED_EDGE_ADD = float(os.getenv("ATS_UNCALIBRATED_EDGE_ADD", "0.02"))
ATS_UNCALIBRATED_MAX_UNITS = float(os.getenv("ATS_UNCALIBRATED_MAX_UNITS", "0.5"))
ATS_UNCALIBRATED_EDGE_OVERRIDE = float(os.getenv("ATS_UNCALIBRATED_EDGE_OVERRIDE", "0.08"))
NBA_ATS_SHRINK = float(os.getenv("NBA_ATS_SHRINK", "1.25"))
NBA_ML_MIN_EDGE_OVERRIDE = os.getenv("NBA_ML_MIN_EDGE_OVERRIDE")


def _edge_thresholds_for_sport(sport: str) -> Tuple[float, float]:
    config = get_sport_bet_config(sport)
    base = float(config.min_edge_cal)
    dynamic = _dynamic_min_edge(sport, base, config)
    return float(dynamic), float(dynamic)


def _to_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def normalize_decision_flags(flags: object) -> str:
    if flags is None:
        return ""
    if isinstance(flags, str):
        parts = flags.split(",")
    elif isinstance(flags, (list, tuple, set)):
        parts = []
        for item in flags:
            if item is None:
                continue
            if isinstance(item, str):
                parts.extend(item.split(","))
            else:
                parts.append(str(item))
    else:
        parts = [str(flags)]

    cleaned: List[str] = []
    seen = set()
    for part in parts:
        item = str(part).strip()
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    return ",".join(cleaned)


_UNCERTAINTY_CACHE: Dict[str, Dict[str, float]] = {}


def _uncertainty_cache_key(sport: str, market: Optional[str]) -> str:
    sport_key = str(sport).lower()
    if market:
        return f"{sport_key}:{str(market).upper()}"
    return sport_key


def _load_uncertainty_data(sport: str, market: Optional[str] = None) -> Dict[str, float]:
    cache_key = _uncertainty_cache_key(sport, market)
    if cache_key in _UNCERTAINTY_CACHE:
        return _UNCERTAINTY_CACHE[cache_key]
    data = load_uncertainty(str(sport).lower(), market=market) or {}
    _UNCERTAINTY_CACHE[cache_key] = data
    return data


def _uncertainty_value(sport: str, market: Optional[str] = None) -> float:
    data = _load_uncertainty_data(sport, market)
    for key in ("uncertainty", "error_std", "brier"):
        val = safe_float(data.get(key))
        if val is not None and np.isfinite(val):
            if key == "brier":
                return float(math.sqrt(float(val)))
            return float(val)
    return 0.0


_UNCERTAINTY_LOGGED: Dict[str, bool] = {}
_NHL_SAMPLE_CACHE: Optional[int] = None
_NHL_SAMPLE_CACHE_CHECKED = False


def _uncertainty_for_threshold(
    sport: str,
    config: SportBetConfig,
    *,
    market: Optional[str] = None,
    row: Optional[pd.Series] = None,
) -> float:
    effective_uncertainty, _sample_size, _raw_uncertainty, _quality, _sample_factor, _quality_factor = (
        _effective_uncertainty(sport, config, use_floor=True, row=row, market=market)
    )
    if not np.isfinite(effective_uncertainty) or float(effective_uncertainty) <= 0.0:
        return float(config.uncertainty_floor)
    return float(effective_uncertainty)


def _log_uncertainty_once(
    sport: str,
    uncertainty: float,
    base_min_edge: float,
    config: SportBetConfig,
    *,
    effective_uncertainty: Optional[float] = None,
    sample_size: Optional[int] = None,
    quality: Optional[float] = None,
    sample_factor: Optional[float] = None,
    quality_factor: Optional[float] = None,
) -> None:
    key = str(sport).lower()
    if _UNCERTAINTY_LOGGED.get(key):
        return
    _UNCERTAINTY_LOGGED[key] = True
    extra = ""
    if effective_uncertainty is not None and np.isfinite(effective_uncertainty):
        extra = f" effective={effective_uncertainty:.3f}"
    if sample_size is not None:
        extra = f"{extra} n={sample_size}"
    if quality is not None and np.isfinite(quality):
        extra = f"{extra} quality={quality:.2f}"
    if sample_factor is not None and np.isfinite(sample_factor):
        extra = f"{extra} sample_f={sample_factor:.2f}"
    if quality_factor is not None and np.isfinite(quality_factor):
        extra = f"{extra} qual_f={quality_factor:.2f}"
    print(
        f"[uncertainty] sport={key} value={uncertainty:.3f} "
        f"edge_mult={config.uncertainty_edge_mult:.2f} base_min_edge={base_min_edge:.3f}{extra}"
    )


def _uncertainty_samples_from_data(data: Optional[Dict[str, float]]) -> Optional[int]:
    if not data:
        return None
    for key in ("n_samples", "sample_size", "samples", "n"):
        val = safe_float(data.get(key))
        if val is not None and np.isfinite(val) and val > 0:
            return int(val)
    return None


def _feature_quality(
    row: Optional[pd.Series],
    *,
    sport: str,
    market: Optional[str],
) -> float:
    if row is None:
        return 1.0

    quality = 1.0
    sport_key = str(sport).lower()
    market_key = str(market or "").upper()

    if sport_key == "nhl":
        goalie_confirmed = _nhl_goalie_confirmed(row)
        if not goalie_confirmed:
            quality *= 0.65

    injury_conf = str(row.get("injury_confidence", "") or "").lower()
    if injury_conf == "low":
        quality *= 0.7
    elif injury_conf == "medium":
        quality *= 0.85

    if sport_key in {"nba", "nfl"}:
        if str(row.get("injury_report_status", "")).lower() in {"unknown", "missing"}:
            quality *= 0.85

    if market_key == "ATS":
        spread_val = safe_float(row.get("home_spread"))
        if spread_val is None or not np.isfinite(spread_val):
            quality *= 0.75
        spread_price = safe_float(row.get("spread_price"))
        if spread_price is None or not np.isfinite(spread_price):
            quality *= 0.85
    if market_key == "TOTAL":
        total_val = safe_float(row.get("total_points"))
        if total_val is None or not np.isfinite(total_val):
            quality *= 0.75
        over_price = safe_float(row.get("total_over_price"))
        under_price = safe_float(row.get("total_under_price"))
        if not (
            (over_price is not None and np.isfinite(over_price))
            or (under_price is not None and np.isfinite(under_price))
        ):
            quality *= 0.85
    if market_key == "ML":
        home_ml = safe_float(row.get("home_ml"))
        away_ml = safe_float(row.get("away_ml"))
        if not (
            (home_ml is not None and np.isfinite(home_ml))
            or (away_ml is not None and np.isfinite(away_ml))
        ):
            quality *= 0.85

    return float(max(0.2, min(1.0, quality)))


def _effective_uncertainty(
    sport: str,
    config: SportBetConfig,
    *,
    use_floor: bool,
    row: Optional[pd.Series] = None,
    market: Optional[str] = None,
) -> Tuple[float, Optional[int], float, float, float, float]:
    raw_uncertainty = _uncertainty_value(sport, market)
    if (not np.isfinite(raw_uncertainty) or raw_uncertainty <= 0.0) and use_floor:
        raw_uncertainty = float(config.uncertainty_floor)

    data = _load_uncertainty_data(sport, market)
    sample_size = _uncertainty_samples_from_data(data)
    sample_size_for_scaling = sample_size if sample_size is not None else float(config.uncertainty_sample_ref)
    sample_factor = 1.0
    if sample_size_for_scaling > 0 and float(config.uncertainty_sample_ref) > 0:
        ratio = float(config.uncertainty_sample_ref) / float(sample_size_for_scaling)
        sample_factor = float(pow(ratio, float(config.uncertainty_sample_exp)))
        sample_factor = float(min(1.5, max(0.6, sample_factor)))

    quality = _feature_quality(row, sport=sport, market=market)
    quality_factor = 1.0 + (1.0 - float(quality)) * float(config.uncertainty_quality_mult)
    effective_uncertainty = float(raw_uncertainty) * float(sample_factor) * float(quality_factor)
    if use_floor and effective_uncertainty < float(config.uncertainty_floor):
        effective_uncertainty = float(config.uncertainty_floor)
    return (
        float(effective_uncertainty),
        sample_size,
        float(raw_uncertainty),
        float(quality),
        float(sample_factor),
        float(quality_factor),
    )


def _read_nhl_samples_from_bet_log() -> Optional[int]:
    for path in ("results/tracking/bet_log.csv", "results/bet_log.csv"):
        if not os.path.exists(path):
            continue
        try:
            bet_log = pd.read_csv(path)
        except Exception:
            continue
        if bet_log is None or bet_log.empty:
            continue
        if "sport" not in bet_log.columns or "market" not in bet_log.columns:
            continue
        sports = bet_log["sport"].astype(str).str.lower()
        markets = bet_log["market"].astype(str).str.upper()
        mask = (sports == "nhl") & (markets == "ML")
        return int(mask.sum())
    return None


def _nhl_sample_size() -> Optional[int]:
    global _NHL_SAMPLE_CACHE
    global _NHL_SAMPLE_CACHE_CHECKED
    if _NHL_SAMPLE_CACHE_CHECKED:
        return _NHL_SAMPLE_CACHE
    _NHL_SAMPLE_CACHE_CHECKED = True
    data = _load_uncertainty_data("nhl") or {}
    sample_size = _uncertainty_samples_from_data(data)
    if sample_size is None:
        sample_size = _read_nhl_samples_from_bet_log()
    _NHL_SAMPLE_CACHE = sample_size
    return sample_size


def _nhl_uncertainty_context(
    config: SportBetConfig,
    *,
    use_floor: bool,
    row: Optional[pd.Series] = None,
    market: Optional[str] = None,
) -> Tuple[float, int, float]:
    effective_uncertainty, sample_size, raw_uncertainty, _quality, _sample_factor, _quality_factor = (
        _effective_uncertainty("nhl", config, use_floor=use_floor, row=row, market=market)
    )
    sample_size_for_scaling = sample_size if sample_size is not None else int(config.uncertainty_sample_ref)
    return float(effective_uncertainty), int(sample_size_for_scaling), float(raw_uncertainty)


def _dynamic_min_edge(
    sport: str,
    base_min_edge: float,
    config: SportBetConfig,
    *,
    row: Optional[pd.Series] = None,
    market: Optional[str] = None,
) -> float:
    if str(sport).lower() == "nhl":
        cap_env = os.getenv("NHL_DYNAMIC_MIN_EDGE_CAP")
        if cap_env is not None and str(cap_env).strip() != "":
            try:
                cap = float(cap_env)
            except Exception:
                cap = float(base_min_edge) + 0.025
        else:
            cap = float(base_min_edge) + 0.025
        min_edge = float(base_min_edge)
        if cap > 0:
            min_edge = min(float(min_edge), cap)
        return float(max(0.0, float(min_edge)))
    cap_add = float(config.uncertainty_edge_cap_add)
    min_edge = float(base_min_edge)
    if cap_add > 0:
        min_edge = min(float(min_edge), float(base_min_edge) + float(cap_add))
    return float(max(0.0, float(min_edge)))


@dataclass(frozen=True)
class NhlAnchorContext:
    anchor_weight: float
    effective_uncertainty: float
    low_unc_applied: bool
    base_weight: float
    goalie_confirmed: bool


def _dynamic_anchor_weight(sport: str, base_weight: float, config: SportBetConfig) -> float:
    if str(sport).lower() == "nhl":
        effective_uncertainty, _sample_size, _raw_uncertainty = _nhl_uncertainty_context(
            config, use_floor=False
        )
        mult = float(config.uncertainty_anchor_mult)
        try:
            cap_add = float(os.getenv("NHL_ANCHOR_CAP_ADD", "0.08"))
        except Exception:
            cap_add = 0.08
        adjusted = float(base_weight) + float(mult) * float(effective_uncertainty)
        if cap_add > 0:
            adjusted = min(adjusted, float(base_weight) + float(cap_add))
        try:
            low_unc_mult = float(os.getenv("NHL_LOW_UNC_ANCHOR_MULT", "0.85"))
        except Exception:
            low_unc_mult = 0.85
        try:
            low_unc_threshold = float(os.getenv("NHL_LOW_UNC_THRESHOLD", "0.01"))
        except Exception:
            low_unc_threshold = 0.01
        if effective_uncertainty < float(low_unc_threshold):
            adjusted = float(adjusted) * float(low_unc_mult)
        return float(max(0.0, min(1.0, adjusted)))
    uncertainty = _uncertainty_value(sport)
    mult = float(config.uncertainty_anchor_mult)
    return float(max(0.0, min(1.0, float(base_weight) + float(mult) * float(uncertainty))))


def _nhl_goalie_confirmed(row: pd.Series) -> bool:
    if "goalie_confirmed" in row:
        val = row.get("goalie_confirmed")
        if val is not None and not pd.isna(val):
            return bool(val)
    return not _goalie_unconfirmed(
        row.get("goalie_status"),
        row.get("goalie_home_status"),
        row.get("goalie_away_status"),
    )


def _nhl_anchor_base_weight(goalie_confirmed: bool) -> float:
    if not goalie_confirmed and nhl_goalie_unconfirmed_edge_add(goalie_confirmed) > 0:
        goalie_confirmed = True
    env_key = "NHL_ANCHOR_W" if goalie_confirmed else "NHL_ANCHOR_W_UNCONFIRMED"
    default = "0.50" if goalie_confirmed else "0.60"
    try:
        return float(os.getenv(env_key, default))
    except Exception:
        return float(default)


def nhl_anchor_context_for_row(row: pd.Series, config: SportBetConfig) -> NhlAnchorContext:
    goalie_confirmed = _nhl_goalie_confirmed(row)
    base_weight = _nhl_anchor_base_weight(goalie_confirmed)
    effective_uncertainty, _sample_size, _raw_uncertainty = _nhl_uncertainty_context(
        config, use_floor=False, row=row, market="ML"
    )
    mult = float(config.uncertainty_anchor_mult)
    try:
        cap_add = float(os.getenv("NHL_ANCHOR_CAP_ADD", "0.08"))
    except Exception:
        cap_add = 0.08
    adjusted = float(base_weight) + float(mult) * float(effective_uncertainty)
    if cap_add > 0:
        adjusted = min(adjusted, float(base_weight) + float(cap_add))
    try:
        low_unc_mult = float(os.getenv("NHL_LOW_UNC_ANCHOR_MULT", "0.85"))
    except Exception:
        low_unc_mult = 0.85
    try:
        low_unc_threshold = float(os.getenv("NHL_LOW_UNC_THRESHOLD", "0.01"))
    except Exception:
        low_unc_threshold = 0.01
    low_unc_applied = bool(effective_uncertainty < float(low_unc_threshold))
    if low_unc_applied:
        adjusted = float(adjusted) * float(low_unc_mult)
    anchor_weight = float(max(0.0, min(1.0, adjusted)))
    return NhlAnchorContext(
        anchor_weight=anchor_weight,
        effective_uncertainty=float(effective_uncertainty),
        low_unc_applied=low_unc_applied,
        base_weight=float(base_weight),
        goalie_confirmed=goalie_confirmed,
    )


def nhl_anchor_weight_for_row(row: pd.Series, config: SportBetConfig) -> float:
    return nhl_anchor_context_for_row(row, config).anchor_weight


def nhl_goalie_unconfirmed_edge_add(goalie_confirmed: bool) -> float:
    if goalie_confirmed:
        return 0.0
    try:
        env = os.getenv("NHL_GOALIE_UNKNOWN_MIN_EDGE_BUMP")
        if env is None:
            env = os.getenv("NHL_GOALIE_UNCONF_EDGE_ADD", "0.01")
        return float(env)
    except Exception:
        return 0.01


def nhl_goalie_unknown_penalty() -> float:
    try:
        return float(os.getenv("NHL_GOALIE_UNKNOWN_PENALTY", "0.25"))
    except Exception:
        return 0.25


def nhl_goalie_unknown_max_units() -> float:
    try:
        return float(os.getenv("NHL_GOALIE_UNKNOWN_MAX_UNITS", "0.5"))
    except Exception:
        return 0.5


def nhl_min_edge_cap_value() -> float:
    try:
        return float(os.getenv("NHL_MIN_EDGE_CAP", "0.055"))
    except Exception:
        return 0.055


def _goalie_unconfirmed(
    goalie_status: Optional[str],
    home_status: Optional[str],
    away_status: Optional[str],
) -> bool:
    status = str(goalie_status or "").upper().strip()
    home_status = str(home_status or "").upper().strip()
    away_status = str(away_status or "").upper().strip()
    if home_status or away_status:
        return home_status != "CONFIRMED" or away_status != "CONFIRMED"
    if status:
        return status != "OK"
    return False


def _ml_calibration_samples(sport: str, price: Optional[float]) -> Optional[int]:
    bucket = odds_bucket_from_price(price)
    params = load_calibrator(sport, market_type="ML", bucket=bucket)
    if params is None and bucket is not None:
        params = load_calibrator(sport, market_type="ML", bucket=None)
    if not params:
        return None
    try:
        n = params.get("n_samples")
        if n is None:
            return None
        return int(n)
    except Exception:
        return None


def _apply_ml_sample_shrink(
    p_final: float,
    p_market: float,
    *,
    sport: str,
    price: Optional[float],
) -> Tuple[float, Optional[int], float]:
    if not np.isfinite(p_final):
        return p_final, None, 1.0
    n_samples = _ml_calibration_samples(sport, price)
    if n_samples is None:
        return p_final, None, 1.0
    try:
        n0 = float(os.getenv(f"{str(sport).upper()}_ML_CAL_SHRINK_N0", "200"))
    except Exception:
        n0 = 200.0
    if n0 <= 0:
        return p_final, n_samples, 1.0
    weight = float(n_samples) / (float(n_samples) + float(n0))
    weight = float(max(0.0, min(1.0, weight)))
    anchor = float(p_market) if np.isfinite(p_market) else 0.5
    shrunk = float(weight * float(p_final) + (1.0 - weight) * anchor)
    return shrunk, n_samples, weight


def _calibration_sample_count(
    sport: str,
    *,
    market: str,
    price: Optional[float],
) -> Optional[int]:
    market_key = str(market).upper()
    if market_key == "ML":
        bucket = odds_bucket_from_price(price)
        params = load_calibrator(sport, market_type="ML", bucket=bucket)
        if params is None and bucket is not None:
            params = load_calibrator(sport, market_type="ML", bucket=None)
    else:
        params = load_calibrator(sport, market_type=market_key)
    if not params:
        return None
    try:
        return int(params.get("n_samples", 0))
    except Exception:
        return None


def _calibration_multiplier(
    *,
    uncalibrated: bool,
    sample_count: Optional[int],
    min_samples: int,
    config: SportBetConfig,
) -> float:
    if uncalibrated:
        return float(config.calibration_risk_multiplier)
    if sample_count is None or sample_count <= 0:
        return float(config.calibration_risk_multiplier)
    if min_samples <= 0:
        return 1.0
    weight = min(1.0, float(sample_count) / float(min_samples))
    return float(config.calibration_risk_multiplier + (1.0 - config.calibration_risk_multiplier) * weight)


def _uncertainty_multiplier(uncertainty: float, config: SportBetConfig) -> float:
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return 1.0
    scale = float(config.uncertainty_unit_scale)
    if scale <= 0:
        return 1.0
    mult = 1.0 / (1.0 + float(uncertainty) * float(scale))
    return float(max(0.1, min(1.0, mult)))


def _goalie_multiplier(row: pd.Series, config: SportBetConfig, *, sport: str) -> float:
    if str(sport).lower() != "nhl":
        return 1.0
    goalie_confirmed = _nhl_goalie_confirmed(row)
    if goalie_confirmed:
        return 1.0
    return float(max(0.1, min(1.0, config.goalie_unconfirmed_units_mult)))


def _injury_multiplier(row: pd.Series, config: SportBetConfig, *, sport: str) -> float:
    sport_key = str(sport).lower()
    if sport_key not in {"nba", "nfl"}:
        return 1.0
    injury_conf = str(row.get("injury_confidence", "") or "").lower()
    if injury_conf == "low":
        return float(config.injury_low_units_mult)
    if injury_conf in {"unknown", "missing"}:
        return float(config.injury_unknown_units_mult)
    return 1.0


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
    kelly_mult: float = 0.25
    kelly_max_pct: float = 0.015


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
    calibration_multiplier: float = 1.0
    uncertainty_multiplier: float = 1.0
    goalie_multiplier: float = 1.0
    injury_multiplier: float = 1.0


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
    ev: Optional[float] = None,
    min_ev: float = MIN_EV_TO_PLAY,
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
    if ev is not None and np.isfinite(ev) and ev < float(min_ev):
        return BetDecision("PASS", 0.0, float(unit_dollars), 0.0, f"{reason_prefix}ev<{min_ev:.3f}")

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


def _downgrade_confidence_tier(tier: str) -> str:
    order = ["LOW", "MEDIUM", "HIGH"]
    if tier not in order:
        return tier
    return order[max(0, order.index(tier) - 1)]


def _safe_value_tier(v: Optional[str]) -> str:
    if v is None or str(v).strip() == "":
        return "NO VALUE"
    return str(v)


def _downgrade_value_tier(tier: Optional[str]) -> str:
    t = _safe_value_tier(tier).upper()
    if "HIGH" in t:
        return "MED VALUE"
    if "MED" in t:
        return "LOW VALUE"
    if "LOW" in t or "NO" in t:
        return "NO VALUE"
    return _safe_value_tier(tier)


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
        if model_prob is None or not np.isfinite(model_prob):
            return None, None, spread_price, None, "missing ATS cover probability"
        if market_prob is None or not np.isfinite(market_prob):
            return None, None, spread_price, None, "missing ATS market probability"
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


def _apply_nba_ats_shrink(p_final: float) -> float:
    if not np.isfinite(p_final):
        return p_final
    shrink = float(NBA_ATS_SHRINK)
    if not np.isfinite(shrink) or shrink <= 0:
        return p_final
    return 0.5 + (float(p_final) - 0.5) / float(shrink)


def _apply_goalie_shift(
    p_final: float,
    shift: Optional[float],
    *,
    goalie_status: Optional[str] = None,
    home_status: Optional[str] = None,
    away_status: Optional[str] = None,
) -> float:
    if shift is None or not np.isfinite(shift) or not np.isfinite(p_final):
        return p_final
    adj_shift = float(shift)
    return float(max(0.01, min(0.99, p_final + adj_shift)))


def _goalie_shift_multiplier(
    goalie_status: Optional[str],
    home_status: Optional[str],
    away_status: Optional[str],
) -> float:
    def _normalize(status: Optional[str]) -> str:
        return str(status or "").strip().upper()

    def _mult_for_status(status: str) -> float:
        if status in {"CONFIRMED", "OK"}:
            return 1.0
        if status in {"PROBABLE", "LIKELY"}:
            return 0.75
        if status in {"PROJECTED", "EXPECTED"}:
            return 0.55
        if not status:
            return 0.40
        return 0.40

    home_norm = _normalize(home_status)
    away_norm = _normalize(away_status)
    if home_norm or away_norm:
        return min(_mult_for_status(home_norm), _mult_for_status(away_norm))
    return _mult_for_status(_normalize(goalie_status))


def _goalie_shift_context(
    shift: Optional[float],
    *,
    goalie_status: Optional[str],
    home_status: Optional[str],
    away_status: Optional[str],
) -> Tuple[float, float, bool]:
    if shift is None or not np.isfinite(shift):
        return float("nan"), float("nan"), False
    mult = _goalie_shift_multiplier(goalie_status, home_status, away_status)
    try:
        cap = float(os.getenv("NHL_GOALIE_SHIFT_CAP", "0.020"))
    except Exception:
        cap = 0.02
    scaled = float(shift) * float(mult)
    cap_hit = False
    if np.isfinite(cap) and cap > 0 and abs(scaled) > float(cap):
        cap_hit = True
        scaled = float(np.sign(scaled) * float(cap))
    return float(scaled), float(mult), bool(cap_hit)


def _ml_probabilities_for_side(
    model_home: float,
    market_home: float,
    side: str,
    *,
    sport: str,
    config: SportBetConfig,
    price: Optional[float],
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

    p_cal = calibrate_prob(sport, p_raw, market_type="ML", price=price)
    if not np.isfinite(p_cal):
        p_cal = p_raw

    p_final = _anchor_prob(p_cal, p_market, config)
    p_final, capped = _apply_underdog_cap(p_final, p_market, config)
    if capped:
        flags.append("UNDERDOG_CAP")
    p_final, _ml_cal_n, _ml_shrink_w = _apply_ml_sample_shrink(
        p_final,
        p_market,
        sport=sport,
        price=price,
    )

    return p_raw, p_cal, p_final, p_market, flags


def ml_probabilities_for_row(row: pd.Series, sport: str = "nba") -> Dict[str, float]:
    model_home = safe_float(row.get("model_home_prob_raw", row.get("model_home_prob")))
    market_home = safe_float(row.get("market_home_prob"))
    config = get_sport_bet_config(sport)
    if str(sport).lower() == "nhl":
        anchor_weight = nhl_anchor_weight_for_row(row, config)
    else:
        anchor_weight = _dynamic_anchor_weight(sport, config.anchor_weight, config)
    if anchor_weight != config.anchor_weight:
        config = replace(config, anchor_weight=anchor_weight)

    home_raw, home_cal, home_final, home_market, _ = _ml_probabilities_for_side(
        float(model_home) if model_home is not None else float("nan"),
        float(market_home) if market_home is not None else float("nan"),
        "HOME",
        sport=sport,
        config=config,
        price=safe_float(row.get("home_ml")),
    )
    away_raw, away_cal, away_final, away_market, _ = _ml_probabilities_for_side(
        float(model_home) if model_home is not None else float("nan"),
        float(market_home) if market_home is not None else float("nan"),
        "AWAY",
        sport=sport,
        config=config,
        price=safe_float(row.get("away_ml")),
    )

    home_final_pre_goalie = home_final
    away_final_pre_goalie = away_final
    if sport.lower() == "nhl":
        goalie_shift = safe_float(row.get("goalie_prob_shift"))
        goalie_status = row.get("goalie_status")
        home_status = row.get("goalie_home_status")
        away_status = row.get("goalie_away_status")
        goalie_shift_used = float("nan")
        goalie_shift_mult = float("nan")
        goalie_shift_cap_hit = False
        if goalie_shift is not None and np.isfinite(goalie_shift):
            goalie_shift_used, goalie_shift_mult, goalie_shift_cap_hit = _goalie_shift_context(
                goalie_shift,
                goalie_status=goalie_status,
                home_status=home_status,
                away_status=away_status,
            )
            home_final = _apply_goalie_shift(
                home_final,
                goalie_shift_used,
                goalie_status=goalie_status,
                home_status=home_status,
                away_status=away_status,
            )
            away_final = _apply_goalie_shift(
                away_final,
                -goalie_shift_used,
                goalie_status=goalie_status,
                home_status=home_status,
                away_status=away_status,
            )
        z_home = safe_float(row.get("goalie_home_rating"))
        z_away = safe_float(row.get("goalie_away_rating"))
        diff = safe_float(row.get("goalie_rating_diff"))
        p_before = home_final_pre_goalie
        p_after = home_final
        def _fmt(v: Optional[float]) -> str:
            return f"{float(v):.4f}" if v is not None and np.isfinite(v) else "nan"

        print(
            "[goalie impact] "
            f"{row.get('home')} vs {row.get('away')} | "
            f"z_home={_fmt(z_home)} z_away={_fmt(z_away)} diff={_fmt(diff)} "
            f"shift={_fmt(goalie_shift)} mult={_fmt(goalie_shift_mult)} "
            f"shift_used={_fmt(goalie_shift_used)} p_before={_fmt(p_before)} p_after={_fmt(p_after)}"
        )

    return {
        "model_home_prob_raw": home_raw,
        "model_home_prob_cal": home_cal,
        "model_home_prob_final_pre_goalie": home_final_pre_goalie,
        "model_home_prob_final": home_final,
        "model_away_prob_raw": away_raw,
        "model_away_prob_cal": away_cal,
        "model_away_prob_final_pre_goalie": away_final_pre_goalie,
        "model_away_prob_final": away_final,
        "market_home_prob": home_market,
        "market_away_prob": away_market,
        "goalie_shift_used": goalie_shift_used if sport.lower() == "nhl" else float("nan"),
        "goalie_shift_mult": goalie_shift_mult if sport.lower() == "nhl" else float("nan"),
        "goalie_shift_cap_hit": goalie_shift_cap_hit if sport.lower() == "nhl" else False,
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
    float,
    str,
    str,
    str,
    bool,
    Optional[float],
    Optional[str],
    str,
    float,
    float,
    float,
]:
    uncalibrated = False
    value_tier = "NO VALUE"
    conf = "LOW"
    primary_market = "NONE"
    primary_side = "NONE"
    primary_ev = float("nan")
    abs_edge = float("nan")
    decision_flags: List[str] = []

    primary_market, primary_side = _primary_market_and_side(row)
    p_model_raw, p_market, primary_price, opp_price, data_reason = _primary_probabilities(
        row, primary_market, primary_side
    )

    config = get_sport_bet_config(sport)
    goalie_confirmed = None
    if str(sport).lower() == "nhl":
        goalie_confirmed = _nhl_goalie_confirmed(row)
        nhl_anchor_weight = nhl_anchor_weight_for_row(row, config)
        if nhl_anchor_weight != config.anchor_weight:
            config = replace(config, anchor_weight=nhl_anchor_weight)
    p_model_raw_val = float(p_model_raw) if p_model_raw is not None and np.isfinite(p_model_raw) else float("nan")
    p_market_val = float(p_market) if p_market is not None and np.isfinite(p_market) else float("nan")

    existing_flags = normalize_decision_flags(row.get("decision_flags") or "")
    existing_flag_list = [f for f in existing_flags.split(",") if f]

    calibrator_missing = False
    calibrator_samples = None
    market_key = str(primary_market).upper()
    price_val = safe_float(primary_price)
    if market_key == "ML":
        price_for_bucket = price_val if price_val is not None and np.isfinite(price_val) else None
        calibrator_params = load_calibrator(
            sport,
            market_type="ML",
            bucket=odds_bucket_from_price(price_for_bucket),
        )
        if calibrator_params is None and price_for_bucket is not None:
            calibrator_params = load_calibrator(sport, market_type="ML", bucket=None)
        calibrator_missing = calibrator_params is None
        if calibrator_params is not None:
            try:
                calibrator_samples = int(calibrator_params.get("n_samples", 0))
            except Exception:
                calibrator_samples = 0
            if calibrator_samples <= 0:
                calibrator_missing = True
    p_model_cal = calibrate_prob(
        sport,
        p_model_raw_val,
        market_type=primary_market,
        price=price_val if price_val is not None and np.isfinite(price_val) else None,
    )
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
        model_raw_home = safe_float(row.get("model_home_prob_raw", row.get("model_home_prob")))
        model_cal_home = safe_float(row.get("model_home_prob_cal"))
        model_final_home = safe_float(row.get("model_home_prob_final"))
        side = (primary_side or "").upper()

        def _side_prob(val: Optional[float]) -> float:
            if val is None or not np.isfinite(val):
                return float("nan")
            return float(1.0 - val) if side == "AWAY" else float(val)

        p_model_raw_val = _side_prob(model_raw_home)
        p_model_cal = _side_prob(model_cal_home)
        p_model_final = _side_prob(model_final_home)
        model_final_available = np.isfinite(p_model_final)

        uncalibrated = not np.isfinite(p_model_cal) or not np.isfinite(p_model_final) or calibrator_missing
        if calibrator_samples is not None and calibrator_samples <= 0:
            flags.append("UNCALIBRATED_ZERO_SAMPLES")

            if not np.isfinite(p_model_cal):
                p_model_cal = calibrate_prob(
                    sport,
                    p_model_raw_val,
                    market_type=primary_market,
                    price=price_val if price_val is not None and np.isfinite(price_val) else None,
                )
        if not np.isfinite(p_model_cal):
            p_model_cal = p_model_raw_val

        if not np.isfinite(p_model_final):
            p_model_final = _anchor_prob(p_model_cal, p_market_val, config)

        if np.isfinite(p_model_final):
            p_model_final, capped = _apply_underdog_cap(p_model_final, p_market_val, config)
            if capped:
                flags.append("UNDERDOG_CAP")
            if np.isfinite(p_model_final) and not model_final_available:
                p_model_final, _ml_cal_n, _ml_shrink_w = _apply_ml_sample_shrink(
                    p_model_final,
                    p_market_val,
                    sport=sport,
                    price=price_val if price_val is not None and np.isfinite(price_val) else None,
                )
        if sport.lower() == "nhl":
            goalie_shift = safe_float(row.get("goalie_prob_shift"))
            goalie_confirmed = _nhl_goalie_confirmed(row)
            if np.isfinite(p_model_final):
                goalie_shift_used, _goalie_mult, _goalie_cap = _goalie_shift_context(
                    goalie_shift,
                    goalie_status=row.get("goalie_status"),
                    home_status=row.get("goalie_home_status"),
                    away_status=row.get("goalie_away_status"),
                )
                p_model_final = _apply_goalie_shift(
                    p_model_final,
                    goalie_shift_used,
                    goalie_status=row.get("goalie_status"),
                    home_status=row.get("goalie_home_status"),
                    away_status=row.get("goalie_away_status"),
                )
            if not goalie_confirmed and np.isfinite(p_model_final) and np.isfinite(p_market_val):
                penalty = nhl_goalie_unknown_penalty()
                if penalty > 0:
                    shrink = float(max(0.0, 1.0 - float(penalty)))
                    edge_prob_final = float(p_model_final - p_market_val) * shrink
                    p_model_final = float(p_market_val + edge_prob_final)
                    flags.append("GOALIE_UNKNOWN_PENALTY")

        edge_prob_final = (
            float(p_model_final) - float(p_market_val)
            if np.isfinite(p_model_final) and np.isfinite(p_market_val)
            else float("nan")
        )
    if primary_market == "ATS":
        ats_cal_used = row.get("ats_cal_used")
        if (
            (ats_cal_used is False)
            or ("ATS_UNCALIBRATED_MARGIN" in existing_flag_list)
            or ("ATS_GATED_UNCALIBRATED_MARGIN" in existing_flag_list)
        ):
            uncalibrated = True
        calibrator_params = load_calibrator(sport, market_type="ATS")
        if calibrator_params is None:
            uncalibrated = True
        if not np.isfinite(p_model_cal):
            p_model_cal = p_model_raw_val
        if not np.isfinite(p_model_final):
            p_model_final = p_model_cal
        if str(sport).lower() == "nba" and np.isfinite(p_model_final):
            p_model_final = _apply_nba_ats_shrink(p_model_final)
        if np.isfinite(p_model_final) and np.isfinite(p_market_val):
            edge_prob_final = float(p_model_final) - float(p_market_val)
    if primary_market == "TOTAL":
        calibrator_params = load_calibrator(sport, market_type="TOTAL")
        if calibrator_params is None:
            uncalibrated = True

    if uncalibrated:
        flags.append("UNCALIBRATED_FALLBACK")
        if primary_market == "ATS":
            flags.append("ATS_UNCALIBRATED")

    value_tier = value_tier_from_edge(edge_prob_final, config.min_edge_cal)
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
    if str(sport).lower() == "nhl" and primary_market == "ML":
        goalie_confirmed = _nhl_goalie_confirmed(row)
        if not goalie_confirmed:
            conf = _downgrade_confidence_tier(conf)
            value_tier = _downgrade_value_tier(_safe_value_tier(value_tier))
            if conf_reason:
                conf_reason = f"{conf_reason}, GOALIE_UNCONFIRMED"
            else:
                conf_reason = "GOALIE_UNCONFIRMED"
    if uncalibrated:
        conf = _downgrade_confidence_tier(conf)
        if conf_reason:
            conf_reason = f"{conf_reason}, UNCALIBRATED_CONF_DOWN"
        else:
            conf_reason = "UNCALIBRATED_CONF_DOWN"

    edge_shift = (
        abs(float(edge_prob_raw - edge_prob_final))
        if np.isfinite(edge_prob_raw) and np.isfinite(edge_prob_final)
        else 0.0
    )
    if edge_shift >= float(EDGE_SHIFT_DOWNGRADE):
        conf = _downgrade_confidence_tier(conf)
        value_tier = _downgrade_value_tier(_safe_value_tier(value_tier))
        flags.append("EDGE_SHIFT_DOWNGRADE")

    primary_ev = (
        ev_per_dollar(p_model_final, primary_price)
        if primary_price is not None and np.isfinite(primary_price)
        else float("nan")
    )

    extra_edge = 0.0
    decision_flags = [f for f in flags]
    if "UNCALIBRATED_FALLBACK" in existing_flags and "UNCALIBRATED_FALLBACK" not in decision_flags:
        decision_flags.append("UNCALIBRATED_FALLBACK")
    if "ATS_UNCALIBRATED" in existing_flags and "ATS_UNCALIBRATED" not in decision_flags:
        decision_flags.append("ATS_UNCALIBRATED")
    if "UNCALIBRATED_FALLBACK" in decision_flags:
        extra_edge = float(config.uncalibrated_edge_add)
    if primary_market == "ATS" and "ATS_UNCALIBRATED" in decision_flags:
        extra_edge = max(extra_edge, float(ATS_UNCALIBRATED_EDGE_ADD))

    min_edge_dynamic = _dynamic_min_edge(
        sport,
        config.min_edge_cal,
        config,
        row=row,
        market=primary_market,
    ) + extra_edge
    if str(sport).lower() == "nhl":
        goalie_confirmed = _nhl_goalie_confirmed(row)
        if not goalie_confirmed and "GOALIE_UNCONFIRMED" not in decision_flags:
            decision_flags.append("GOALIE_UNCONFIRMED")
        min_edge_dynamic += nhl_goalie_unconfirmed_edge_add(goalie_confirmed)
        min_edge_cap = nhl_min_edge_cap_value()
        min_edge_dynamic_before_cap = float(min_edge_dynamic)
        # Example: edge_prob_raw=0.043 -> edge_prob_final=0.034, min_edge ~0.064 -> capped at 0.055.
        min_edge_dynamic = min(float(min_edge_dynamic_before_cap), float(min_edge_cap))
        if min_edge_dynamic_before_cap > float(min_edge_cap) + 1e-6:
            print(
                "[nhl min edge cap] "
                f"pre_cap={min_edge_dynamic_before_cap:.6f} cap={float(min_edge_cap):.6f}"
            )
        assert float(min_edge_dynamic) <= float(min_edge_cap) + 1e-6

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
        bool(uncalibrated),
        primary_price,
        data_reason,
        normalize_decision_flags(decision_flags),
        primary_ev,
        float(min_edge_dynamic),
        float(min_edge_dynamic),
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
    config = get_sport_bet_config(sport)
    existing_flags = [f for f in str(row.get("decision_flags") or "").split(",") if f]
    for flag in existing_flags:
        if flag not in flags:
            flags.append(flag)
    existing_reason = str(row.get("decision_reason") or "").strip()
    if existing_reason:
        reason_parts.append(existing_reason)

    uncertainty_effective = _uncertainty_value(sport)
    if not np.isfinite(uncertainty_effective):
        uncertainty_effective = float(config.uncertainty_floor)
    uncertainty_raw = float(uncertainty_effective)
    uncertainty_quality = 1.0
    uncertainty_samples = None

    nhl_uncertainty_used = None
    nhl_uncertainty_samples = None
    nhl_uncertainty_raw = None

    def _fmt(x: float) -> str:
        return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"

    def _print_decision(
        decision: DecisionOutcome,
        *,
        p_model_raw_val: float = float("nan"),
        p_model_cal_val: float = float("nan"),
        p_model_final_val: float = float("nan"),
        p_market_val: float = float("nan"),
        edge_prob_raw_val: float = float("nan"),
        edge_prob_final_val: float = float("nan"),
        min_edge_dyn: float = float("nan"),
    ) -> None:
        print(
            f"[decision] {row.get('home','')} vs {row.get('away','')} "
            f"model_raw={_fmt(p_model_raw_val)} model_cal={_fmt(p_model_cal_val)} "
            f"model_final={_fmt(p_model_final_val)} market={_fmt(p_market_val)} "
            f"edge_raw={_fmt(edge_prob_raw_val)} edge_final={_fmt(edge_prob_final_val)} "
            f"uncertainty={uncertainty_effective:.3f} min_edge_dyn={_fmt(min_edge_dyn)} "
            f"decision={decision.play_pass} reason={decision.reason}"
        )

    if require_pick and ("PICK" not in primary.upper()):
        decision = DecisionOutcome(
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
        _print_decision(decision)
        return decision

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
        _uncalibrated,
        primary_price,
        data_reason,
        metric_flags,
        primary_ev,
        min_play_edge_abs_used,
        min_primary_edge_abs_used,
    ) = primary_metrics_for_row(row, sport=sport, settings=settings)

    uncertainty_ctx = _effective_uncertainty(
        sport,
        config,
        use_floor=True,
        row=row,
        market=primary_market,
    )
    uncertainty_effective = float(uncertainty_ctx[0])
    uncertainty_samples = uncertainty_ctx[1]
    uncertainty_raw = float(uncertainty_ctx[2])
    uncertainty_quality = float(uncertainty_ctx[3])
    uncertainty_sample_factor = float(uncertainty_ctx[4])
    uncertainty_quality_factor = float(uncertainty_ctx[5])
    _log_uncertainty_once(
        sport,
        uncertainty_raw,
        config.min_edge_cal,
        config,
        effective_uncertainty=uncertainty_effective,
        sample_size=uncertainty_samples,
        quality=uncertainty_quality,
        sample_factor=uncertainty_sample_factor,
        quality_factor=uncertainty_quality_factor,
    )

    if str(sport).lower() == "nhl":
        nhl_uncertainty_used, nhl_uncertainty_samples, nhl_uncertainty_raw = _nhl_uncertainty_context(
            config, use_floor=True, row=row, market=primary_market
        )

    if str(sport).lower() == "nhl":
        anchor_ctx = nhl_anchor_context_for_row(row, config)
        eff_unc = nhl_uncertainty_used if nhl_uncertainty_used is not None else anchor_ctx.effective_uncertainty
        reason_parts.append(
            "min_edge="
            f"{min_primary_edge_abs_used:.3f} unc={uncertainty_raw:.3f} eff_unc={eff_unc:.3f} "
            f"goalie_confirmed={anchor_ctx.goalie_confirmed} anchor_w_used={anchor_ctx.anchor_weight:.3f}"
        )

    if data_reason or not np.isfinite(p_model_cal) or not np.isfinite(p_market):
        flags.append("MISSING_DATA_PASS")
        decision = DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            f"NO BET: {data_reason or 'missing probabilities'}",
            ",".join(flags),
            f"NO BET: {data_reason or 'missing probabilities'}",
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
        _print_decision(
            decision,
            p_model_raw_val=p_model_raw,
            p_model_cal_val=p_model_cal,
            p_model_final_val=p_model_final,
            p_market_val=p_market,
            edge_prob_raw_val=edge_prob_raw,
            edge_prob_final_val=edge_prob_final,
            min_edge_dyn=min_primary_edge_abs_used,
        )
        return decision

    ev = primary_ev
    abs_edge = abs(edge_prob_final) if np.isfinite(edge_prob_final) else float("nan")

    if primary_market == "ML" and primary_price is not None and np.isfinite(primary_price):
        if (float(primary_price) >= float(EXTREME_ML_POS_ODDS)) or (
            float(primary_price) <= float(EXTREME_ML_NEG_ODDS)
        ):
            if not np.isfinite(abs_edge) or abs_edge < float(EXTREME_EDGE_REQUIREMENT):
                flags.append("EXTREME_ODDS_PASS")
                decision = DecisionOutcome(
                    "PASS",
                    0.0,
                    float(unit_dollars),
                    0.0,
                    "PASS: extreme odds require huge edge",
                    ",".join(flags),
                    "EXTREME_ODDS_PASS",
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
                _print_decision(
                    decision,
                    p_model_raw_val=p_model_raw,
                    p_model_cal_val=p_model_cal,
                    p_model_final_val=p_model_final,
                    p_market_val=p_market,
                    edge_prob_raw_val=edge_prob_raw,
                    edge_prob_final_val=edge_prob_final,
                    min_edge_dyn=min_primary_edge_abs_used,
                )
                return decision

    if str(sport).lower() == "nba" and primary_market == "ML":
        decision_conf = str(_conf or "").upper()
        is_extreme = False
        if primary_price is not None and np.isfinite(primary_price):
            is_extreme = (float(primary_price) >= float(EXTREME_ML_POS_ODDS)) or (
                float(primary_price) <= float(EXTREME_ML_NEG_ODDS)
            )
        if decision_conf == "HIGH" and not is_extreme:
            override_val = 0.045
            if NBA_ML_MIN_EDGE_OVERRIDE is not None and str(NBA_ML_MIN_EDGE_OVERRIDE).strip():
                try:
                    override_val = float(NBA_ML_MIN_EDGE_OVERRIDE)
                except Exception:
                    override_val = 0.045
            if np.isfinite(override_val) and override_val > 0:
                min_primary_edge_abs_used = min(float(min_primary_edge_abs_used), float(override_val))

    if not np.isfinite(edge_prob_final):
        decision = DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            "NO BET: missing edge",
            "MISSING_EDGE_PASS",
            "NO BET: missing edge",
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
        _print_decision(
            decision,
            p_model_raw_val=p_model_raw,
            p_model_cal_val=p_model_cal,
            p_model_final_val=p_model_final,
            p_market_val=p_market,
            edge_prob_raw_val=edge_prob_raw,
            edge_prob_final_val=edge_prob_final,
            min_edge_dyn=min_primary_edge_abs_used,
        )
        return decision
    if float(edge_prob_final) <= 0.0:
        decision = DecisionOutcome(
            "PASS",
            0.0,
            float(unit_dollars),
            0.0,
            "NO BET: edge_final<=0",
            "NON_POSITIVE_EDGE",
            "NO BET: edge_final<=0",
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
        _print_decision(
            decision,
            p_model_raw_val=p_model_raw,
            p_model_cal_val=p_model_cal,
            p_model_final_val=p_model_final,
            p_market_val=p_market,
            edge_prob_raw_val=edge_prob_raw,
            edge_prob_final_val=edge_prob_final,
            min_edge_dyn=min_primary_edge_abs_used,
        )
        return decision
    if abs_edge < float(min_primary_edge_abs_used):
        flags.append("LOW_EDGE_SIZE_DOWN")
        reason_parts.append(
            f"soft_edge_play<{min_primary_edge_abs_used:.3f}>=floor{float(config.min_edge_soft_floor):.3f}"
        )

    min_conf = str(min_confidence or "").upper()
    conf_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if str(sport).lower() == "nba" and min_conf in conf_rank:
        decision_conf = str(_conf or "").upper()
        if conf_rank.get(decision_conf, -1) < conf_rank[min_conf]:
            flags.append("LOW_CONFIDENCE_SIZE_DOWN")
            reason_parts.append(f"confidence<{min_conf}")

    base_units = default_bet_units_from_tier(value_tier)
    kelly_units = float("nan")
    if primary_price is not None and np.isfinite(primary_price) and np.isfinite(p_model_final):
        bankroll = float(unit_dollars) / float(settings.flat_pct) if settings.flat_pct > 0 else float(unit_dollars)
        kelly_bet_size = bet_size_kelly_ml(
            bankroll,
            float(p_model_final),
            float(primary_price),
            kelly_mult=settings.kelly_mult,
            max_pct=settings.kelly_max_pct,
        )
        if np.isfinite(kelly_bet_size) and unit_dollars > 0:
            kelly_units = float(kelly_bet_size) / float(unit_dollars)
    if np.isfinite(kelly_units):
        base_units = float(kelly_units)

    if "LOW_EDGE_SIZE_DOWN" in flags and base_units <= 0:
        base_units = float(config.soft_edge_base_units)
    bet_size = float(base_units * float(unit_dollars))

    raw_units = bet_size / float(unit_dollars) if unit_dollars > 0 else 0.0

    calibration_samples = _calibration_sample_count(
        sport,
        market=primary_market,
        price=primary_price if np.isfinite(primary_price) else None,
    )
    calibration_multiplier = _calibration_multiplier(
        uncalibrated=bool(_uncalibrated),
        sample_count=calibration_samples,
        min_samples=int(MIN_SAMPLES),
        config=config,
    )
    uncertainty_for_sizing = _uncertainty_for_threshold(
        sport,
        config,
        market=primary_market,
        row=row,
    )
    uncertainty_multiplier = _uncertainty_multiplier(uncertainty_for_sizing, config)
    goalie_multiplier = _goalie_multiplier(row, config, sport=sport)
    injury_multiplier = _injury_multiplier(row, config, sport=sport)
    ev_multiplier = 1.0
    if not np.isfinite(ev) or ev <= 0.0:
        flags.append("LOW_EV_SIZE_DOWN")
        reason_parts.append("ev<=0")
        ev_multiplier = float(os.getenv("EV_NEGATIVE_UNITS_MULT", "0.5"))

    multiplier_stack = float(
        calibration_multiplier
        * uncertainty_multiplier
        * goalie_multiplier
        * injury_multiplier
        * ev_multiplier
    )
    final_units = float(raw_units) * float(multiplier_stack)
    final_units = min(final_units, float(config.max_units))

    if "UNCALIBRATED_FALLBACK" in flags:
        reason_parts.append("calibration_risk_mult")

    if uncertainty_for_sizing >= float(config.uncertainty_flat_threshold):
        flags.append("UNCERTAINTY_FLAT")
        final_units = min(final_units, float(config.flat_units_when_uncertain))
        reason_parts.append("uncertainty_cap")

    if primary_market == "ML" and primary_price is not None and np.isfinite(primary_price):
        if float(primary_price) >= float(config.longshot_odds):
            flags.append("LONGSHOT_CAP")
            final_units = min(final_units, float(config.longshot_cap_units))
            reason_parts.append(f"longshot {primary_price}>=+{config.longshot_odds:.0f}")

    if metric_flags:
        for flag in metric_flags.split(","):
            if flag:
                flags.append(flag)

    if str(sport).lower() == "nhl":
        if "GOALIE_UNCONFIRMED" in flags or "GOALIE_UNKNOWN_PENALTY" in flags:
            max_units = nhl_goalie_unknown_max_units()
            if max_units > 0:
                if final_units > float(max_units):
                    flags.append("GOALIE_UNKNOWN_CAP")
                    final_units = min(final_units, float(max_units))
                    reason_parts.append(f"goalie_unconfirmed_cap<{float(max_units):.2f}")

    disagreement = abs(float(p_model_cal - p_market)) if np.isfinite(p_model_cal) and np.isfinite(p_market) else 0.0
    if disagreement > float(config.disagree_pass_edge):
        flags.append("DISAGREE_PASS")
        decision = DecisionOutcome(
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
        _print_decision(
            decision,
            p_model_raw_val=p_model_raw,
            p_model_cal_val=p_model_cal,
            p_model_final_val=p_model_final,
            p_market_val=p_market,
            edge_prob_raw_val=edge_prob_raw,
            edge_prob_final_val=edge_prob_final,
            min_edge_dyn=min_primary_edge_abs_used,
        )
        return decision

    if disagreement > float(config.disagree_cap_edge):
        flags.append("DISAGREE_CAP")
        reason_parts.append(f"|model-market|={disagreement:.3f}>{float(config.disagree_cap_edge):.3f}")
        final_units = min(final_units, float(config.disagree_cap_units))

    ats_uncalibrated = primary_market == "ATS" and (
        "ATS_UNCALIBRATED" in flags or "ATS_UNCALIBRATED_MARGIN" in flags
    )
    if ats_uncalibrated:
        allow_override = np.isfinite(abs_edge) and abs_edge >= float(ATS_UNCALIBRATED_EDGE_OVERRIDE)
        if "DISAGREE_CAP" in flags:
            allow_override = False
        if not allow_override:
            flags.append("ATS_UNCALIBRATED_CAP")
            reason_parts.append("ats_uncalibrated_cap")
            final_units = min(final_units, float(ATS_UNCALIBRATED_MAX_UNITS))

    if (
        final_units > 0
        and float(config.test_bet_min_units_enabled)
        and final_units < float(config.test_bet_min_units)
    ):
        if not {"LONGSHOT_CAP", "DISAGREE_CAP"}.intersection(flags):
            flags.append("MIN_UNIT_FLOOR")
            final_units = float(config.test_bet_min_units)

    bet_size = float(final_units * float(unit_dollars))
    play_pass = "PLAY" if final_units > 0 else "PASS"
    decision_reason = ", ".join(reason_parts) if reason_parts else ""
    if flags and not decision_reason:
        decision_reason = ",".join(flags)

    decision = DecisionOutcome(
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
        calibration_multiplier,
        uncertainty_multiplier,
        goalie_multiplier,
        injury_multiplier,
    )
    _print_decision(
        decision,
        p_model_raw_val=p_model_raw,
        p_model_cal_val=p_model_cal,
        p_model_final_val=p_model_final,
        p_market_val=p_market,
        edge_prob_raw_val=edge_prob_raw,
        edge_prob_final_val=edge_prob_final,
        min_edge_dyn=min_primary_edge_abs_used,
    )
    return decision


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
        decision_flags = [f for f in str(r.get("decision_flags") or "").split(",") if f]
        if str(sport).lower() == "nba" and (
            "ATS_UNCALIBRATED_MARGIN" in decision_flags
            or "ATS_GATED_UNCALIBRATED_MARGIN" in decision_flags
        ):
            ats_ev = float("nan")

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
    out["uncalibrated"] = [m[12] for m in metrics]
    out["primary_price"] = [m[13] for m in metrics]
    out["primary_ev"] = [m[16] for m in metrics]
    out["min_play_edge_abs_used"] = [m[17] for m in metrics]
    out["min_primary_edge_abs_used"] = [m[18] for m in metrics]
    if str(sport).lower() == "nhl":
        anchor_contexts = [nhl_anchor_context_for_row(r, get_sport_bet_config("nhl")) for _, r in out.iterrows()]
        nhl_uncertainty_used, nhl_uncertainty_samples, nhl_uncertainty_raw = _nhl_uncertainty_context(
            get_sport_bet_config("nhl"), use_floor=True
        )
        out["nhl_uncertainty"] = float(nhl_uncertainty_raw)
        out["nhl_uncertainty_n"] = float(nhl_uncertainty_samples)
        out["nhl_uncertainty_effective"] = float(nhl_uncertainty_used)
        out["nhl_uncertainty_used"] = float(nhl_uncertainty_used)
        out["nhl_uncertainty_samples"] = (
            int(nhl_uncertainty_samples) if nhl_uncertainty_samples is not None else np.nan
        )
        out["nhl_min_edge_cap_used"] = nhl_min_edge_cap_value()
        out["nhl_anchor_w_used"] = [ctx.anchor_weight for ctx in anchor_contexts]
        out["nhl_low_unc_applied"] = [ctx.low_unc_applied for ctx in anchor_contexts]
        out["nhl_goalie_unconf_edge_add"] = [
            nhl_goalie_unconfirmed_edge_add(ctx.goalie_confirmed) for ctx in anchor_contexts
        ]
        out["effective_uncertainty"] = [ctx.effective_uncertainty for ctx in anchor_contexts]
        out["nhl_effective_uncertainty"] = [ctx.effective_uncertainty for ctx in anchor_contexts]
    else:
        out["nhl_uncertainty"] = np.nan
        out["nhl_uncertainty_n"] = np.nan
        out["nhl_uncertainty_effective"] = np.nan
        out["nhl_uncertainty_used"] = np.nan
        out["nhl_uncertainty_samples"] = np.nan
        out["nhl_min_edge_cap_used"] = np.nan
        out["nhl_anchor_w_used"] = np.nan
        out["nhl_low_unc_applied"] = np.nan
        out["nhl_goalie_unconf_edge_add"] = np.nan
        out["effective_uncertainty"] = np.nan
        out["nhl_effective_uncertainty"] = np.nan

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
    out["calibration_multiplier"] = [d.calibration_multiplier for d in decisions]
    out["uncertainty_multiplier"] = [d.uncertainty_multiplier for d in decisions]
    out["goalie_multiplier"] = [d.goalie_multiplier for d in decisions]
    out["injury_multiplier"] = [d.injury_multiplier for d in decisions]
    out["edge_prob_raw"] = [d.edge_prob_raw for d in decisions]
    out["edge_prob_cal"] = [d.edge_prob_cal for d in decisions]
    out["edge_prob_final"] = [d.edge_prob_final for d in decisions]
    out["edge_shrink_factor"] = [
        (float(fp) / float(fr))
        if fp is not None
        and fr is not None
        and np.isfinite(fp)
        and np.isfinite(fr)
        and abs(float(fr)) > 1e-9
        else np.nan
        for fp, fr in zip(out["edge_prob_final"], out["edge_prob_raw"])
    ]
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
