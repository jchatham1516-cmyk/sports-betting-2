"""Helpers for logging daily plays into a persistent bet log."""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sports.common.util import safe_float


BET_LOG_COLUMNS = [
    "bet_id",
    "date",
    "sport",
    "home",
    "away",
    "market_type",
    "side",
    "line_at_bet",
    "price_at_bet",
    "model_prob",
    "market_prob",
    "edge",
    "confidence",
    "value_tier",
    "units",
    "result",
    "unit_dollars",
    "stake_dollars",
]


def _hash_bet_id(*parts: object) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _safe_number(value: object) -> float:
    val = safe_float(value)
    try:
        return float(val)
    except Exception:
        return float("nan")


def _extract_probs(row: pd.Series, side: str) -> (float, float):
    model_prob = _safe_number(row.get("model_home_prob"))
    market_prob = _safe_number(row.get("market_home_prob"))
    if side == "AWAY":
        if np.isfinite(model_prob):
            model_prob = 1.0 - model_prob
        if np.isfinite(market_prob):
            market_prob = 1.0 - market_prob
    return model_prob, market_prob


def _edge_for_row(row: pd.Series, market_type: str) -> object:
    market = market_type.lower()
    if market == "moneyline":
        return row.get("primary_ev", row.get("ml_ev_best"))
    if market == "spread":
        return row.get("primary_ev", row.get("ats_ev_best"))
    if market == "total":
        return row.get("primary_ev", row.get("total_ev_best"))
    return None


def _build_bet_row(row: pd.Series, sport: str) -> Optional[Dict[str, object]]:
    play_flag = str(row.get("play_pass", "")).upper()
    if play_flag != "PLAY":
        return None

    units = _safe_number(row.get("units"))
    if not np.isfinite(units) or units <= 0:
        return None

    primary_market = str(row.get("primary_market", "")).upper()
    primary_side = str(row.get("primary_side", "")).upper()
    if not primary_market or not primary_side:
        return None

    market_type: Optional[str] = None
    line_at_bet: object = ""
    price_at_bet: object = np.nan
    model_prob = np.nan
    market_prob = np.nan

    if "ML" in primary_market:
        market_type = "moneyline"
        price_at_bet = row.get("home_ml") if primary_side == "HOME" else row.get("away_ml")
        model_prob, market_prob = _extract_probs(row, primary_side)
    elif "ATS" in primary_market or "SPREAD" in primary_market:
        market_type = "spread"
        line_at_bet = row.get("home_spread")
        price_at_bet = row.get("spread_price")
    elif "TOTAL" in primary_market:
        market_type = "total"
        line_at_bet = row.get("total_points")
        if primary_side == "OVER":
            price_at_bet = row.get("total_over_price")
        elif primary_side == "UNDER":
            price_at_bet = row.get("total_under_price")

    if market_type is None:
        return None

    p_model_used = _safe_number(row.get("p_model_used"))
    p_market_used = _safe_number(row.get("p_market_used"))
    if np.isfinite(p_model_used):
        model_prob = p_model_used
    if np.isfinite(p_market_used):
        market_prob = p_market_used

    date_str = row.get("date")
    bet_id = _hash_bet_id(
        date_str,
        sport,
        row.get("home"),
        row.get("away"),
        market_type,
        primary_side,
        line_at_bet,
        price_at_bet,
    )

    unit_dollars = _safe_number(row.get("unit_dollars"))
    if not np.isfinite(unit_dollars):
        unit_dollars = 10.0

    stake_dollars = units * unit_dollars

    return {
        "bet_id": bet_id,
        "date": date_str,
        "sport": sport,
        "home": row.get("home"),
        "away": row.get("away"),
        "market_type": market_type,
        "side": primary_side,
        "line_at_bet": line_at_bet if not pd.isna(line_at_bet) else "",
        "price_at_bet": price_at_bet,
        "model_prob": model_prob,
        "market_prob": market_prob,
        "edge": _edge_for_row(row, market_type),
        "confidence": row.get("confidence"),
        "value_tier": row.get("value_tier"),
        "units": units,
        "result": row.get("result", ""),
        "unit_dollars": unit_dollars,
        "stake_dollars": stake_dollars,
    }


def append_plays_to_bet_log(
    predictions_df: pd.DataFrame,
    sport: str,
    bet_log_path: str = "results/tracking/bet_log.csv",
) -> int:
    """Append qualifying PLAY bets to the persistent bet_log.

    Returns number of new bets written (deduped by bet_id).
    """

    if predictions_df is None or predictions_df.empty:
        return 0

    bets: List[Dict[str, object]] = []
    for _, row in predictions_df.iterrows():
        bet = _build_bet_row(row, sport)
        if bet:
            bets.append(bet)

    if not bets:
        return 0

    os.makedirs(os.path.dirname(bet_log_path) or ".", exist_ok=True)

    new_df = pd.DataFrame(bets)
    if os.path.exists(bet_log_path):
        existing = pd.read_csv(bet_log_path)
    else:
        existing = pd.DataFrame(columns=BET_LOG_COLUMNS)

    combined = pd.concat([existing, new_df], ignore_index=True)

    # Ensure all expected columns exist for consistency
    for col in set(BET_LOG_COLUMNS + list(new_df.columns) + list(existing.columns)):
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined.drop_duplicates(subset=["bet_id"], keep="first")
    combined.to_csv(bet_log_path, index=False)

    return max(0, len(combined) - len(existing))
