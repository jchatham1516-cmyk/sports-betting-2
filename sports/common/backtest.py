"""Simple backtest utilities to evaluate calibration + EV."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from sports.common.eval import brier_score, calibration_table


def _log_loss(y_true: Iterable[float], p: Iterable[float]) -> float:
    y = np.array(list(y_true), dtype=float)
    p_arr = np.clip(np.array(list(p), dtype=float), 1e-6, 1 - 1e-6)
    mask = ~np.isnan(y) & ~np.isnan(p_arr)
    if mask.sum() == 0:
        return float("nan")
    y = y[mask]
    p_arr = p_arr[mask]
    return float(-np.mean(y * np.log(p_arr) + (1 - y) * np.log(1 - p_arr)))


def _ev_from_price(prob: float, price: float) -> float:
    prob = float(prob)
    price = float(price)
    if math.isnan(prob) or math.isnan(price):
        return float("nan")
    payout = price / 100.0 if price > 0 else 100.0 / abs(price)
    lose = 1.0
    return prob * payout - (1 - prob) * lose


def backtest_metrics(df_preds: pd.DataFrame, df_results: pd.DataFrame) -> Dict[str, object]:
    """Compute calibration + ROI style diagnostics without mutating production flow."""

    joined = df_preds.merge(
        df_results[["home", "away", "home_score", "away_score"]],
        how="inner",
        on=["home", "away"],
        suffixes=("", "_res"),
    )

    joined["actual_home_win"] = (joined["home_score"] > joined["away_score"]).astype(float)

    metrics: Dict[str, object] = {}
    metrics["brier"] = brier_score(joined["actual_home_win"], joined["model_home_prob"])
    metrics["log_loss"] = _log_loss(joined["actual_home_win"], joined["model_home_prob"])

    # ATS / totals hit rate when corresponding columns exist
    if {"home_spread", "model_spread_home"}.issubset(joined.columns):
        joined["ats_hit"] = ((joined["home_score"] + joined["home_spread"]) > joined["away_score"]).astype(float)
        metrics["ats_hit_rate"] = float(joined["ats_hit"].mean())

    if {"total_points", "home_score", "away_score"}.issubset(joined.columns):
        joined["actual_total"] = joined["home_score"] + joined["away_score"]
        joined["totals_hit"] = (joined["actual_total"] > joined["total_points"]).astype(float)
        metrics["totals_hit_rate"] = float(joined["totals_hit"].mean())

    # ROI style metrics using EV columns when present
    for col in ["ev_ml", "ev_spread", "ev_total"]:
        if col in joined.columns:
            metrics[f"expected_roi_{col}"] = float(np.nanmean(joined[col]))

    # Calibration bins
    metrics["calibration_bins"] = calibration_table(joined["actual_home_win"], joined["model_home_prob"])
    return metrics

