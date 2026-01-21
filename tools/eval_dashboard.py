#!/usr/bin/env python3
"""Evaluate recent NBA/NHL prediction CSVs and summarize performance."""
from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

RESULT_COLUMNS = ("won", "result", "outcome", "home_won")
MODEL_PROB_COLUMNS = (
    "p_model_used",
    "p_model_final",
    "model_home_prob_final",
    "model_home_prob",
)
MARKET_PROB_COLUMNS = ("p_market", "market_home_prob")


@dataclass(frozen=True)
class SportConfig:
    name: str
    pattern: str
    output_csv: str


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", filename)
    if not match:
        return None
    raw = match.group(1)
    for fmt in ("%Y-%m-%d", "%Y_%m_%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def select_recent_files(files: Iterable[str], count: int = 3) -> list[str]:
    dated_files = []
    for path in files:
        parsed = parse_date_from_filename(os.path.basename(path))
        if parsed:
            dated_files.append((parsed, path))
    dated_files.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in dated_files[:count]]


def normalize_outcome(series: pd.Series) -> pd.Series:
    def to_outcome(value) -> Optional[float]:
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            if value in (0, 1):
                return float(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "win", "won", "w"}:
            return 1.0
        if text in {"0", "false", "f", "no", "n", "loss", "lost", "l"}:
            return 0.0
        return None

    return series.map(to_outcome)


def pick_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def compute_metrics(df: pd.DataFrame) -> dict:
    metrics: dict[str, Optional[float]] = {}
    metrics["n_games"] = len(df)

    if "play_pass" in df.columns:
        metrics["n_bets"] = df["play_pass"].astype(str).str.upper().eq("PLAY").sum()
    elif "stake_dollars" in df.columns:
        metrics["n_bets"] = (pd.to_numeric(df["stake_dollars"], errors="coerce") > 0).sum()
    else:
        metrics["n_bets"] = None

    result_col = pick_first_column(df, RESULT_COLUMNS)
    outcome_series = None
    if result_col:
        outcome_series = normalize_outcome(df[result_col])
        metrics["win_rate"] = outcome_series.dropna().mean()
    else:
        metrics["win_rate"] = None

    model_col = pick_first_column(df, MODEL_PROB_COLUMNS)
    market_col = pick_first_column(df, MARKET_PROB_COLUMNS)

    if model_col:
        metrics["avg_p_model_used"] = pd.to_numeric(df[model_col], errors="coerce").mean()
    else:
        metrics["avg_p_model_used"] = None

    if market_col:
        metrics["avg_p_market_used"] = pd.to_numeric(df[market_col], errors="coerce").mean()
    else:
        metrics["avg_p_market_used"] = None

    if model_col and market_col:
        model_vals = pd.to_numeric(df[model_col], errors="coerce")
        market_vals = pd.to_numeric(df[market_col], errors="coerce")
        metrics["avg_edge"] = (model_vals - market_vals).mean()
    else:
        metrics["avg_edge"] = None

    if "primary_ev" in df.columns:
        metrics["avg_primary_ev"] = pd.to_numeric(df["primary_ev"], errors="coerce").mean()
    else:
        metrics["avg_primary_ev"] = None

    if model_col and outcome_series is not None:
        model_vals = pd.to_numeric(df[model_col], errors="coerce")
        aligned = pd.concat([model_vals, outcome_series], axis=1).dropna()
        if not aligned.empty:
            metrics["brier_score"] = ((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2).mean()
        else:
            metrics["brier_score"] = None
    else:
        metrics["brier_score"] = None

    return metrics


def summarize_metrics(sport: str, metrics: dict) -> str:
    lines = [f"{sport} Evaluation Summary"]
    for key, value in metrics.items():
        if value is None:
            lines.append(f"- {key}: n/a")
        elif isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def load_csvs(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except (pd.errors.EmptyDataError, csv.Error):
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def process_sport(config: SportConfig) -> None:
    matched_files = glob.glob(config.pattern)
    recent_files = select_recent_files(matched_files, count=3)

    df = load_csvs(recent_files)
    metrics = compute_metrics(df) if not df.empty else {
        "n_games": 0,
        "n_bets": None,
        "win_rate": None,
        "avg_p_model_used": None,
        "avg_p_market_used": None,
        "avg_edge": None,
        "avg_primary_ev": None,
        "brier_score": None,
    }

    summary_text = summarize_metrics(config.name, metrics)
    print(summary_text)
    print("-")

    output_df = pd.DataFrame([metrics])
    output_df.to_csv(config.output_csv, index=False)


def main() -> None:
    configs = [
        SportConfig(
            name="NBA",
            pattern="/results/predictions_nba_*.csv",
            output_csv="results/eval_dashboard_nba.csv",
        ),
        SportConfig(
            name="NHL",
            pattern="/results/predictions_nhl_*.csv",
            output_csv="results/eval_dashboard_nhl.csv",
        ),
    ]

    for config in configs:
        process_sport(config)


if __name__ == "__main__":
    main()
