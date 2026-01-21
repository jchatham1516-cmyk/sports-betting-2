#!/usr/bin/env python3
"""Generate a simple evaluation dashboard for recent predictions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PREDICTIONS_DIR = Path("results")
BET_LOG_CANDIDATES = [
    Path("results/tracking/bet_log.csv"),
    Path("results/bet_log.csv"),
]

SPORTS = ["nba", "nhl"]


@dataclass(frozen=True)
class PredictionsFile:
    path: Path
    date: datetime


def _parse_date_from_filename(path: Path, sport: str) -> datetime | None:
    patterns = [
        rf"predictions_{sport}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$",
        rf"predictions_{sport}_(\d{{2}}-\d{{2}}-\d{{4}})\.csv$",
        rf"predictions_{sport}_(\d{{8}})\.csv$",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if not match:
            continue
        token = match.group(1)
        try:
            if "-" in token:
                if token.count("-") == 2 and len(token.split("-")[0]) == 4:
                    return datetime.strptime(token, "%Y-%m-%d")
                return datetime.strptime(token, "%m-%d-%Y")
            return datetime.strptime(token, "%Y%m%d")
        except ValueError:
            continue
    return None


def _parse_date_value(value: object) -> datetime | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalize_team(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _pick_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _choose_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series | None:
    col = _pick_first_column(df, candidates)
    if not col:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _load_predictions_files(sport: str) -> list[PredictionsFile]:
    files = []
    for path in PREDICTIONS_DIR.glob(f"predictions_{sport}_*.csv"):
        parsed = _parse_date_from_filename(path, sport)
        if parsed is None:
            parsed = datetime.fromtimestamp(path.stat().st_mtime)
        files.append(PredictionsFile(path=path, date=parsed))
    return files


def _load_recent_predictions(sport: str, days: int = 3) -> tuple[pd.DataFrame, list[str]]:
    files = _load_predictions_files(sport)
    if not files:
        return pd.DataFrame(), []
    unique_dates = sorted({pf.date.date() for pf in files})
    recent_dates = unique_dates[-days:]
    recent_date_strs = [d.strftime("%Y-%m-%d") for d in recent_dates]

    frames = []
    for pf in files:
        if pf.date.date() not in recent_dates:
            continue
        df = pd.read_csv(pf.path)
        date_col = "date" if "date" in df.columns else None
        if date_col:
            parsed_dates = df[date_col].apply(_parse_date_value)
            df["date"] = parsed_dates.apply(
                lambda d: d.strftime("%Y-%m-%d") if d else pf.date.strftime("%Y-%m-%d")
            )
        else:
            df["date"] = pf.date.strftime("%Y-%m-%d")
        df["sport"] = sport
        frames.append(df)

    if not frames:
        return pd.DataFrame(), []

    combined = pd.concat(frames, ignore_index=True)
    return combined, recent_date_strs


def _load_bet_log() -> pd.DataFrame:
    for path in BET_LOG_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _add_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_first_column(df, ["date", "bet_date"]) or "date"
    if date_col not in df.columns:
        df[date_col] = ""
    parsed = df[date_col].apply(_parse_date_value)
    df["date_norm"] = parsed.apply(lambda d: d.strftime("%Y-%m-%d") if d else "")

    home_col = _pick_first_column(df, ["home", "home_team", "team_home", "home_canon"])
    away_col = _pick_first_column(df, ["away", "away_team", "team_away", "away_canon"])

    if home_col:
        df["home_norm"] = df[home_col].apply(_normalize_team)
    else:
        df["home_norm"] = ""
    if away_col:
        df["away_norm"] = df[away_col].apply(_normalize_team)
    else:
        df["away_norm"] = ""
    return df


def _result_to_outcome(value: object) -> float | None:
    text = str(value).strip().upper()
    if text == "WIN":
        return 1.0
    if text == "LOSS":
        return 0.0
    if text == "PUSH":
        return 0.5
    return None


def _build_dashboard(sport: str) -> None:
    preds_df, recent_dates = _load_recent_predictions(sport)
    if preds_df.empty:
        print(f"[{sport}] No predictions found.")
        return

    bet_log = _load_bet_log()
    if bet_log.empty:
        print(f"[{sport}] bet_log.csv not found or empty; metrics will be limited.")

    preds_df = _add_join_keys(preds_df)
    bet_log = _add_join_keys(bet_log)

    if "sport" in bet_log.columns:
        bet_log = bet_log[bet_log["sport"].str.lower() == sport]

    if recent_dates:
        bet_log = bet_log[bet_log["date_norm"].isin(recent_dates)]

    merged = bet_log.merge(
        preds_df,
        on=["date_norm", "home_norm", "away_norm"],
        how="left",
        suffixes=("_bet", "_pred"),
    )

    games = preds_df.drop_duplicates(subset=["date_norm", "home_norm", "away_norm"])
    n_games = len(games)
    n_bets = len(merged)
    n_wins = int((merged.get("result", pd.Series(dtype=str)).astype(str).str.upper() == "WIN").sum())
    win_rate = (n_wins / n_bets) if n_bets else np.nan

    model_prob = _choose_series(merged, [
        "p_model_final",
        "p_model_used",
        "model_prob",
        "p_model_cal",
        "p_model_raw",
    ])
    market_prob = _choose_series(merged, ["p_market_used", "p_market", "market_prob"])

    edge = None
    if model_prob is not None and market_prob is not None:
        edge = model_prob - market_prob
    else:
        edge = _choose_series(merged, [
            "edge_prob_final",
            "edge_prob_cal",
            "edge_prob_raw",
            "edge",
        ])

    avg_model_prob = float(model_prob.mean()) if model_prob is not None else np.nan
    avg_market_prob = float(market_prob.mean()) if market_prob is not None else np.nan
    avg_edge = float(edge.mean()) if edge is not None else np.nan

    outcomes = merged.get("result", pd.Series(dtype=str)).apply(_result_to_outcome)
    if model_prob is not None:
        valid_mask = outcomes.notna() & model_prob.notna()
        if valid_mask.any():
            brier = float(((model_prob[valid_mask] - outcomes[valid_mask]) ** 2).mean())
        else:
            brier = np.nan
    else:
        brier = np.nan

    ev_series = _choose_series(merged, ["primary_ev", "edge", "edge_prob_final", "edge_prob_cal", "edge_prob_raw"])
    avg_ev = float(ev_series.mean()) if ev_series is not None else np.nan

    edge_for_sort = edge
    if edge_for_sort is None:
        edge_for_sort = _choose_series(merged, ["edge", "edge_prob_final", "edge_prob_cal", "edge_prob_raw", "primary_ev"])

    merged = merged.copy()
    merged["edge_calc"] = edge if edge is not None else edge_for_sort
    merged["abs_edge_calc"] = merged["edge_calc"].abs()

    top_bets = merged.sort_values(by="abs_edge_calc", ascending=False).head(10)

    print(f"\n=== {sport.upper()} ({', '.join(recent_dates)}) ===")
    print(f"n_games: {n_games}")
    print(f"n_bets: {n_bets}")
    print(f"n_wins: {n_wins}")
    print(f"win_rate: {win_rate:.3f}" if np.isfinite(win_rate) else "win_rate: N/A")
    print(f"avg_model_prob: {avg_model_prob:.4f}" if np.isfinite(avg_model_prob) else "avg_model_prob: N/A")
    print(f"avg_market_prob: {avg_market_prob:.4f}" if np.isfinite(avg_market_prob) else "avg_market_prob: N/A")
    print(f"avg_edge: {avg_edge:.4f}" if np.isfinite(avg_edge) else "avg_edge: N/A")
    print(f"brier_score: {brier:.4f}" if np.isfinite(brier) else "brier_score: N/A")
    print(f"avg_ev: {avg_ev:.4f}" if np.isfinite(avg_ev) else "avg_ev: N/A")

    if top_bets.empty:
        print("Top bets: none")
    else:
        display_cols = [
            col
            for col in [
                "date_bet",
                "date_pred",
                "home_bet",
                "away_bet",
                "market_type",
                "side",
                "result",
                "edge_calc",
                "p_model_used",
                "p_market_used",
            ]
            if col in top_bets.columns
        ]
        if not display_cols:
            display_cols = ["date_norm", "home_norm", "away_norm", "result", "edge_calc"]
        print("Top 10 bets by abs(edge):")
        print(top_bets[display_cols].to_string(index=False))

    summary_rows = [
        {"section": "summary", "metric": "n_games", "value": n_games},
        {"section": "summary", "metric": "n_bets", "value": n_bets},
        {"section": "summary", "metric": "n_wins", "value": n_wins},
        {"section": "summary", "metric": "win_rate", "value": win_rate},
        {"section": "summary", "metric": "avg_model_prob", "value": avg_model_prob},
        {"section": "summary", "metric": "avg_market_prob", "value": avg_market_prob},
        {"section": "summary", "metric": "avg_edge", "value": avg_edge},
        {"section": "summary", "metric": "brier_score", "value": brier},
        {"section": "summary", "metric": "avg_ev", "value": avg_ev},
    ]

    top_rows = top_bets.copy()
    top_rows["section"] = "top_bets"

    dashboard_df = pd.concat([pd.DataFrame(summary_rows), top_rows], ignore_index=True, sort=False)
    output_path = PREDICTIONS_DIR / f"eval_dashboard_{sport}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_df.to_csv(output_path, index=False)
    print(f"Saved dashboard to {output_path}")


def main() -> None:
    for sport in SPORTS:
        _build_dashboard(sport)


if __name__ == "__main__":
    main()
