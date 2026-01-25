#!/usr/bin/env python3
"""Build performance dashboards from bet_log."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from sports.common.util import american_to_decimal


RESULT_MAP = {
    "WIN": 1.0,
    "LOSS": 0.0,
    "PUSH": 0.5,
}


@dataclass(frozen=True)
class SummaryMetrics:
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    avg_odds: float
    avg_edge: float
    avg_ev: float
    total_profit: float
    roi_per_bet: float
    roi_on_stake: float
    total_stake: float


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_date(value: object) -> datetime | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _result_to_outcome(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().upper()
    if key in RESULT_MAP:
        return RESULT_MAP[key]
    return None


def _profit_from_result(row: pd.Series) -> float:
    profit = row.get("profit_dollars")
    profit_val = float(profit) if profit is not None and np.isfinite(profit) else np.nan
    if np.isfinite(profit_val):
        return float(profit_val)

    stake = row.get("stake_dollars")
    stake_val = float(stake) if stake is not None and np.isfinite(stake) else np.nan
    price = row.get("price_at_bet")
    dec = american_to_decimal(price) if price is not None else float("nan")
    outcome = _result_to_outcome(row.get("result"))
    if not np.isfinite(stake_val) or not np.isfinite(dec) or outcome is None:
        return float("nan")
    if outcome == 1.0:
        return float(stake_val * (dec - 1.0))
    if outcome == 0.0:
        return float(-stake_val)
    return 0.0


def _summary_metrics(df: pd.DataFrame) -> SummaryMetrics:
    results = df.get("result", pd.Series(dtype=str)).astype(str).str.upper()
    wins = int((results == "WIN").sum())
    losses = int((results == "LOSS").sum())
    pushes = int((results == "PUSH").sum())
    total_bets = int(len(df))
    win_rate = wins / (wins + losses) if wins + losses > 0 else float("nan")

    avg_odds = float(_to_numeric(df.get("price_at_bet", pd.Series(dtype=float))).mean())

    edge_series = df.get("edge_prob_final")
    if edge_series is None or edge_series.empty:
        edge_series = df.get("edge")
    avg_edge = float(_to_numeric(edge_series).mean()) if edge_series is not None else float("nan")

    ev_series = df.get("edge")
    avg_ev = float(_to_numeric(ev_series).mean()) if ev_series is not None else float("nan")

    profits = _to_numeric(df.get("profit_dollars", pd.Series(dtype=float)))
    if profits.isna().all():
        profits = df.apply(_profit_from_result, axis=1)
        profits = _to_numeric(pd.Series(profits))
    total_profit = float(profits.sum())

    stake_series = _to_numeric(df.get("stake_dollars", pd.Series(dtype=float)))
    total_stake = float(stake_series.sum())
    roi_on_stake = total_profit / total_stake if total_stake > 0 else float("nan")
    roi_per_bet = total_profit / total_bets if total_bets > 0 else float("nan")

    return SummaryMetrics(
        total_bets=total_bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        avg_odds=avg_odds,
        avg_edge=avg_edge,
        avg_ev=avg_ev,
        total_profit=total_profit,
        roi_per_bet=roi_per_bet,
        roi_on_stake=roi_on_stake,
        total_stake=total_stake,
    )


def _group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = []
    for key, group in df.groupby(group_col):
        summary = _summary_metrics(group)
        grouped.append(
            {
                group_col: key,
                "bets": summary.total_bets,
                "wins": summary.wins,
                "losses": summary.losses,
                "pushes": summary.pushes,
                "win_rate": summary.win_rate,
                "avg_odds": summary.avg_odds,
                "avg_edge": summary.avg_edge,
                "avg_ev": summary.avg_ev,
                "profit_dollars": summary.total_profit,
                "total_stake": summary.total_stake,
                "roi_per_bet": summary.roi_per_bet,
                "roi_on_stake": summary.roi_on_stake,
            }
        )
    return pd.DataFrame(grouped).sort_values("bets", ascending=False)


def _equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["date_parsed"] = work["date"].apply(_parse_date)
    work = work[work["date_parsed"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["date", "profit_dollars", "cumulative_profit"])
    work["profit_dollars"] = _to_numeric(work.get("profit_dollars", pd.Series(dtype=float)))
    if work["profit_dollars"].isna().all():
        work["profit_dollars"] = work.apply(_profit_from_result, axis=1)
    daily = (
        work.groupby(work["date_parsed"].dt.date)["profit_dollars"]
        .sum()
        .reset_index()
        .rename(columns={"date_parsed": "date"})
    )
    daily["cumulative_profit"] = daily["profit_dollars"].cumsum()
    daily["date"] = daily["date"].astype(str)
    return daily


def _drawdowns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["date", "cumulative_profit", "peak", "drawdown", "max_drawdown"])
    work = equity.copy()
    work["peak"] = work["cumulative_profit"].cummax()
    work["drawdown"] = work["cumulative_profit"] - work["peak"]
    work["max_drawdown"] = work["drawdown"].cummin()
    return work


def _rolling_windows(df: pd.DataFrame, windows: Iterable[int]) -> Dict[int, SummaryMetrics]:
    results: Dict[int, SummaryMetrics] = {}
    if df.empty:
        return results
    dates = df["date"].apply(_parse_date)
    if dates.isna().all():
        return results
    latest = dates.dropna().max()
    df = df.copy()
    df["date_parsed"] = dates
    for window in windows:
        cutoff = latest - timedelta(days=int(window))
        subset = df[df["date_parsed"] >= cutoff]
        if subset.empty:
            continue
        results[window] = _summary_metrics(subset)
    return results


def _calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    probs = _to_numeric(df.get("p_model_final", pd.Series(dtype=float)))
    outcomes = df.get("result", pd.Series(dtype=str)).apply(_result_to_outcome)
    mask = probs.notna() & outcomes.notna()
    if not mask.any():
        return pd.DataFrame(columns=["bin", "count", "avg_pred", "win_rate"])
    probs = probs[mask]
    outcomes = outcomes[mask]
    bins = pd.cut(probs, bins=10, labels=False, include_lowest=True)
    table = (
        pd.DataFrame({"bin": bins, "pred": probs, "outcome": outcomes})
        .groupby("bin")
        .agg(count=("pred", "size"), avg_pred=("pred", "mean"), win_rate=("outcome", "mean"))
        .reset_index()
    )
    table["bin"] = table["bin"].apply(lambda x: f"{x + 1}" if pd.notna(x) else "")
    return table


def build_performance_dashboard(bet_log_path: str, out_dir: str = "results/tracking") -> Tuple[Path, Path]:
    bet_log_path = str(bet_log_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bet_log = pd.read_csv(bet_log_path)
    if bet_log.empty:
        raise ValueError("bet_log.csv is empty")

    bet_log = bet_log.copy()
    if "date" not in bet_log.columns:
        bet_log["date"] = ""

    summary = _summary_metrics(bet_log)
    summary_rows = [
        ("total_bets", summary.total_bets),
        ("wins", summary.wins),
        ("losses", summary.losses),
        ("pushes", summary.pushes),
        ("win_rate", summary.win_rate),
        ("avg_odds", summary.avg_odds),
        ("avg_edge", summary.avg_edge),
        ("avg_ev", summary.avg_ev),
        ("total_profit", summary.total_profit),
        ("roi_per_bet", summary.roi_per_bet),
        ("roi_on_stake", summary.roi_on_stake),
        ("total_stake", summary.total_stake),
    ]

    rolling_stats = _rolling_windows(bet_log, windows=[7, 14, 30])
    for window, stats in rolling_stats.items():
        summary_rows.extend(
            [
                (f"last_{window}_bets", stats.total_bets),
                (f"last_{window}_win_rate", stats.win_rate),
                (f"last_{window}_roi_on_stake", stats.roi_on_stake),
            ]
        )

    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    summary_path = out_path / f"perf_summary_{date_tag}.csv"
    summary_df.to_csv(summary_path, index=False)

    by_sport = _group_metrics(bet_log, "sport") if "sport" in bet_log.columns else pd.DataFrame()
    by_market = _group_metrics(bet_log, "market_type") if "market_type" in bet_log.columns else pd.DataFrame()

    by_sport_path = out_path / "perf_by_sport.csv"
    by_market_path = out_path / "perf_by_market.csv"
    by_sport.to_csv(by_sport_path, index=False)
    by_market.to_csv(by_market_path, index=False)

    equity_curve = _equity_curve(bet_log)
    equity_path = out_path / "equity_curve.csv"
    equity_curve.to_csv(equity_path, index=False)

    drawdowns = _drawdowns(equity_curve)
    drawdown_path = out_path / "drawdowns.csv"
    drawdowns.to_csv(drawdown_path, index=False)

    calib_table = _calibration_table(bet_log)

    report_path = out_path / "perf_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Performance Report\n\n")
        handle.write(f"Generated: {date_tag}\n\n")
        handle.write("## Summary\n\n")
        for metric, value in summary_rows:
            handle.write(f"- **{metric}**: {value}\n")
        if rolling_stats:
            handle.write("\n## Rolling Windows\n\n")
            for window, stats in rolling_stats.items():
                handle.write(
                    f"- Last {window} days: bets={stats.total_bets}, win_rate={stats.win_rate:.3f}, "
                    f"roi_on_stake={stats.roi_on_stake:.3f}\n"
                )
        if not by_sport.empty:
            handle.write("\n## By Sport\n\n")
            handle.write(by_sport.to_string(index=False))
            handle.write("\n")
        if not by_market.empty:
            handle.write("\n## By Market\n\n")
            handle.write(by_market.to_string(index=False))
            handle.write("\n")
        handle.write("\n## Calibration (p_model_final bins)\n\n")
        if calib_table.empty:
            handle.write("No calibration data available.\n")
        else:
            handle.write(calib_table.to_string(index=False))
            handle.write("\n")
        handle.write("\n## Drawdowns\n\n")
        if drawdowns.empty:
            handle.write("No drawdown data available.\n")
        else:
            max_drawdown = float(drawdowns["drawdown"].min()) if not drawdowns.empty else 0.0
            handle.write(f"Max drawdown: {max_drawdown:.2f}\n")

    return summary_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build performance dashboards from bet_log.csv")
    parser.add_argument("--bet-log", required=True, help="Path to results/tracking/bet_log.csv")
    parser.add_argument(
        "--out-dir",
        default="results/tracking",
        help="Output directory for perf dashboard artifacts.",
    )
    args = parser.parse_args()

    build_performance_dashboard(args.bet_log, out_dir=args.out_dir)
    print(f"[perf_dashboard] Wrote outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
