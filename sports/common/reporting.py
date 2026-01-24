from __future__ import annotations

import json
import os
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.util import american_to_decimal, normalize_result_label
from sports.common.bet_config import get_sport_bet_config


ODDS_BUCKETS = [
    (100, 150, "+100..+150"),
    (151, 300, "+151..+300"),
    (301, 500, "+301..+500"),
    (501, 10_000, "+500+"),
]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _profit_from_row(price: float, result: str, stake: float) -> float:
    result = normalize_result_label(result)
    if not np.isfinite(stake) or stake <= 0:
        return float("nan")
    if result == "WIN":
        dec = american_to_decimal(price)
        return stake * (dec - 1.0)
    if result == "LOSS":
        return -stake
    if result == "PUSH":
        return 0.0
    return float("nan")


def _bucket_for_odds(price: float) -> str:
    if not np.isfinite(price) or price < 100:
        return "OTHER"
    for low, high, label in ODDS_BUCKETS:
        if low <= price <= high:
            return label
    if price > 500:
        return "+500+"
    return "OTHER"


def _grade_moneyline_from_history(
    bets: pd.DataFrame, history_df: pd.DataFrame
) -> pd.DataFrame:
    if bets.empty or history_df.empty:
        return bets

    hist = history_df.copy()
    hist["home_canon"] = hist["home"].apply(canon_team)
    hist["away_canon"] = hist["away"].apply(canon_team)
    hist = hist[["date", "home_canon", "away_canon", "home_win"]]

    bets = bets.copy()
    bets["home_canon"] = bets["home"].apply(canon_team)
    bets["away_canon"] = bets["away"].apply(canon_team)

    merged = bets.merge(hist, on=["date", "home_canon", "away_canon"], how="left")
    missing = merged["result"].isna() | (merged["result"].astype(str).str.strip() == "")
    if not missing.any():
        return bets

    def _infer_result(row: pd.Series) -> Optional[str]:
        if pd.isna(row.get("home_win")):
            return None
        side = str(row.get("side", "")).upper()
        if side == "HOME":
            return "WIN" if int(row.get("home_win")) == 1 else "LOSS"
        if side == "AWAY":
            return "WIN" if int(row.get("home_win")) == 0 else "LOSS"
        return None

    merged.loc[missing, "result"] = merged[missing].apply(_infer_result, axis=1)
    merged = merged.drop(columns=["home_canon", "away_canon", "home_win"])
    return merged


def generate_backtest_report(
    sport: str,
    history_csv_path: str,
    *,
    bet_log_path: str = "results/tracking/bet_log.csv",
) -> Optional[Dict[str, object]]:
    if not os.path.exists(bet_log_path):
        print(f"[report] No bet log at {bet_log_path}; skipping report.")
        return None

    bets = pd.read_csv(bet_log_path)
    if bets.empty:
        print("[report] Bet log empty; skipping report.")
        return None

    bets = bets[bets["sport"].astype(str).str.lower() == str(sport).lower()].copy()
    if bets.empty:
        print("[report] No bets for sport; skipping report.")
        return None

    history_df = pd.DataFrame()
    if history_csv_path and os.path.exists(history_csv_path):
        history_df = pd.read_csv(history_csv_path)

    if not history_df.empty:
        bets = _grade_moneyline_from_history(bets, history_df)

    bets["result"] = bets["result"].apply(normalize_result_label)
    bets = bets[bets["result"].isin(["WIN", "LOSS", "PUSH"])]
    if bets.empty:
        print("[report] No graded bets; skipping report.")
        return None

    bets["price"] = bets["price_at_bet"].apply(_safe_float)
    bets["units"] = bets["units"].apply(_safe_float)
    bets["unit_dollars"] = bets["unit_dollars"].apply(_safe_float)
    bets["stake"] = bets["units"] * bets["unit_dollars"]
    bets["profit"] = bets.apply(
        lambda r: _profit_from_row(r["price"], r["result"], r["stake"]),
        axis=1,
    )

    def _summary(df: pd.DataFrame) -> Dict[str, float]:
        stake = df["stake"].sum()
        profit = df["profit"].sum()
        wins = int((df["result"] == "WIN").sum())
        losses = int((df["result"] == "LOSS").sum())
        pushes = int((df["result"] == "PUSH").sum())
        bets_count = wins + losses + pushes
        win_pct = wins / (wins + losses) if wins + losses > 0 else 0.0
        avg_odds = df["price"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        roi = profit / stake if stake > 0 else 0.0
        return {
            "bets": bets_count,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_pct": win_pct,
            "roi": roi,
            "avg_odds": avg_odds,
        }

    bets["odds_bucket"] = bets["price"].apply(_bucket_for_odds)

    report = {
        "sport": str(sport).lower(),
        "overall": _summary(bets),
        "by_odds_bucket": {
            bucket: _summary(group)
            for bucket, group in bets.groupby("odds_bucket")
        },
        "by_confidence": {
            str(conf): _summary(group)
            for conf, group in bets.groupby(bets["confidence"].fillna("UNKNOWN"))
        },
        "by_value_tier": {
            str(tier): _summary(group)
            for tier, group in bets.groupby(bets["value_tier"].fillna("UNKNOWN"))
        },
    }

    os.makedirs(os.path.join("results", "reports"), exist_ok=True)
    out_path = os.path.join("results", "reports", f"{sport}_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def self_check_recent_bets(
    *,
    bet_log_path: str = "results/tracking/bet_log.csv",
    last_n: int = 200,
) -> Optional[Dict[str, object]]:
    if not os.path.exists(bet_log_path):
        print(f"[self-check] No bet log at {bet_log_path}; skipping.")
        return None

    bets = pd.read_csv(bet_log_path)
    if bets.empty:
        print("[self-check] Bet log empty; skipping.")
        return None

    bets = bets.tail(int(last_n)).copy()
    bets["result"] = bets["result"].apply(normalize_result_label)
    bets = bets[bets["result"].isin(["WIN", "LOSS", "PUSH"])]
    if bets.empty:
        print("[self-check] No graded bets in recent window; skipping.")
        return None

    bets["units"] = bets["units"].apply(_safe_float)
    bets["unit_dollars"] = bets["unit_dollars"].apply(_safe_float)
    bets["stake"] = bets["units"] * bets["unit_dollars"]
    bets["price"] = bets["price_at_bet"].apply(_safe_float)

    def _result_num(res: str) -> float:
        if res == "WIN":
            return 1.0
        if res == "LOSS":
            return 0.0
        return 0.5

    bets["result_num"] = bets["result"].apply(_result_num)
    bets["model_prob"] = bets["model_prob"].apply(_safe_float)
    bets["market_prob"] = bets["market_prob"].apply(_safe_float)
    bets["edge_prob_final"] = bets["edge_prob_final"].apply(_safe_float)
    bets["abs_edge_prob"] = bets["abs_edge_prob"].apply(_safe_float)

    bets["implied_edge"] = bets["model_prob"] - bets["market_prob"]

    overall_units = float(bets["units"].sum())
    wins = int((bets["result"] == "WIN").sum())
    losses = int((bets["result"] == "LOSS").sum())
    pushes = int((bets["result"] == "PUSH").sum())
    win_rate = wins / (wins + losses) if wins + losses > 0 else 0.0
    avg_implied_edge = float(bets["implied_edge"].replace([np.inf, -np.inf], np.nan).mean())
    avg_model_edge = float(bets["edge_prob_final"].replace([np.inf, -np.inf], np.nan).mean())

    brier = float(
        np.nanmean((bets["model_prob"] - bets["result_num"]) ** 2)
        if bets["model_prob"].notna().any()
        else float("nan")
    )

    by_market = {}
    for market, group in bets.groupby(bets["market_type"].fillna("UNKNOWN")):
        wins_m = int((group["result"] == "WIN").sum())
        losses_m = int((group["result"] == "LOSS").sum())
        pushes_m = int((group["result"] == "PUSH").sum())
        win_rate_m = wins_m / (wins_m + losses_m) if wins_m + losses_m > 0 else 0.0
        by_market[str(market)] = {
            "bets": int(len(group)),
            "wins": wins_m,
            "losses": losses_m,
            "pushes": pushes_m,
            "win_rate": float(win_rate_m),
            "avg_implied_edge": float(group["implied_edge"].replace([np.inf, -np.inf], np.nan).mean()),
            "avg_model_edge": float(group["edge_prob_final"].replace([np.inf, -np.inf], np.nan).mean()),
        }

    def _thin_edge_flag(row: pd.Series) -> bool:
        sport = str(row.get("sport", "")).lower()
        config = get_sport_bet_config(sport)
        edge = row.get("abs_edge_prob")
        return bool(np.isfinite(edge) and float(edge) < float(config.min_edge_cal))

    thin_edge_rate = float(bets.apply(_thin_edge_flag, axis=1).mean())

    report = {
        "sample_size": int(len(bets)),
        "overall_units": overall_units,
        "win_rate": win_rate,
        "avg_implied_edge": avg_implied_edge,
        "avg_model_edge": avg_model_edge,
        "brier": brier,
        "by_market": by_market,
        "thin_edge_rate": thin_edge_rate,
    }

    print(
        "[self-check] "
        f"n={report['sample_size']} units={overall_units:.2f} win_rate={win_rate:.3f} "
        f"avg_implied_edge={avg_implied_edge:.4f} avg_model_edge={avg_model_edge:.4f} brier={brier:.4f}"
    )
    for market, stats in by_market.items():
        print(
            "[self-check] "
            f"{market}: bets={stats['bets']} win_rate={stats['win_rate']:.3f} "
            f"avg_implied_edge={stats['avg_implied_edge']:.4f} avg_model_edge={stats['avg_model_edge']:.4f}"
        )
    if thin_edge_rate > 0.4:
        print(f"[self-check] WARNING: {thin_edge_rate:.0%} of recent bets are thin-edge (<min_edge_cal).")

    return report


def daily_bet_report(
    *,
    bet_log_path: str = "results/tracking/bet_log.csv",
    thin_edge_warn_pct: float = 0.4,
    bucket_size: float = 0.05,
) -> Optional[Dict[str, object]]:
    if not os.path.exists(bet_log_path):
        print(f"[daily-report] No bet log at {bet_log_path}; skipping.")
        return None

    bets = pd.read_csv(bet_log_path)
    if bets.empty:
        print("[daily-report] Bet log empty; skipping.")
        return None

    bets = bets.copy()
    bets["result"] = bets["result"].apply(normalize_result_label)
    bets = bets[bets["result"].isin(["WIN", "LOSS", "PUSH"])]
    if bets.empty:
        print("[daily-report] No graded bets; skipping.")
        return None

    bets["units"] = bets["units"].apply(_safe_float)
    bets["unit_dollars"] = bets["unit_dollars"].apply(_safe_float)
    bets["stake"] = bets["units"] * bets["unit_dollars"]
    bets["price"] = bets["price_at_bet"].apply(_safe_float)
    bets["profit"] = bets.apply(
        lambda r: _profit_from_row(r["price"], r["result"], r["stake"]),
        axis=1,
    )

    def _summary(df: pd.DataFrame) -> Dict[str, float]:
        stake = df["stake"].sum()
        profit = df["profit"].sum()
        wins = int((df["result"] == "WIN").sum())
        losses = int((df["result"] == "LOSS").sum())
        pushes = int((df["result"] == "PUSH").sum())
        bets_count = wins + losses + pushes
        win_pct = wins / (wins + losses) if wins + losses > 0 else 0.0
        roi = profit / stake if stake > 0 else 0.0
        return {
            "bets": bets_count,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_pct,
            "roi": roi,
            "units": float(df["units"].sum()),
        }

    by_sport = {
        str(sport): _summary(group)
        for sport, group in bets.groupby(bets["sport"].fillna("UNKNOWN"))
    }
    by_market = {
        str(market): _summary(group)
        for market, group in bets.groupby(bets["market_type"].fillna("UNKNOWN"))
    }

    prob_col = "model_prob" if "model_prob" in bets.columns else "p_model_final"
    bets["prob_used"] = bets.get(prob_col, np.nan).apply(_safe_float)
    bets = bets[bets["prob_used"].notna()]
    bets["prob_used"] = bets["prob_used"].clip(0.0, 1.0)

    def _bucket_label(p: float) -> str:
        lo = math.floor(float(p) / bucket_size) * bucket_size
        hi = lo + bucket_size
        return f"{lo:.2f}-{hi:.2f}"

    bets["prob_bucket"] = bets["prob_used"].apply(_bucket_label)
    calib = {}
    for bucket, group in bets.groupby("prob_bucket"):
        wins = int((group["result"] == "WIN").sum())
        losses = int((group["result"] == "LOSS").sum())
        win_rate = wins / (wins + losses) if wins + losses > 0 else 0.0
        calib[str(bucket)] = {
            "bets": int(len(group)),
            "avg_prob": float(group["prob_used"].mean()),
            "win_rate": float(win_rate),
        }

    def _thin_edge_flag(row: pd.Series) -> bool:
        sport = str(row.get("sport", "")).lower()
        config = get_sport_bet_config(sport)
        edge = row.get("abs_edge_prob")
        if not np.isfinite(edge):
            return False
        return float(edge) < float(config.min_edge_cal) + 0.01

    thin_edge_rate = float(bets.apply(_thin_edge_flag, axis=1).mean())

    report = {
        "by_sport": by_sport,
        "by_market": by_market,
        "calibration": calib,
        "thin_edge_rate": thin_edge_rate,
    }

    print("[daily-report] Summary by sport:")
    for sport, stats in by_sport.items():
        print(
            f"  {sport}: units={stats['units']:.2f} win_rate={stats['win_rate']:.3f} ROI={stats['roi']:.3f}"
        )
    print("[daily-report] Summary by market:")
    for market, stats in by_market.items():
        print(
            f"  {market}: units={stats['units']:.2f} win_rate={stats['win_rate']:.3f} ROI={stats['roi']:.3f}"
        )
    print("[daily-report] Calibration buckets:")
    for bucket, stats in calib.items():
        print(
            f"  {bucket}: bets={stats['bets']} avg_prob={stats['avg_prob']:.3f} win_rate={stats['win_rate']:.3f}"
        )
    if thin_edge_rate > float(thin_edge_warn_pct):
        print(
            f"[daily-report] WARNING: {thin_edge_rate:.0%} of bets are thin edges "
            f"(<min_edge+0.01)."
        )

    return report
