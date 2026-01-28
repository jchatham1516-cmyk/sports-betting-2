#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _profit_per_dollar(american_odds: float) -> float:
    try:
        odds = float(american_odds)
    except Exception:
        return float("nan")
    if np.isnan(odds) or odds == 0:
        return float("nan")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def _roi_for_bets(df: pd.DataFrame, units_col: str) -> Tuple[float, int]:
    if df.empty:
        return float("nan"), 0
    profit = 0.0
    count = 0
    for _, row in df.iterrows():
        units = float(row.get(units_col, 0.0) or 0.0)
        if units <= 0:
            continue
        price = row.get("price")
        actual = row.get("actual_result")
        profit_per = _profit_per_dollar(price)
        if not np.isfinite(profit_per) or not np.isfinite(actual):
            continue
        profit += units * (profit_per if actual == 1.0 else -1.0)
        count += 1
    if count == 0:
        return float("nan"), 0
    return profit / float(count), count


def _candidate_units(
    row: pd.Series,
    *,
    min_edge: float,
    longshot_odds: float,
    longshot_cap_units: float,
    calibration_risk_mult: float,
) -> float:
    model_prob = pd.to_numeric(row.get("model_prob"), errors="coerce")
    market_prob = pd.to_numeric(row.get("market_prob"), errors="coerce")
    if not np.isfinite(model_prob) or not np.isfinite(market_prob):
        return 0.0
    edge = float(model_prob) - float(market_prob)
    if edge < float(min_edge):
        return 0.0

    units = 1.0
    price = pd.to_numeric(row.get("price"), errors="coerce")
    if np.isfinite(price) and float(price) >= float(longshot_odds):
        units = min(units, float(longshot_cap_units))

    calibrated = np.isfinite(pd.to_numeric(row.get("model_prob_raw"), errors="coerce"))
    if not calibrated:
        units *= float(calibration_risk_mult)

    return float(units)


def _split_by_date(df: pd.DataFrame, cutoff_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in df.columns:
        return df, df
    work = df.copy()
    work["date_parsed"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date_parsed"])
    if work.empty:
        return df, df
    dates = sorted(work["date_parsed"].unique())
    cutoff_index = max(1, int(len(dates) * cutoff_ratio))
    cutoff = dates[cutoff_index - 1]
    train = work[work["date_parsed"] <= cutoff].copy()
    test = work[work["date_parsed"] > cutoff].copy()
    return train, test if not test.empty else train


def _avg_clv(clv_log: pd.DataFrame, *, sport: str) -> float:
    if clv_log.empty:
        return float("nan")
    df = clv_log.copy()
    if "sport" in df.columns:
        df = df[df["sport"].astype(str).str.lower() == sport.lower()]
    if df.empty:
        return float("nan")
    open_df = df[df["stage"] == "open"]
    close_df = df[df["stage"] == "close"]
    if open_df.empty or close_df.empty:
        return float("nan")
    merged = open_df.merge(
        close_df[["bet_id", "price"]].rename(columns={"price": "close_price"}),
        on="bet_id",
        how="left",
    )
    merged["open_price"] = merged["price"]
    merged["open_prob"] = merged["open_price"].apply(lambda x: 100.0 / (x + 100.0) if x > 0 else (-x) / ((-x) + 100.0))
    merged["close_prob"] = merged["close_price"].apply(lambda x: 100.0 / (x + 100.0) if x > 0 else (-x) / ((-x) + 100.0))
    merged["clv_prob"] = merged["close_prob"] - merged["open_prob"]
    return float(pd.to_numeric(merged.get("clv_prob", np.nan), errors="coerce").mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--eval-history", default="results/eval_history.csv")
    parser.add_argument("--clv-log", default="results/clv_log.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--clv-weight", type=float, default=0.5)
    args = parser.parse_args()

    if not args.out:
        args.out = f"results/thresholds_{args.sport.lower()}.json"

    try:
        hist = pd.read_csv(args.eval_history)
    except Exception:
        print(f"[tune] missing eval history: {args.eval_history}")
        return 1
    if hist.empty:
        print("[tune] no eval history rows to tune.")
        return 1

    if "sport" in hist.columns:
        hist = hist[hist["sport"].astype(str).str.lower() == args.sport.lower()].copy()
    if hist.empty:
        print("[tune] no rows for sport.")
        return 1

    train, test = _split_by_date(hist)

    clv_log = pd.DataFrame()
    try:
        clv_log = pd.read_csv(args.clv_log)
    except Exception:
        clv_log = pd.DataFrame()
    clv_avg = _avg_clv(clv_log, sport=args.sport)

    candidates = []
    for min_edge in (0.02, 0.03, 0.04, 0.05):
        for longshot_cap in (0.25, 0.5):
            for cal_mult in (0.5, 0.7, 0.9):
                work = test.copy()
                work["units"] = work.apply(
                    lambda r: _candidate_units(
                        r,
                        min_edge=min_edge,
                        longshot_odds=400.0,
                        longshot_cap_units=longshot_cap,
                        calibration_risk_mult=cal_mult,
                    ),
                    axis=1,
                )
                roi, bet_count = _roi_for_bets(work, "units")
                if not np.isfinite(roi):
                    continue
                score = roi + float(args.clv_weight) * (clv_avg if np.isfinite(clv_avg) else 0.0)
                candidates.append(
                    {
                        "min_edge_cal": min_edge,
                        "longshot_cap_units": longshot_cap,
                        "calibration_risk_multiplier": cal_mult,
                        "roi": roi,
                        "bets": bet_count,
                        "score": score,
                    }
                )

    if not candidates:
        print("[tune] no valid candidate thresholds found.")
        return 1

    best = max(candidates, key=lambda c: c["score"])
    payload: Dict[str, object] = {
        "sport": args.sport.lower(),
        "min_edge_cal": best["min_edge_cal"],
        "longshot_cap_units": best["longshot_cap_units"],
        "calibration_risk_multiplier": best["calibration_risk_multiplier"],
        "tuned_at": datetime.utcnow().isoformat() + "Z",
        "objective": {
            "roi": best["roi"],
            "clv_weight": args.clv_weight,
            "clv_avg": clv_avg,
            "score": best["score"],
        },
        "notes": "Walk-forward tune on eval_history with CLV-weighted ROI.",
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[tune] wrote {args.out}")
    print(f"[tune] best min_edge={best['min_edge_cal']:.3f} "
          f"longshot_cap={best['longshot_cap_units']:.2f} "
          f"cal_risk_mult={best['calibration_risk_multiplier']:.2f} "
          f"roi={best['roi']:.4f} score={best['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
