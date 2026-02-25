from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest betting picks.")
    parser.add_argument("--picks", required=True, help="CSV with picks and outcomes")
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def max_drawdown(returns: pd.Series) -> float:
    equity = returns.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    return float(drawdown.min())


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.picks)
    required = {"model_prob", "outcome", "odds", "bet_units"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    profit = np.where(df["outcome"] == 1, df["bet_units"] * (df["odds"] - 1.0), -df["bet_units"])
    staked = df["bet_units"].sum()

    roi = float(profit.sum() / staked) if staked else 0.0
    win_rate = float(df["outcome"].mean())
    brier = float(np.mean((df["model_prob"] - df["outcome"]) ** 2))
    drawdown = max_drawdown(pd.Series(profit))

    bins = pd.cut(df["model_prob"], bins=args.bins)
    calib = df.groupby(bins, observed=True).agg(mean_pred=("model_prob", "mean"), hit_rate=("outcome", "mean"), n=("outcome", "size"))

    print(f"Bets: {len(df)}")
    print(f"Staked Units: {staked:.3f}")
    print(f"ROI: {roi:.4f}")
    print(f"Yield (profit/bets): {profit.sum()/len(df):.4f}")
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Max Drawdown (units): {drawdown:.4f}")
    print("\nCalibration bins:")
    print(calib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
