from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sports_betting.sports.nba.dataset import HISTORICAL_PATH, build_nba_historical_dataset
from sports_betting.sports.nba.training import save_trained_nba_models, train_nba_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NBA moneyline/spread/totals models.")
    parser.add_argument(
        "--historical-csv",
        default=str(HISTORICAL_PATH),
        help="Path to historical NBA dataset CSV.",
    )
    parser.add_argument(
        "--model-dir",
        default="sports_betting/data/models",
        help="Directory to save local model artifacts.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained models into --model-dir.",
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild dataset from available local snapshots before training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    historical_path = Path(args.historical_csv)

    if args.rebuild_dataset or not historical_path.exists():
        df = build_nba_historical_dataset(historical_path)
    else:
        df = pd.read_csv(historical_path)

    labeled_rows = df[["home_win", "home_cover", "over_hit"]].dropna().shape[0] if {"home_win", "home_cover", "over_hit"}.issubset(df.columns) else 0
    if labeled_rows < 200:
        print(f"Not enough labeled historical rows to train models (found {labeled_rows}, need >= 200).")
        return 0

    trained = train_nba_models(df)

    print("NBA model training complete")
    for market, metrics in trained.metrics.items():
        metric_line = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[{market}] {metric_line}")

    if args.save_models:
        save_trained_nba_models(trained, args.model_dir)
        print(f"Saved models to: {args.model_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
