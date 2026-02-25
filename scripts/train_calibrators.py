from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sports.common.calibration import ProbabilityCalibrator, save_calibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probability calibrators.")
    parser.add_argument("--history", required=True, help="CSV with columns sport,market_type,model_prob,outcome")
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--out-dir", default="data/models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history = pd.read_csv(args.history)
    out_dir = Path(args.out_dir)

    required = {"sport", "market_type", "model_prob", "outcome"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for (sport, market), group in history.groupby(["sport", "market_type"]):
        if len(group) < args.min_samples:
            continue
        calibrator = ProbabilityCalibrator(method=args.method).fit(
            group["model_prob"].to_numpy(),
            group["outcome"].to_numpy(),
        )
        save_calibrator(out_dir, sport, market, calibrator)
        print(f"Saved calibrator: sport={sport} market={market} n={len(group)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
