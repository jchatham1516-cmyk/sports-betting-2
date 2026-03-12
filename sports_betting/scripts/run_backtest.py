from __future__ import annotations

import argparse
import json

from sports_betting.backtesting.engine import run_backtest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    summary = run_backtest(args.predictions)
    print(json.dumps(summary, indent=2))
