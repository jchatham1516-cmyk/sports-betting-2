from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from sports_betting.config_loader import load_yaml
from sports_betting.logging_utils import configure_logging
from sports_betting.sports.common.io import load_games
from sports_betting.sports.common.reporting import render_console_card, write_outputs
from sports_betting.sports.common.selection import apply_selection, rank_predictions
from sports_betting.sports.common.staking import StakingConfig
from sports_betting.sports.nba.model import NbaModel
from sports_betting.sports.nfl.model import NflModel
from sports_betting.sports.nhl.model import NhlModel


MODEL_MAP = {"nba": NbaModel, "nfl": NflModel, "nhl": NhlModel}


def run_daily(target_date: str, config_path: str, top_n: int) -> int:
    cfg = load_yaml(config_path)
    configure_logging(cfg.get("logging", {}).get("level", "INFO"))

    selected_sports = cfg.get("sports", ["nba", "nfl", "nhl"])
    staking_cfg = StakingConfig(**cfg.get("bankroll", {}))

    all_predictions = []
    for sport in selected_sports:
        model = MODEL_MAP[sport]()
        input_file = Path(f"sports_betting/data/processed/{sport}_{target_date}.json")
        games = load_games(input_file, sport.upper())
        thresholds = cfg.get("thresholds", {}).get(sport, cfg.get("thresholds", {}).get("default", {}))
        for g in games:
            preds = model.predict_game(g)
            for p in preds:
                decimal_odds = next(m.decimal_odds for m in g.markets if m.market == p.market and m.side == p.side)
                all_predictions.append(apply_selection(p, thresholds, decimal_odds, staking_cfg))

    ranked = rank_predictions(all_predictions)
    stamp = target_date.replace("-", "")
    out_dir = Path("sports_betting/data/outputs")
    write_outputs(ranked, out_dir, stamp)
    print(render_console_card(target_date, ranked, top_n=top_n))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-sport betting pipeline")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--config", default="sports_betting/config/default.json")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    return run_daily(args.date, args.config, args.top_n)


if __name__ == "__main__":
    raise SystemExit(main())
