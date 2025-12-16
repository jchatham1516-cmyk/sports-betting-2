# current_of_sports_betting_algorithm.py
#
# Thin runner: picks sport module, loads odds, runs model, applies recos + play/pass + sizing, saves output.
# NBA implemented. NFL/NHL placeholders for now.

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sports.nfl.model import run_daily_nfl
from sports.nhl.model import run_daily_nhl

from recommendations import add_recommendations_to_df, Thresholds

from sports.common.odds_sources import (
    fetch_odds_for_date_from_odds_api,
    fetch_odds_for_date_from_csv,
    SPORT_TO_ODDS_KEY,
)
from sports.common.bankroll import (
    DEFAULT_BANKROLL,
    UNIT_PCT,
    play_pass_rule,
    compute_bet_size,
)

from sports.nba.bdl_client import get_bdl_api_key, season_start_year_for_date, fetch_team_ratings_bdl
from sports.nba.model import run_daily_probs_for_date as run_nba_daily


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run sports betting model (NBA now; NFL/NHL later).")

    parser.add_argument("--sport", type=str, default="nba", choices=["nba", "nfl", "nhl"], help="Which sport to run")
    parser.add_argument("--date", type=str, default=None, help="Game date in MM/DD/YYYY (default: today UTC).")

    # Backtest flags kept for later expansion (NBA backtest still lives in NBA module until we refactor fully)
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, help="Starting bankroll (default 250)")
    parser.add_argument("--sizing", type=str, default="flat", choices=["flat", "kelly"], help="Sizing mode")
    parser.add_argument("--flat_pct", type=float, default=UNIT_PCT, help="Flat stake percent (default 0.04)")
    parser.add_argument("--kelly_mult", type=float, default=0.5, help="Kelly multiplier (0.5 = half Kelly)")
    parser.add_argument("--kelly_max_pct", type=float, default=0.03, help="Max Kelly stake pct (cap)")

    parser.add_argument("--play_require_pick", action="store_true", help="Require 'PICK' in primary to PLAY")
    parser.add_argument("--play_value_tier", type=str, default="HIGH VALUE", help="Require value_tier")
    parser.add_argument("--play_min_conf", type=str, default="MEDIUM", choices=["LOW", "MEDIUM", "HIGH"], help="Min confidence")
    parser.add_argument("--play_max_abs_ml", type=int, default=400, help="Pass if selected ML |odds| > this (0 disables)")

    args = parser.parse_args(argv)

    # Date
    if args.date is None:
        game_date = datetime.utcnow().strftime("%m/%d/%Y")
    else:
        game_date = args.date

    print(f"Running {args.sport.upper()} model for {game_date}...")

    # Odds (API first, fallback CSV)
    odds_dict, spreads_dict = {}, {}
    try:
        odds_dict, spreads_dict = fetch_odds_for_date_from_odds_api(
            game_date,
            sport_key=SPORT_TO_ODDS_KEY[args.sport],
        )
        if odds_dict:
            print(f"[odds_api] Loaded odds for {len(odds_dict)} games.")
        else:
            print("[odds_api] No odds returned; will try CSV fallback.")
    except Exception as e:
        print(f"[odds_api] WARNING: failed to load odds from API: {e}")

    if not odds_dict:
        try:
            odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date, sport=args.sport)
        except Exception as e:
            print(f"[odds_csv] WARNING: failed to load odds from CSV: {e}")
            odds_dict, spreads_dict = {}, {}

    # Run sport model
    if args.sport == "nba":
        api_key = get_bdl_api_key()
        game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
        season_year = season_start_year_for_date(game_date_obj)
        end_date_iso = game_date_obj.strftime("%Y-%m-%d")
        stats_df = fetch_team_ratings_bdl(season_year=season_year, end_date_iso=end_date_iso, api_key=api_key)

        results_df = run_nba_daily(
            game_date=game_date,
            odds_dict=odds_dict,
            spreads_dict=spreads_dict,
            stats_df=stats_df,
            api_key=api_key,
        )
    else:
        raise RuntimeError(f"{args.sport.upper()} not wired yet. NBA is working; NFL/NHL coming next.")

    if results_df is None or results_df.empty:
        print("[model] No games returned.")
        return

    # Recommendations
    results_df, debug_df = add_recommendations_to_df(
        results_df,
        thresholds=Thresholds(
            ml_edge_strong=0.06,
            ml_edge_lean=0.035,
            ats_edge_strong_pts=3.0,
            ats_edge_lean_pts=1.5,
            conf_high=0.18,
            conf_med=0.10,
        ),
        model_spread_home_col="model_spread_home",
        model_margin_home_col=None,
    )

    # Play/pass + sizing
    play_max_abs_ml = None if args.play_max_abs_ml == 0 else args.play_max_abs_ml

    results_df["play_pass"] = results_df.apply(
        lambda r: play_pass_rule(
            r,
            require_pick=args.play_require_pick,
            require_value_tier=args.play_value_tier,
            min_confidence=args.play_min_conf,
            max_abs_moneyline=play_max_abs_ml,
        ),
        axis=1,
    )

    results_df["bet_size"] = results_df.apply(
        lambda r: compute_bet_size(
            r,
            args.bankroll,
            sizing_mode=args.sizing,
            flat_pct=args.flat_pct,
            kelly_mult=args.kelly_mult,
            kelly_max_pct=args.kelly_max_pct,
        ),
        axis=1,
    )

    unit_dollars = float(args.bankroll) * UNIT_PCT
    results_df["unit_dollars"] = unit_dollars
    results_df["units"] = results_df["bet_size"].apply(lambda x: 0.0 if not x else float(x) / unit_dollars)

    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{args.sport}_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    if debug_df is not None and not debug_df.empty:
        dbg_name = f"results/debug_why_ml_vs_ats_{args.sport}_{game_date.replace('/', '-')}.csv"
        debug_df.to_csv(dbg_name, index=False)

    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")
    print(f"Bankroll=${float(args.bankroll):.2f} | 1 unit={UNIT_PCT*100:.1f}% = ${unit_dollars:.2f}")


if __name__ == "__main__":
    main()
