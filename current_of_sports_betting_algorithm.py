# current_of_sports_betting_algorithm.py
#
# Thin runner:
# - Loads odds (Odds API with CSV fallback)
# - Runs per-sport model (NBA/NFL/NHL)
# - Applies recommendations + play/pass + sizing
# - Saves results to results/predictions_<sport>_<MM-DD-YYYY>.csv

from __future__ import annotations

import os
import argparse
from datetime import datetime

import pandas as pd

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

from sports.nba.bdl_client import (
    get_bdl_api_key,
    season_start_year_for_date,
    fetch_team_ratings_bdl,
)
from sports.nba.model import run_daily_probs_for_date as run_nba_daily

from sports.nfl.model import run_daily_nfl
from sports.nhl.model import run_daily_nhl


def _cap_to_top_plays(df: pd.DataFrame, max_plays: int) -> pd.DataFrame:
    """
    If play_pass == PLAY is more than max_plays, keep only top max_plays by pick_score (desc).
    Everyone else gets set to PASS and bet_size=0.
    """
    if df is None or df.empty:
        return df
    if max_plays is None:
        return df

    if "play_pass" not in df.columns:
        return df

    plays = df[df["play_pass"].astype(str) == "PLAY"].copy()
    if plays.empty:
        return df

    if "pick_score" in df.columns:
        plays = plays.sort_values("pick_score", ascending=False)
    else:
        # fallback: abs_edge_home if present
        if "abs_edge_home" in df.columns:
            plays = plays.assign(_score=plays["abs_edge_home"].astype(float))
            plays = plays.sort_values("_score", ascending=False)
        else:
            plays = plays.iloc[:]

    if len(plays) <= int(max_plays):
        return df

    keep_ids = set(plays.head(int(max_plays)).index.tolist())
    for i in df.index:
        if str(df.loc[i, "play_pass"]) == "PLAY" and i not in keep_ids:
            df.loc[i, "play_pass"] = "PASS"
            if "bet_size" in df.columns:
                df.loc[i, "bet_size"] = 0.0
            if "units" in df.columns:
                df.loc[i, "units"] = 0.0
            if "why_bet" in df.columns:
                df.loc[i, "why_bet"] = str(df.loc[i, "why_bet"]) + " | filtered: top-N plays"
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run sports betting model (NBA/NFL/NHL).")
    parser.add_argument("--sport", type=str, default="nba", choices=["nba", "nfl", "nhl"])
    parser.add_argument("--date", type=str, default=None, help="Game date in MM/DD/YYYY (default: today UTC).")

    # Odds API window padding (days)
    parser.add_argument("--days_padding", type=int, default=int(os.getenv("ODDS_DAYS_PADDING", "1")))

    # Sizing
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--sizing", type=str, default="flat", choices=["flat", "kelly"])
    parser.add_argument("--flat_pct", type=float, default=UNIT_PCT)
    parser.add_argument("--kelly_mult", type=float, default=0.5)
    parser.add_argument("--kelly_max_pct", type=float, default=0.03)

    # Play/pass
    parser.add_argument("--play_require_pick", action="store_true")
    parser.add_argument("--play_value_tier", type=str, default="HIGH VALUE")
    parser.add_argument("--play_min_conf", type=str, default="MEDIUM", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--play_max_abs_ml", type=int, default=400)

    # Cap plays
    parser.add_argument("--max_plays", type=int, default=int(os.getenv("MAX_PLAYS_PER_SPORT_PER_DAY", "3")))

    # Elo rebuild for NBA
    parser.add_argument("--force_full_rebuild", action="store_true", help="Force full Elo backfill before daily run.")

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
            days_padding=int(args.days_padding),
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
            print(f"[odds_csv] games found: {len(odds_dict)}")
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
            force_full_rebuild=args.force_full_rebuild,
        )

    elif args.sport == "nfl":
        results_df = run_daily_nfl(game_date, odds_dict=odds_dict)

    elif args.sport == "nhl":
        results_df = run_daily_nhl(game_date, odds_dict=odds_dict)

    else:
        raise RuntimeError("Unsupported sport")

    if results_df is None:
        print("[model] No dataframe returned.")
        results_df = pd.DataFrame([])

    print(f"[model] rows returned: {len(results_df)}")

    # Recommendations
    debug_df = pd.DataFrame([])
    if not results_df.empty:
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
            model_spread_home_col="model_spread_home" if "model_spread_home" in results_df.columns else None,
            model_margin_home_col=None,
        )

    # Play/pass + sizing
    play_max_abs_ml = None if int(args.play_max_abs_ml) == 0 else int(args.play_max_abs_ml)
    unit_dollars = float(args.bankroll) * UNIT_PCT

    if not results_df.empty:
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

        results_df["unit_dollars"] = unit_dollars
        results_df["units"] = results_df["bet_size"].apply(lambda x: 0.0 if not x else float(x) / unit_dollars)

        # cap to top plays
        results_df = _cap_to_top_plays(results_df, int(args.max_plays))

    # Save (even if empty, so Actions artifact exists)
    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{args.sport}_{game_date.replace('/', '-')}.csv"
    print(f"[save] writing {len(results_df)} rows -> {out_name}")
    results_df.to_csv(out_name, index=False)

    if debug_df is not None and not debug_df.empty:
        dbg_name = f"results/debug_why_ml_vs_ats_{args.sport}_{game_date.replace('/', '-')}.csv"
        debug_df.to_csv(dbg_name, index=False)

    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")
    print(f"Bankroll=${float(args.bankroll):.2f} | 1 unit={UNIT_PCT*100:.1f}% = ${unit_dollars:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
