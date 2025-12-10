def main(argv=None):
    parser = argparse.ArgumentParser(description="Run daily NBA betting model (BallDontLie).")

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Game date in MM/DD/YYYY (default: today in UTC).",
    )

    # Training mode (you can ignore this unless needed)
    parser.add_argument(
        "--train-season",
        type=int,
        default=None,
        help="Train matchup weights on a full NBA season (e.g. 2023).",
    )

    args = parser.parse_args(argv)

    api_key = get_bdl_api_key()

    # --------------------------
    # TRAINING MODE
    # --------------------------
    if args.train_season is not None:
        season_year = args.train_season
        print(f"[MAIN] Training matchup weights for season {season_year}...")
        train_matchup_weights(season_year, api_key)
        return

    # --------------------------
    # NORMAL DAILY MODE
    # --------------------------
    if args.date is None:
        today = datetime.utcnow().date()
        game_date = today.strftime("%m/%d/%Y")
    else:
        game_date = args.date

    print(f"Running model for {game_date}...")

    # 1) Ensure odds template exists
    build_odds_csv_template_if_missing(game_date, api_key=api_key)

    # 2) Load odds
    try:
        odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date)
        print(f"Loaded odds for {len(odds_dict)} games from CSV.")
    except Exception as e:
        print(f"Warning: failed to load odds: {e}")
        odds_dict = {}
        spreads_dict = {}

    # 3) Determine season year
    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)
    end_date_iso = game_date_obj.strftime("%Y-%m-%d")

    # 4) Fetch team ratings
    try:
        stats_df = fetch_team_ratings_bdl(
            season_year=season_year,
            end_date_iso=end_date_iso,
            api_key=api_key,
        )
    except Exception as e:
        print(f"Error: failed to fetch team ratings: {e}")
        return

    # 5) Run model
    try:
        results_df = run_daily_probs_for_date(
            game_date=game_date,
            odds_dict=odds_dict,
            spreads_dict=spreads_dict,
            stats_df=stats_df,
            api_key=api_key,
        )
    except Exception as e:
        print(f"Error: failed to run daily model: {e}")
        return

    # 6) Save output CSV
    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")
