def main(argv=None):
    parser = argparse.ArgumentParser(description="Run daily NBA betting model (BallDontLie).")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Game date in MM/DD/YYYY (default: today in UTC).",
    )
    args = parser.parse_args(argv)

    if args.date is None:
        today = datetime.utcnow().date()
        game_date = today.strftime("%m/%d/%Y")
    else:
        game_date = args.date

    print(f"Running model for {game_date}...")

    bdl_api_key = get_bdl_api_key()

    # 1) Load odds from local CSV (manual odds)
    try:
        odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date)
        print(f"Loaded odds for {len(odds_dict)} games from CSV.")
    except Exception as e:
        print(f"Warning: failed to load odds from CSV: {e}")
        odds_dict = {}
        spreads_dict = {}
        print("Proceeding with market_home_prob = 0.5 defaults.")

    # 2) Determine season year for BallDontLie based on the game date
    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)
    end_date_iso = game_date_obj.strftime("%Y-%m-%d")

    # 3) Fetch team ratings from BallDontLie
    try:
        stats_df = fetch_team_ratings_bdl(
            season_year=season_year,
            end_date_iso=end_date_iso,
            api_key=bdl_api_key,
        )
    except Exception as e:
        print(f"Error: Failed to fetch team ratings from BallDontLie: {e}")
        print("Exiting without predictions so the workflow can complete gracefully.")
        return

    # 4) Run daily model
    try:
        results_df = run_daily_probs_for_date(
            game_date=game_date,
            odds_dict=odds_dict,
            spreads_dict=spreads_dict,
            stats_df=stats_df,
            api_key=bdl_api_key,
        )
    except Exception as e:
        print(f"Error: Failed to run daily model: {e}")
        print("Exiting without predictions so the workflow can complete gracefully.")
        return

    # 5) Save CSV
    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")


if __name__ == "__main__":
    main()
