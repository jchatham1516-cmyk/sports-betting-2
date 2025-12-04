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

    # Fetch odds from API-Sports
    try:
        odds_dict = fetch_odds_for_date_apisports(
            game_date_str=game_date,
            api_key=get_apisports_api_key(),
        )
        spreads_dict = {
            k: v["home_spread"]
            for k, v in odds_dict.items()
            if v.get("home_spread") is not None
        }
        print(f"Fetched odds for {len(odds_dict)} games from API-Sports.")
    except Exception as e:
        print(f"Warning: failed to fetch odds from API-Sports: {e}")
        odds_dict = {}
        spreads_dict = {}
        print("Proceeding with market_home_prob = 0.5 defaults.")

    # Determine season year for BallDontLie based on the game date
    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)
    end_date_iso = game_date_obj.strftime("%Y-%m-%d")

    # Fetch team ratings from BallDontLie
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

    # Run daily model; also fail gracefully if something blows up
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

    # Ensure output directory
    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    # Pretty print to console
    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")


if __name__ == "__main__":
    main()
