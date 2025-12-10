def run_daily_probs_for_date(
    game_date="12/04/2025",
    odds_dict=None,
    spreads_dict=None,
    stats_df=None,
    api_key=None,
    edge_threshold=0.03,
    lam=0.20,
):
    """
    Run the full model for one NBA date.
    """
    if api_key is None:
        api_key = get_bdl_api_key()

    if stats_df is None:
        raise ValueError("stats_df must be precomputed for BallDontLie version.")

    if odds_dict is None:
        odds_dict = {}

    if spreads_dict is None:
        spreads_dict = {}

    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)

    # Schedule from BallDontLie
    games_df = fetch_games_for_date(game_date, stats_df, api_key)

    # Injuries
    try:
        injury_df = fetch_injury_report_espn()
    except Exception as e:
        print(f"Warning: failed to fetch ESPN injuries: {e}")
        injury_df = pd.DataFrame(columns=["Player", "Team", "Pos", "Status", "Injury"])

    rows = []

    for _, g in games_df.iterrows():
        home_name = g["HOME_TEAM_NAME"]
        away_name = g["AWAY_TEAM_NAME"]

        home_row = find_team_row(home_name, stats_df)
        away_row = find_team_row(away_name, stats_df)
        home_id = int(home_row["TEAM_ID"])
        away_id = int(away_row["TEAM_ID"])

        # Base matchup
        base_score = season_matchup_base_score(home_row, away_row)

        # Injuries
        home_inj = build_injury_list_for_team_espn(home_name, injury_df)
        away_inj = build_injury_list_for_team_espn(away_name, injury_df)
        inj_adj = injury_adjustment(home_inj, away_inj)

        # Schedule fatigue
        home_last = get_team_last_game_date(home_id, game_date_obj, season_year, api_key)
        away_last = get_team_last_game_date(away_id, game_date_obj, season_year, api_key)

        home_rest_days = (game_date_obj - home_last).days if home_last else None
        away_rest_days = (game_date_obj - away_last).days if away_last else None

        home_fatigue = rest_days_to_fatigue_adjustment(home_rest_days)
        away_fatigue = rest_days_to_fatigue_adjustment(away_rest_days)

        fatigue_adj = home_fatigue - away_fatigue  # positive helps home

        # Head-to-head historical adjustment
        h2h_adj = compute_head_to_head_adjustment(home_id, away_id, season_year, api_key)

        # Final score
        adj_score = base_score + inj_adj + fatigue_adj + h2h_adj

        # Model win prob & spread
        model_home_prob = score_to_prob(adj_score, lam)
        model_spread = score_to_spread(adj_score)  # VEGAS STYLE: negative = home favorite

        # -------------------------
        # Market odds (ML)
        # -------------------------
        key = (home_name, away_name)
        odds_info = odds_dict.get(key)
        if odds_info is None:
            print(f"[run_daily] No odds found for {home_name} vs {away_name}")
            odds_info = {}

        home_ml = odds_info.get("home_ml")
        away_ml = odds_info.get("away_ml")

        # Convert American odds -> fair win probabilities (vig removed)
        if home_ml is not None and away_ml is not None:
            raw_home_prob = american_to_implied_prob(home_ml)
            raw_away_prob = american_to_implied_prob(away_ml)
            total = raw_home_prob + raw_away_prob
            if total > 0:
                home_imp = raw_home_prob / total
                away_imp = raw_away_prob / total
            else:
                home_imp = away_imp = 0.5
        elif home_ml is not None:
            home_imp = american_to_implied_prob(home_ml)
            away_imp = 1.0 - home_imp
        elif away_ml is not None:
            away_imp = american_to_implied_prob(away_ml)
            home_imp = 1.0 - away_imp
        else:
            home_imp = away_imp = 0.5

        edge_home = model_home_prob - home_imp
        edge_away = (1.0 - model_home_prob) - away_imp

        # -------------------------
        # Spreads (Vegas style)
        # -------------------------
        home_spread = spreads_dict.get(key, odds_info.get("home_spread"))
        if home_spread is not None:
            home_spread = float(home_spread)

            # spread_edge_home > 0  => model likes HOME vs this line
            # spread_edge_home < 0  => model likes AWAY vs this line
            #
            # Example:
            #   model_spread = -5   (home -5)
            #   home_spread = -10   (home -10)
            #   spread_edge_home = -10 - (-5) = -5  => prefers away +10
            spread_edge_home = home_spread - model_spread
        else:
            spread_edge_home = None

        # -------------------------
        # Recommendation logic
        # -------------------------

        # 1) Moneyline recommendation
        ml_rec = "No strong ML edge"
        if home_ml is not None or away_ml is not None:
            if edge_home > edge_threshold and home_ml is not None:
                ml_rec = f"Bet HOME ML ({home_ml:+})"
            elif edge_away > edge_threshold and away_ml is not None:
                ml_rec = f"Bet AWAY ML ({away_ml:+})"

        # 2) Spread recommendation (separate threshold in points)
        spread_rec = "No strong spread edge"
        spread_threshold_pts = 3.0  # tweak if you want tighter/looser filter

        if home_spread is not None and spread_edge_home is not None:
            if spread_edge_home > spread_threshold_pts:
                # model likes home side vs line
                if home_spread > 0:
                    line_str = f"home +{abs(home_spread)}"
                elif home_spread < 0:
                    line_str = f"home {home_spread}"
                else:
                    line_str = "home pk"
                spread_rec = f"Bet HOME spread ({line_str})"

            elif spread_edge_home < -spread_threshold_pts:
                # model likes away side vs line
                if home_spread > 0:
                    line_str = f"away -{abs(home_spread)}"
                elif home_spread < 0:
                    line_str = f"away +{abs(home_spread)}"
                else:
                    line_str = "away pk"
                spread_rec = f"Bet AWAY spread ({line_str})"

        # 3) Primary recommendation: compare ML vs spread on a similar scale
        primary_rec = "No clear edge"

        # Only count ML edge if we actually like a side
        if ml_rec.startswith("Bet HOME") and home_ml is not None:
            ml_edge_abs = abs(edge_home)
        elif ml_rec.startswith("Bet AWAY") and away_ml is not None:
            ml_edge_abs = abs(edge_away)
        else:
            ml_edge_abs = 0.0

        # Only count spread edge if we actually like a side
        if spread_edge_home is not None and spread_rec != "No strong spread edge":
            spread_edge_prob = min(abs(spread_edge_home) * 0.04, 0.5)  # cap at 50% edge
        else:
            spread_edge_prob = 0.0

        if ml_edge_abs == 0.0 and spread_edge_prob == 0.0:
            primary_rec = "No clear edge"
        elif ml_edge_abs >= spread_edge_prob:
            primary_rec = f"ML: {ml_rec}"
        else:
            primary_rec = f"Spread: {spread_rec}"

        rows.append(
            {
                "date": game_date,
                "home": home_name,
                "away": away_name,
                "model_home_prob": model_home_prob,
                "market_home_prob": home_imp,
                "edge_home": edge_home,
                "edge_away": edge_away,
                "model_spread_home": model_spread,  # vegas-style (negative = fav)
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_spread": home_spread,
                "spread_edge_home": spread_edge_home,
                "ml_recommendation": ml_rec,
                "spread_recommendation": spread_rec,
                "primary_recommendation": primary_rec,
            }
        )

    df = pd.DataFrame(rows)

    # ------------------------------------
    # Add absolute edge column (best side)
    # ------------------------------------
    df["abs_edge_home"] = df[["edge_home", "edge_away"]].abs().max(axis=1)

    # ------------------------------------
    # Add Value Tier Classification
    # ------------------------------------
    def classify_value(edge):
        if edge >= 0.20:
            return "HIGH VALUE"
        elif edge >= 0.10:
            return "MEDIUM VALUE"
        else:
            return "LOW VALUE"

    df["value_tier"] = df["abs_edge_home"].apply(classify_value)

    # ------------------------------------
    # Sort by strongest edges
    # ------------------------------------
    df = df.sort_values("abs_edge_home", ascending=False).reset_index(drop=True)

    return df
