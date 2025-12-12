from datetime import datetime, timedelta
from current_of_sports_betting_algorithm import run_daily_probs_for_date
from recommendations import add_recommendations_to_df

def backtest(
    start_date,
    end_date,
    initial_bankroll=1000
):
    bankroll = initial_bankroll
    history = []

    cur = start_date
    while cur <= end_date:
        date_str = cur.strftime("%m/%d/%Y")

        try:
            df = run_daily_probs_for_date(
                game_date=date_str,
                odds_dict=None,      # load from CSV inside main
                spreads_dict=None,
                stats_df=stats_df,   # reuse your cached stats
                api_key=api_key,
            )
        except Exception:
            cur += timedelta(days=1)
            continue

        if df.empty:
            cur += timedelta(days=1)
            continue

        df, _ = add_recommendations_to_df(df)
        df["play_pass"] = df.apply(play_pass_rule, axis=1)

        for _, row in df[df["play_pass"] == "PLAY"].iterrows():
            bet = compute_bet_size(row, bankroll)
            if bet <= 0:
                continue

            # TODO: resolve bet using final score
            # win = True / False
            # payout = bet * (decimal_odds - 1) if win else -bet

            bankroll += payout

        history.append({
            "date": date_str,
            "bankroll": bankroll,
            "num_bets": len(df[df["play_pass"] == "PLAY"])
        })

        cur += timedelta(days=1)

    return pd.DataFrame(history)
