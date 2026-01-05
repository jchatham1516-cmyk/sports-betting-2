# Automatic Bet Tracking

This project can now grade and log completed bets automatically using the model outputs in `results/`.

## How it works

1. The tracker locates the most recent recommendations file for the target date (prefers filenames with the date in them, e.g. `predictions_nba_01-05-2026.csv`).
2. Bets are normalized into one-row-per-bet (moneyline, spread, totals) using the model's recommendations and bet sizing.
3. Final scores are fetched from The Odds API historical endpoint.
4. Each bet is graded (WIN/LOSS/PUSH/MISSING_SCORE) and profit/ROI is calculated from the stake and odds.
5. Results are written to `results/tracking/` for easy inspection and long-term logging.

## Running tracking

After your normal daily run you can add one of the tracking flags:

```bash
python current_of_sports_betting_algorithm.py --sport nba --track_yesterday
# or
python current_of_sports_betting_algorithm.py --sport nba --track_date 2026-01-05
```

Flags:

- `--track_yesterday`: grades the previous day's bets after generating today's picks.
- `--track_date YYYY-MM-DD`: grades bets for a specific date.

Tracking does **not** change the normal CLI defaults; if no tracking flag is provided nothing extra runs.

## Outputs

All tracking artifacts live under `results/tracking/`:

- `bets_YYYY-MM-DD_graded.csv`: graded bets for the requested date.
- `bet_history.csv`: cumulative log of all graded bets (de-duplicated by bet identity).
- `summary.json`: lifetime + last-30-day stats (bets, win%, ROI, profit, max drawdown).

## Grading rules

- Moneyline: HOME wins if `home_score > away_score` (AWAY otherwise).
- Spread: uses the home spread line; HOME wins if `home_score + spread > away_score`, AWAY if `away_score - spread > home_score`, PUSH on equality.
- Totals: OVER wins if combined score is above the total line, UNDER if below, PUSH on equality.
- Missing or incomplete scores are marked `MISSING_SCORE` instead of raising an error.

## Profit/ROI

Profit is calculated from the stake and odds (American or decimal):

- WIN: `stake * (decimal_odds - 1)`
- LOSS: `-stake`
- PUSH: `0`

ROI in the summary is `total_profit / total_stake` for graded bets.
