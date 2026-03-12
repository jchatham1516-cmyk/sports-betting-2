# Production Sports Betting Engine (NBA / NFL / NHL)

A modular, auditable Python framework focused on **disciplined, long-term expected value** betting recommendations for moneyline, spread, and totals markets.

## Project Structure

```text
sports_betting/
  config/
    default.json
  data/
    raw/
    processed/
    historical/
    outputs/
  sports/
    common/
    nba/
    nfl/
    nhl/
  models/
  backtesting/
  scripts/
  main.py
main.py
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Daily Predictions

The pipeline expects prepared per-sport JSON schedule + odds files in `sports_betting/data/processed/`.

```bash
python main.py --date 2026-01-15 --config sports_betting/config/default.json --top-n 5
```

Outputs are generated in `sports_betting/data/outputs/`:
- `daily_recommendations_YYYYMMDD.csv`
- `daily_recommendations_YYYYMMDD.json`

Each recommendation includes:
- market, side, line, sportsbook odds
- model probability, market implied probability
- edge and EV
- confidence and units
- explanation and flags

## Backtesting

Use a predictions file with an `outcome` column (`1` win, `0` loss) to compute ROI and quality diagnostics:

```bash
python -m sports_betting.scripts.run_backtest --predictions sports_betting/data/historical/predictions_with_outcomes.csv
```

## Configuration

`default.json` controls:
- per-sport thresholds (`min_edge`, `min_ev`, `min_confidence`)
- bankroll/staking mode (flat or fractional Kelly)
- max/min unit caps
- uncertainty penalties

## Modeling Approach

- **NBA:** net rating, rest, injuries, pace-driven totals projection.
- **NFL:** EPA/play, QB impact, success rate, weather effects.
- **NHL:** xG, goalie strength, special teams with volatility-sensitive totals.

Design principle is interpretable baseline models now, with clean upgrade paths to calibrated ensembles later.

## Testing

```bash
pytest -q sports_betting/tests
```

Coverage includes:
- odds conversion and implied probabilities
- no-vig calculations
- EV and staking behavior
- threshold gating and output structure

## Profitability Philosophy

- This system **does not guarantee profits**.
- It prioritizes calibrated probabilities, edge discipline, and selective betting.
- Passes are expected and encouraged when edge/EV is insufficient.

## Known Limitations

- Baseline feature models are deterministic and intentionally simple.
- Real-time line movement, liquidity, and true CLV capture are stub-level.
- Injury/lineup uncertainty handling is basic and should be expanded.
- No automated training loop in this first pass.

## Highest-Impact Next Upgrades

1. Add walk-forward model training per market with strict time splits.
2. Store and model closing line value (CLV) to tune thresholds by realized edge retention.
3. Add market microstructure filters (steam moves, stale books, limit-aware staking).
4. Integrate richer injury/lineup/goalie/QB APIs with probabilistic uncertainty modeling.
5. Add calibrated ensemble layer (logistic baseline + gradient boosting blend).
