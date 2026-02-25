# Sports Betting System (NBA/NHL/NFL)

Production-ready, deterministic Python 3.11 sports betting framework focused on long-run EV.

## Features
- Odds ingestion from **The Odds API** only (`h2h`, `spreads`, `totals`).
- Team name normalization, odds conversion, and no-vig market probabilities.
- Elo baseline with sport-specific adjustments:
  - NBA: rest modifier
  - NHL: goalie adjustment (shrunk + capped) + missing-data uncertainty penalty
  - NFL: injury/QB/weather adjustments (shrunk + capped) + missing-data uncertainty penalty
- Optional probability calibration (isotonic or Platt) per sport/market.
- Fractional Kelly staking with strict thresholds and caps.
- Raw odds snapshots in `data/raw/` and picks output in `data/results/`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment
Create `.env`:
```bash
ODDS_API_KEY=your_key_here
```

## Run Daily Picks
```bash
python main.py --sport nba --date 2026-01-15 --days 1
python main.py --sport nhl --date 2026-01-15 --days 1 --debug
python main.py --sport nfl --date 2026-01-15 --days 2
```

Output CSV schema (`data/results/picks_YYYYMMDD.csv`):
- `date,sport,home,away,market_type,model_prob,market_prob,edge,odds,ev,bet_units,play_pass,decision_reason,flags,inputs_used,model_version`

## Optional Input Files
Copy samples and edit:
```bash
cp data/inputs/injuries.sample.json data/inputs/injuries.json
cp data/inputs/goalies.sample.json data/inputs/goalies.json
cp data/inputs/weather.sample.json data/inputs/weather.json
```

Formats:
- `injuries.json`
  - `nba_rest_days`: mapping of team -> rest days
  - NFL by event id: `home_injury_delta`, `home_qb_delta`
- `goalies.json`
  - NHL by event id: `home_goalie_delta`
- `weather.json`
  - NFL by event id: `home_weather_delta`

All adjustment fields are interpreted as probability deltas and are always shrunk and hard-capped.

## Train Calibrators
From local historical predictions/outcomes CSV:
```bash
python scripts/train_calibrators.py --history data/results/history_with_outcomes.csv --method isotonic --min-samples 50
```

Required history columns: `sport,market_type,model_prob,outcome`

## Backtest
```bash
python scripts/backtest.py --picks data/results/history_with_outcomes.csv --bins 10
```

Backtest reports: ROI, yield, win rate, Brier score, max drawdown, and calibration bins.

## Testing
```bash
python -m pytest -q
```

## Guardrails
- Minimum edge and EV required before any bet.
- Fractional Kelly stake only.
- Minimum/maximum bet-size caps.
- Missing optional inputs reduce confidence via uncertainty penalties.
- Optional inputs are shrunk + capped to prevent dominance.
