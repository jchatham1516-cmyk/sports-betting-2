# Model Overview

## Pipeline entry points
- `current_of_sports_betting_algorithm.py` is the main multi-sport CLI. It pulls odds (API then CSV), routes to the sport-specific model (`nba`, `nfl`, `nhl`), attaches recommendations, applies play-pass filters, sizes bets, caps to a maximum number of plays, and writes results CSVs plus optional debug/eval files.【F:current_of_sports_betting_algorithm.py†L91-L200】
- Sport-specific prediction logic lives under `sports/<sport>/model.py` (e.g., `sports/nba/model.py`), while injuries and helpers sit in sibling modules such as `sports/nba/injuries.py`.
- Recommendations and value tiers are assigned via `recommendations.add_recommendations_to_df`, and bet sizing/guardrails are handled in `sports/common/bankroll.py`.

## Moneyline win probability creation
- NBA: Elo ratings updated from recent scores feed the win model. Injury and recent-form Elo adjustments are applied, then an Elo-based win probability is compressed toward 0.5, optionally Platt-calibrated, shrunk by uncertainty, and blended 65/35 with market-implied no-vig probability before edges are computed.【F:sports/nba/model.py†L672-L707】
- NFL/NHL follow similar patterns: Elo-derived probabilities, optional Platt calibration, and margin calibration functions, with sport-specific parameters in their respective model files.

## Spread and total derivation
- Spread: NBA converts Elo difference (with injury/form adjustments) to a model spread, optionally corrected by a fitted margin calibration curve and bounded by a max absolute spread before downstream ATS probability and edge calculations.【F:sports/nba/model.py†L672-L747】【F:sports/nba/model.py†L828-L833】
- Totals: Offense/defense blends, pace adjustments, and injury total adjustments produce a model total that is blended with historical team/league anchors and an affine total-line calibration. Edges are measured via normal-model over/under probabilities versus breakeven odds, with minimum point and edge thresholds gating picks.【F:sports/nba/model.py†L734-L827】

## Injury handling
- NBA injuries are scraped from the official NBA report with ESPN fallback. Players are mapped to (name, role, status multiplier, impact points) tuples; status multipliers vary by availability, and positional impact weights act as player value proxies. Starters/doubtful statuses push weights higher, and season-ending notes enforce full impact.【F:sports/nba/injuries.py†L240-L327】
- Team injury points equal away minus home weighted costs; they translate to Elo and total adjustments in the NBA model. A dampening factor and configurable cap (`MAX_ABS_INJ_POINTS`) limit the adjustment magnitude.【F:sports/nba/model.py†L672-L740】

## Edge computation vs. market odds
- Market moneylines are converted to implied probabilities and de-vigged. Model probabilities (raw and blended) are compared to market to compute `edge_home/edge_away` and select moneyline recommendations when the edge exceeds configured thresholds.【F:sports/nba/model.py†L666-L709】【F:recommendations.py†L86-L138】
- ATS edges compare market spread to model spread; totals edges compare model total to market line and breakeven odds with minimum thresholds for action.【F:sports/nba/model.py†L828-L887】【F:recommendations.py†L140-L181】

## Picks and bet sizing
- `add_recommendations_to_df` tags each game with moneyline/ATS/total picks, confidence labels (based on absolute edge), value tiers, and a primary recommendation per sport ordering.【F:recommendations.py†L64-L200】
- Play filtering (`play_pass_rule`) enforces minimum value tier, confidence, pick presence, and moneyline cap. Bet sizing supports flat percentage of bankroll or fractional Kelly for moneylines (bounded by `kelly_mult` and `kelly_max_pct`). Unit size equals 4% of bankroll by default, and top-N play capping can zero out excess bets.【F:sports/common/bankroll.py†L11-L102】【F:current_of_sports_betting_algorithm.py†L173-L200】【F:current_of_sports_betting_algorithm.py†L37-L68】

## Outputs and evaluation/backtesting
- Daily runs write `results/predictions_<sport>_<date>.csv` and optional debug why-logs plus rolling evaluation snapshots when scores CSVs are found.【F:current_of_sports_betting_algorithm.py†L173-L209】
- `backtest.py` currently loops through dates, reusing daily predictions and placeholder bankroll updates; settlement logic is not yet implemented (marked TODO).【F:backtest.py†L5-L55】
