from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sports_betting.models.types import GameContext, MarketPrediction
from sports_betting.sports.common.base import SportModel
from sports_betting.sports.common.confidence import confidence_score
from sports_betting.sports.common.ev import edge, expected_value
from sports_betting.sports.common.odds import american_to_decimal, american_to_implied_probability
from sports_betting.sports.nba.features import NBA_FEATURE_COLUMNS, build_nba_features
from sports_betting.sports.nba.training import load_trained_nba_models, train_nba_models


class NbaModel(SportModel):
    sport = "NBA"

    def __init__(self, model_dir: str = "sports_betting/data/models", historical_csv: str = "sports_betting/data/historical/nba_historical.csv"):
        self.model_dir = Path(model_dir)
        self.historical_csv = Path(historical_csv)
        self._trained_models = self._load_or_train_models()

    def _load_or_train_models(self) -> dict[str, Any] | None:
        loaded = load_trained_nba_models(self.model_dir)
        if loaded is not None:
            return loaded

        if not self.historical_csv.exists():
            return None

        try:
            df = pd.read_csv(self.historical_csv)
            if not {"home_win", "home_cover", "over_hit", "date", "game_id"}.issubset(df.columns):
                return None
            if df[["home_win", "home_cover", "over_hit"]].dropna().shape[0] < 200:
                return None
            trained = train_nba_models(df)
            return {
                "moneyline": trained.moneyline,
                "spread": trained.spread,
                "totals": trained.totals,
            }
        except Exception:
            return None

    def _build_game_feature_row(self, game: GameContext) -> pd.DataFrame:
        ratings = game.meta.get("team_ratings", {})
        home = ratings.get(game.home_team, {})
        away = ratings.get(game.away_team, {})

        market_prob_home = 0.5
        for m in game.markets:
            if m.market == "moneyline" and m.side == game.home_team:
                market_prob_home = american_to_implied_probability(m.american_odds)
                break

        feature_row = pd.DataFrame(
            [
                {
                    "elo_home": home.get("elo", home.get("power_rating", 1500)),
                    "elo_away": away.get("elo", away.get("power_rating", 1500)),
                    "rest_days_home": home.get("rest_days", 1),
                    "rest_days_away": away.get("rest_days", 1),
                    "injury_impact_home": home.get("injury_impact", 0),
                    "injury_impact_away": away.get("injury_impact", 0),
                    "net_rating_home": home.get("net_rating", 0),
                    "net_rating_away": away.get("net_rating", 0),
                    "pace_home": home.get("pace", 99),
                    "pace_away": away.get("pace", 99),
                    "last5_net_rating_home": home.get("last5_net_rating", home.get("net_rating", 0)),
                    "last5_net_rating_away": away.get("last5_net_rating", away.get("net_rating", 0)),
                    "last10_net_rating_home": home.get("last10_net_rating", home.get("net_rating", 0)),
                    "last10_net_rating_away": away.get("last10_net_rating", away.get("net_rating", 0)),
                    "travel_distance_home": home.get("travel_distance", 0),
                    "travel_distance_away": away.get("travel_distance", 0),
                    "timezone_shift_home": home.get("timezone_shift", 0),
                    "timezone_shift_away": away.get("timezone_shift", 0),
                    "road_trip_length_home": home.get("road_trip_length", 0),
                    "road_trip_length_away": away.get("road_trip_length", 0),
                    "back_to_back_home": home.get("back_to_back", 0),
                    "back_to_back_away": away.get("back_to_back", 0),
                    "three_in_four_home": home.get("three_in_four", 0),
                    "three_in_four_away": away.get("three_in_four", 0),
                    "market_prob_home": market_prob_home,
                }
            ]
        )
        return build_nba_features(feature_row)

    def predict_game(self, game: GameContext) -> list[MarketPrediction]:
        ratings = game.meta.get("team_ratings", {})
        home = ratings.get(game.home_team, {})
        away = ratings.get(game.away_team, {})

        net_diff = home.get("net_rating", 0) - away.get("net_rating", 0)
        rest_diff = home.get("rest_days", 1) - away.get("rest_days", 1)
        injury_diff = away.get("injury_impact", 0) - home.get("injury_impact", 0)

        # Handcrafted fallback probabilities keep the pipeline resilient.
        base_win_prob = min(0.92, max(0.08, 0.5 + net_diff * 0.018 + rest_diff * 0.01 + injury_diff * 0.02))
        spread_cover_prob = min(0.9, max(0.1, base_win_prob + net_diff * 0.007))
        pace_total_adj = (home.get("pace", 99) + away.get("pace", 99) - 198) * 0.003
        over_prob = min(0.88, max(0.12, 0.5 + pace_total_adj + net_diff * 0.0015))

        if self._trained_models:
            try:
                x = self._build_game_feature_row(game)[NBA_FEATURE_COLUMNS]
                base_win_prob = float(self._trained_models["moneyline"].predict_proba(x)[:, 1][0])
                spread_cover_prob = float(self._trained_models["spread"].predict_proba(x)[:, 1][0])
                over_prob = float(self._trained_models["totals"].predict_proba(x)[:, 1][0])
            except Exception:
                pass

        preds: list[MarketPrediction] = []
        for m in game.markets:
            market_prob = american_to_implied_probability(m.american_odds)
            decimal_odds = american_to_decimal(m.american_odds)
            if m.market == "moneyline":
                model_prob = base_win_prob if m.side == game.home_team else (1 - base_win_prob)
                reasons = ["historical win model", "rest/injury adjustments", "market anchor"]
            elif m.market == "spread":
                model_prob = spread_cover_prob if m.side == game.home_team else (1 - spread_cover_prob)
                reasons = ["historical spread model", "net rating differential", "form and fatigue"]
            else:
                model_prob = over_prob if m.side.lower() == "over" else (1 - over_prob)
                reasons = ["historical totals model", "pace and efficiency signal", "injury/rotation context"]

            ev = expected_value(model_prob, decimal_odds)
            e = edge(model_prob, market_prob)
            conf = confidence_score(e, data_quality=0.86, calibration_quality=0.75)
            preds.append(
                MarketPrediction(
                    event_id=game.event_id,
                    date=game.date.date().isoformat(),
                    sport=self.sport,
                    game=f"{game.away_team} @ {game.home_team}",
                    market=m.market,
                    side=m.side,
                    line=m.line,
                    sportsbook_odds=m.american_odds,
                    model_probability=round(model_prob, 4),
                    market_probability=round(market_prob, 4),
                    edge=round(e, 4),
                    expected_value=round(ev, 4),
                    confidence=conf,
                    model_quality=0.75 if self._trained_models else 0.72,
                    explanation=reasons,
                    flags=[] if self._trained_models else ["fallback_model"],
                )
            )
        return preds
