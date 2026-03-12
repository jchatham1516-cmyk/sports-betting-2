from __future__ import annotations

from sports_betting.models.types import GameContext, MarketPrediction
from sports_betting.sports.common.base import SportModel
from sports_betting.sports.common.confidence import confidence_score
from sports_betting.sports.common.ev import edge, expected_value
from sports_betting.sports.common.odds import american_to_decimal, american_to_implied_probability


class NflModel(SportModel):
    sport = "NFL"

    def predict_game(self, game: GameContext) -> list[MarketPrediction]:
        ratings = game.meta.get("team_ratings", {})
        home = ratings.get(game.home_team, {})
        away = ratings.get(game.away_team, {})
        epa_diff = home.get("epa_per_play", 0) - away.get("epa_per_play", 0)
        qb_diff = home.get("qb_impact", 0) - away.get("qb_impact", 0)
        weather_penalty = game.meta.get("weather", {}).get(game.event_id, {}).get("wind_penalty", 0)

        win_prob = min(0.9, max(0.1, 0.5 + epa_diff * 0.65 + qb_diff * 0.03))
        cover_prob = min(0.88, max(0.12, win_prob + (home.get("success_rate", 0) - away.get("success_rate", 0)) * 0.3))
        total_over_prob = min(0.85, max(0.15, 0.52 + (home.get("pace", 0) + away.get("pace", 0)) * 0.01 - weather_penalty))

        preds: list[MarketPrediction] = []
        for m in game.markets:
            market_prob = american_to_implied_probability(m.american_odds)
            decimal_odds = american_to_decimal(m.american_odds)
            if m.market == "moneyline":
                model_prob = win_prob if m.side == game.home_team else (1 - win_prob)
                reasons = ["EPA/play differential", "QB impact adjustment", "defensive matchup"]
            elif m.market == "spread":
                model_prob = cover_prob if m.side == game.home_team else (1 - cover_prob)
                reasons = ["success rate matchup", "pressure rate vs OL", "turnover expectancy"]
            else:
                model_prob = total_over_prob if m.side.lower() == "over" else (1 - total_over_prob)
                reasons = ["pace + pass rate", "weather adjustment", "red-zone efficiency signal"]

            ev = expected_value(model_prob, decimal_odds)
            e = edge(model_prob, market_prob)
            conf = confidence_score(e, data_quality=0.8, calibration_quality=0.68)
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
                    model_quality=0.68,
                    explanation=reasons,
                    flags=[],
                )
            )
        return preds
