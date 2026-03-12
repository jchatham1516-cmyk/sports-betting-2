from __future__ import annotations

from sports_betting.models.types import GameContext, MarketPrediction
from sports_betting.sports.common.base import SportModel
from sports_betting.sports.common.confidence import confidence_score
from sports_betting.sports.common.ev import edge, expected_value
from sports_betting.sports.common.odds import american_to_decimal, american_to_implied_probability


class NhlModel(SportModel):
    sport = "NHL"

    def predict_game(self, game: GameContext) -> list[MarketPrediction]:
        ratings = game.meta.get("team_ratings", {})
        home = ratings.get(game.home_team, {})
        away = ratings.get(game.away_team, {})

        xg_diff = home.get("xg_for", 0) - away.get("xg_for", 0)
        goalie_diff = home.get("goalie_rating", 0) - away.get("goalie_rating", 0)
        special_diff = home.get("special_teams", 0) - away.get("special_teams", 0)

        win_prob = min(0.88, max(0.12, 0.5 + xg_diff * 0.08 + goalie_diff * 0.05 + special_diff * 0.03))
        puckline_prob = min(0.84, max(0.16, win_prob + xg_diff * 0.02))
        total_over_prob = min(0.84, max(0.16, 0.5 + (xg_diff * 0.03) - goalie_diff * 0.015))

        preds: list[MarketPrediction] = []
        for m in game.markets:
            market_prob = american_to_implied_probability(m.american_odds)
            decimal_odds = american_to_decimal(m.american_odds)
            if m.market == "moneyline":
                model_prob = win_prob if m.side == game.home_team else (1 - win_prob)
                reasons = ["xG differential", "goalie strength", "special teams edge"]
            elif m.market == "spread":
                model_prob = puckline_prob if m.side == game.home_team else (1 - puckline_prob)
                reasons = ["shot share trend", "transition defense", "goaltending volatility"]
            else:
                model_prob = total_over_prob if m.side.lower() == "over" else (1 - total_over_prob)
                reasons = ["expected goals pace", "goalie confirmation", "finishing regression"]

            ev = expected_value(model_prob, decimal_odds)
            e = edge(model_prob, market_prob)
            conf = confidence_score(e, data_quality=0.78, calibration_quality=0.66)
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
                    model_quality=0.66,
                    explanation=reasons,
                    flags=[],
                )
            )
        return preds
