from __future__ import annotations

from sports.common.normalization import decimal_to_implied_prob


def expected_value(prob_win: float, decimal_odds: float, stake: float = 1.0) -> float:
    profit_if_win = stake * (decimal_odds - 1.0)
    loss_if_lose = stake
    return prob_win * profit_if_win - (1.0 - prob_win) * loss_if_lose


def market_probability(decimal_odds: float) -> float:
    return decimal_to_implied_prob(decimal_odds)


def edge(model_prob: float, market_prob: float) -> float:
    return model_prob - market_prob


def expected_value_two_way(model_prob: float, offered_decimal_odds: float) -> float:
    return expected_value(model_prob, offered_decimal_odds)
