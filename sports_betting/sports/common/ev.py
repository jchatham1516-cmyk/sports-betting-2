from __future__ import annotations


def expected_value(probability: float, decimal_odds: float, stake: float = 1.0) -> float:
    win_profit = (decimal_odds - 1.0) * stake
    return probability * win_profit - (1 - probability) * stake


def edge(model_probability: float, market_probability: float) -> float:
    return model_probability - market_probability
