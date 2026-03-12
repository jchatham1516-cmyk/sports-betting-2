from __future__ import annotations


def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    return (1 + (odds / 100)) if odds > 0 else (1 + (100 / abs(odds)))


def decimal_to_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    return 1.0 / decimal_odds


def american_to_implied_probability(odds: int) -> float:
    return decimal_to_implied_probability(american_to_decimal(odds))


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Invalid implied probabilities")
    return prob_a / total, prob_b / total
