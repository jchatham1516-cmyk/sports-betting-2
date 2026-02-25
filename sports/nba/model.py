from __future__ import annotations

from dataclasses import dataclass

from sports.common.elo import EloEngine
from sports.common.normalization import bounded_probability


@dataclass
class NbaModelConfig:
    rest_days_cap: float = 3.0
    rest_weight: float = 0.02


class NbaModel:
    def __init__(self, elo: EloEngine, config: NbaModelConfig | None = None) -> None:
        self.elo = elo
        self.config = config or NbaModelConfig()

    def predict_home_win_prob(self, home: str, away: str, rest_days_home: float = 1.0, rest_days_away: float = 1.0) -> float:
        base = self.elo.expected_home_win_prob(home, away)
        rest_diff = max(-self.config.rest_days_cap, min(self.config.rest_days_cap, rest_days_home - rest_days_away))
        adjusted = base + (rest_diff * self.config.rest_weight)
        return bounded_probability(adjusted)
