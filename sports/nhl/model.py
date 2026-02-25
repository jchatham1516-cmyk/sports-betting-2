from __future__ import annotations

from dataclasses import dataclass

from sports.common.elo import EloEngine
from sports.common.normalization import bounded_probability


@dataclass
class NhlModelConfig:
    goalie_cap: float = 0.06
    goalie_shrink: float = 0.5
    uncertainty_penalty_missing_goalies: float = 0.015


class NhlModel:
    def __init__(self, elo: EloEngine, config: NhlModelConfig | None = None) -> None:
        self.elo = elo
        self.config = config or NhlModelConfig()

    def predict_home_win_prob(self, home: str, away: str, goalie_delta: float | None = None) -> tuple[float, list[str]]:
        flags: list[str] = []
        base = self.elo.expected_home_win_prob(home, away)
        if goalie_delta is None:
            flags.append("goalie_data_missing_uncertainty_penalty")
            return bounded_probability(base - self.config.uncertainty_penalty_missing_goalies), flags

        shrunk = goalie_delta * self.config.goalie_shrink
        capped = max(-self.config.goalie_cap, min(self.config.goalie_cap, shrunk))
        return bounded_probability(base + capped), flags
