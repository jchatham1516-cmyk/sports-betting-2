from __future__ import annotations

from dataclasses import dataclass

from sports.common.elo import EloEngine
from sports.common.normalization import bounded_probability


@dataclass
class NflModelConfig:
    injury_cap: float = 0.08
    injury_shrink: float = 0.4
    weather_cap: float = 0.03
    weather_shrink: float = 0.6
    uncertainty_penalty_missing_inputs: float = 0.01


class NflModel:
    def __init__(self, elo: EloEngine, config: NflModelConfig | None = None) -> None:
        self.elo = elo
        self.config = config or NflModelConfig()

    def predict_home_win_prob(
        self,
        home: str,
        away: str,
        injury_delta: float | None = None,
        qb_delta: float | None = None,
        weather_delta: float | None = None,
    ) -> tuple[float, list[str]]:
        flags: list[str] = []
        base = self.elo.expected_home_win_prob(home, away)
        adjustment = 0.0
        missing_count = 0

        for value, cap, shrink, label in [
            (injury_delta, self.config.injury_cap, self.config.injury_shrink, "injury_data_missing"),
            (qb_delta, self.config.injury_cap, self.config.injury_shrink, "qb_data_missing"),
            (weather_delta, self.config.weather_cap, self.config.weather_shrink, "weather_data_missing"),
        ]:
            if value is None:
                missing_count += 1
                flags.append(label)
                continue
            shrunk = value * shrink
            adjustment += max(-cap, min(cap, shrunk))

        if missing_count:
            adjustment -= self.config.uncertainty_penalty_missing_inputs * missing_count

        return bounded_probability(base + adjustment), flags
