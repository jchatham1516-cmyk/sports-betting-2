from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_ELO = 1500.0


@dataclass
class EloEngine:
    k_factor: float = 20.0
    home_advantage: float = 60.0
    ratings: dict[str, float] = field(default_factory=dict)

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, DEFAULT_ELO)

    def expected_home_win_prob(self, home_team: str, away_team: str) -> float:
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)
        return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))

    def update(self, home_team: str, away_team: str, home_win: bool) -> None:
        expected = self.expected_home_win_prob(home_team, away_team)
        result = 1.0 if home_win else 0.0
        delta = self.k_factor * (result - expected)
        self.ratings[home_team] = self.get_rating(home_team) + delta
        self.ratings[away_team] = self.get_rating(away_team) - delta
