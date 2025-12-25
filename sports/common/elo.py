# sports/common/elo.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Any

DEFAULT_ELO = 1500.0


def elo_win_prob(elo_home: float, elo_away: float, home_adv: float = 65.0) -> float:
    # Classic Elo logistic
    diff = (elo_home + home_adv) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

# sports/common/elo.py

import math
from typing import Tuple

def _mov_multiplier(home_score: float, away_score: float, elo_diff: float) -> float:
    """
    Common Elo MOV multiplier (FiveThirtyEight-style-ish).
    Elo diff is from home perspective (home+home_adv - away).
    """
    mov = abs(float(home_score) - float(away_score))
    if mov <= 1e-9:
        return 1.0

    # dampen blowouts, boost close games a bit less
    mult = (mov + 3.0) ** 0.8 / 7.5

    # bigger multiplier when upset vs expectation
    # avoid division by zero
    denom = 1.0 + 0.006 * abs(float(elo_diff))
    return mult * (2.2 / denom)

def elo_update(
    elo_home: float,
    elo_away: float,
    home_score: float,
    away_score: float,
    *,
    k: float = 20.0,
    home_adv: float = 65.0,
) -> Tuple[float, float]:
    from .elo import elo_win_prob  # or inline if circular

    p_home = elo_win_prob(elo_home, elo_away, home_adv=home_adv)

    if home_score > away_score:
        s_home = 1.0
    elif home_score < away_score:
        s_home = 0.0
    else:
        s_home = 0.5

    elo_diff = (elo_home + home_adv) - elo_away
    m = _mov_multiplier(home_score, away_score, elo_diff)

    k_eff = float(k) * float(m)

    new_home = elo_home + k_eff * (s_home - p_home)
    new_away = elo_away + k_eff * ((1.0 - s_home) - (1.0 - p_home))
    return new_home, new_away


@dataclass
class EloState:
    ratings: Dict[str, float]
    processed_games: Dict[str, int]  # game_key -> 1

    @classmethod
    def load(cls, path: str) -> "EloState":
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return cls(
                ratings={k: float(v) for k, v in (obj.get("ratings") or {}).items()},
                processed_games={k: int(v) for k, v in (obj.get("processed_games") or {}).items()},
            )
        return cls(ratings={}, processed_games={})

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"ratings": self.ratings, "processed_games": self.processed_games},
                f,
                indent=2,
                sort_keys=True,
            )

    def get(self, team: str) -> float:
        return float(self.ratings.get(team, DEFAULT_ELO))

    def set(self, team: str, rating: float) -> None:
        self.ratings[team] = float(rating)

    def mark_processed(self, game_key: str) -> None:
        self.processed_games[game_key] = 1

    def is_processed(self, game_key: str) -> bool:
        return game_key in self.processed_games
