from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


# ----------------------------
# Elo State (persisted to disk)
# ----------------------------
@dataclass
class EloState:
    """
    Persistent Elo ratings + processed-game ledger.

    Stored JSON schema:
      {
        "ratings": { "Team A": 1512.3, ... },
        "processed": ["game_key_1", "game_key_2", ...],
        "meta": { ...optional... }
      }
    """
    ratings: Dict[str, float] = field(default_factory=dict)
    processed: Set[str] = field(default_factory=set)
    default_rating: float = 1500.0

    def get(self, team: str) -> float:
        if not team:
            return float(self.default_rating)
        return float(self.ratings.get(team, self.default_rating))

    def set(self, team: str, rating: float) -> None:
        if not team:
            return
        try:
            self.ratings[str(team)] = float(rating)
        except Exception:
            self.ratings[str(team)] = float(self.default_rating)

    def is_processed(self, game_key: str) -> bool:
        if not game_key:
            return False
        return str(game_key) in self.processed

    def mark_processed(self, game_key: str) -> None:
        if not game_key:
            return
        self.processed.add(str(game_key))

    @classmethod
    def load(cls, path: str, *, default_rating: float = 1500.0) -> "EloState":
        try:
            if not path or not os.path.exists(path):
                return cls(ratings={}, processed=set(), default_rating=float(default_rating))
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            ratings_in = data.get("ratings") or {}
            processed_in = data.get("processed") or []

            ratings = {}
            for k, v in ratings_in.items():
                try:
                    ratings[str(k)] = float(v)
                except Exception:
                    continue

            processed = set()
            for g in processed_in:
                if g:
                    processed.add(str(g))

            return cls(ratings=ratings, processed=processed, default_rating=float(default_rating))
        except Exception:
            # fail-safe: never crash because state is corrupted
            return cls(ratings={}, processed=set(), default_rating=float(default_rating))

    def save(self, path: str) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "ratings": {k: float(v) for k, v in (self.ratings or {}).items()},
            "processed": sorted(list(self.processed or set())),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


# ----------------------------
# Elo probability + MOV update
# ----------------------------
def elo_win_prob(
    elo_home: float,
    elo_away: float,
    *,
    home_adv: float = 0.0,
) -> float:
    """
    Standard Elo logistic win probability.
    """
    diff = (float(elo_home) + float(home_adv)) - float(elo_away)
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _mov_multiplier(
    home_score: float,
    away_score: float,
    elo_diff: float,
    *,
    sport: str = "generic",
) -> float:
    """
    Margin-of-victory multiplier (MOV).
    Sport-aware but conservative; improves stability.

    NBA: dampen blowouts more
    NFL: moderate
    NHL: mild (small scoring)
    """
    mov = abs(float(home_score) - float(away_score))
    if mov <= 0.0:
        return 1.0

    sport = (sport or "generic").lower().strip()

    if sport == "nba":
        add = 3.0
        power = 0.78
        scale = 8.5
        denom_k = 0.007
        mult_top = 2.05
    elif sport == "nfl":
        add = 3.0
        power = 0.84
        scale = 7.2
        denom_k = 0.006
        mult_top = 2.20
    elif sport == "nhl":
        add = 1.0
        power = 0.95
        scale = 3.0
        denom_k = 0.0045
        mult_top = 1.60
    else:
        add = 3.0
        power = 0.80
        scale = 7.5
        denom_k = 0.006
        mult_top = 2.20

    base = (mov + add) ** power / scale
    denom = 1.0 + denom_k * abs(float(elo_diff))
    mult = base * (mult_top / denom)

    # safety clamp
    return float(max(0.75, min(3.0, mult)))


def elo_update(
    elo_home: float,
    elo_away: float,
    home_score: float,
    away_score: float,
    *,
    k: float = 20.0,
    home_adv: float = 65.0,
    use_mov: bool = True,
    sport: str = "generic",
) -> Tuple[float, float]:
    """
    Elo update with optional MOV scaling.
    """
    elo_home = float(elo_home)
    elo_away = float(elo_away)
    home_score = float(home_score)
    away_score = float(away_score)

    p_home = float(elo_win_prob(elo_home, elo_away, home_adv=float(home_adv)))

    if home_score > away_score:
        s_home = 1.0
    elif home_score < away_score:
        s_home = 0.0
    else:
        s_home = 0.5

    elo_diff = (elo_home + float(home_adv)) - elo_away
    k_eff = float(k)

    if use_mov:
        k_eff *= _mov_multiplier(home_score, away_score, elo_diff, sport=sport)

    new_home = elo_home + k_eff * (s_home - p_home)
    new_away = elo_away + k_eff * ((1.0 - s_home) - (1.0 - p_home))

    return float(new_home), float(new_away)
