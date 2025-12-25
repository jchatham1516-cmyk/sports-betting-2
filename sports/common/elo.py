# sports/common/elo.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

# ----------------------------
# Elo math
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


def _mov_multiplier(home_score: float, away_score: float, elo_diff: float) -> float:
    """
    Margin-of-victory multiplier.
    Works across NBA / NFL / NHL.
    """
    mov = abs(float(home_score) - float(away_score))
    if mov <= 0.0:
        return 1.0

    # Dampens blowouts, rewards informative wins
    base = (mov + 3.0) ** 0.8 / 7.5
    denom = 1.0 + 0.006 * abs(float(elo_diff))
    return base * (2.2 / denom)


def elo_update(
    elo_home: float,
    elo_away: float,
    home_score: float,
    away_score: float,
    *,
    k: float = 20.0,
    home_adv: float = 65.0,
    use_mov: bool = True,
) -> Tuple[float, float]:
    """
    Elo update with optional MOV scaling.
    """
    p_home = elo_win_prob(elo_home, elo_away, home_adv=home_adv)

    if home_score > away_score:
        s_home = 1.0
    elif home_score < away_score:
        s_home = 0.0
    else:
        s_home = 0.5

    elo_diff = (float(elo_home) + float(home_adv)) - float(elo_away)
    k_eff = float(k)

    if use_mov:
        k_eff *= _mov_multiplier(home_score, away_score, elo_diff)

    new_home = float(elo_home) + k_eff * (s_home - p_home)
    new_away = float(elo_away) + k_eff * ((1.0 - s_home) - (1.0 - p_home))
    return new_home, new_away


# ----------------------------
# Elo state persistence
# ----------------------------
@dataclass
class EloState:
    """
    Stores team Elo ratings and a set of processed game keys to avoid double-updates.
    """
    ratings: Dict[str, float] = field(default_factory=dict)
    processed: Set[str] = field(default_factory=set)
    default_elo: float = 1500.0

    def get(self, team: str) -> float:
        return float(self.ratings.get(team, self.default_elo))

    def set(self, team: str, elo: float) -> None:
        self.ratings[str(team)] = float(elo)

    def is_processed(self, game_key: str) -> bool:
        return str(game_key) in self.processed

    def mark_processed(self, game_key: str) -> None:
        self.processed.add(str(game_key))

    @classmethod
    def load(cls, path: str) -> "EloState":
        if not path or not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            ratings = d.get("ratings") or {}
            processed = set(d.get("processed") or [])
            default_elo = float(d.get("default_elo", 1500.0))
            # coerce ratings floats
            out_r = {}
            for k, v in ratings.items():
                try:
                    out_r[str(k)] = float(v)
                except Exception:
                    continue
            return cls(ratings=out_r, processed=processed, default_elo=default_elo)
        except Exception:
            return cls()

    def save(self, path: str) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "ratings": self.ratings,
            "processed": sorted(list(self.processed)),
            "default_elo": self.default_elo,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
