from __future__ import annotations
from typing import Tuple
import math


def elo_win_prob(
    elo_home: float,
    elo_away: float,
    *,
    home_adv: float = 0.0,
) -> float:
    """
    Standard Elo logistic win probability.
    """
    diff = (elo_home + home_adv) - elo_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _mov_multiplier(
    home_score: float,
    away_score: float,
    elo_diff: float,
) -> float:
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

    elo_diff = (elo_home + home_adv) - elo_away
    k_eff = float(k)

    if use_mov:
        k_eff *= _mov_multiplier(home_score, away_score, elo_diff)

    new_home = elo_home + k_eff * (s_home - p_home)
    new_away = elo_away + k_eff * ((1.0 - s_home) - (1.0 - p_home))
    return new_home, new_away
