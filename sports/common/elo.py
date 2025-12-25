from __future__ import annotations

from typing import Tuple, Optional
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
    Tuned to be stable across NBA / NFL / NHL while allowing sport-specific scaling.

    Notes:
    - NBA: margins can be huge; we dampen blowouts more.
    - NFL: medium; touchdowns matter; still dampen.
    - NHL: margins are small; 1-goal wins are common; MOV shouldn't overreact.
    """
    mov = abs(float(home_score) - float(away_score))
    if mov <= 0.0:
        return 1.0

    sport = (sport or "generic").lower().strip()

    # --- Sport-aware parameters ---
    # These are deliberately conservative; they improve stability without making updates wild.
    if sport == "nba":
        # Big scores -> big margins; dampen blowouts more
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
        # Small scoring; keep MOV effect mild
        add = 1.0
        power = 0.95
        scale = 3.0
        denom_k = 0.0045
        mult_top = 1.60
    else:
        # Your original-ish behavior
        add = 3.0
        power = 0.80
        scale = 7.5
        denom_k = 0.006
        mult_top = 2.20

    # Base MOV factor
    base = (mov + add) ** power / scale

    # Reduce update when a strong team beats a weak team (already expected)
    denom = 1.0 + denom_k * abs(float(elo_diff))

    mult = base * (mult_top / denom)

    # Safety clamp: never let MOV explode or go to zero
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

    If you want sport-aware MOV scaling, pass sport="nba" | "nfl" | "nhl".
    Leaving it as default keeps behavior very close to what you already had.
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

    # Standard Elo update
    new_home = elo_home + k_eff * (s_home - p_home)
    new_away = elo_away + k_eff * ((1.0 - s_home) - (1.0 - p_home))  # == elo_away + k_eff*(p_home - s_home)

    return float(new_home), float(new_away)
