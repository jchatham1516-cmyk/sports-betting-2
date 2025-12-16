import math

from sports.common.util import american_to_decimal


def settle_ml(side: str, home_score: int, away_score: int, home_ml: float, away_ml: float, stake: float) -> float:
    if stake <= 0:
        return 0.0

    side = side.upper()
    if side == "HOME":
        won = home_score > away_score
        odds = home_ml
    else:
        won = away_score > home_score
        odds = away_ml

    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return 0.0

    if won:
        dec = american_to_decimal(odds)
        return stake * (dec - 1.0)
    return -stake


def settle_ats(
    side: str,
    home_score: int,
    away_score: int,
    home_spread: float,
    stake: float,
    price_american: float = -110.0,
) -> float:
    if stake <= 0:
        return 0.0
    if home_spread is None or (isinstance(home_spread, float) and math.isnan(home_spread)):
        return 0.0

    side = side.upper()
    if side == "HOME":
        adj_home = home_score + float(home_spread)
        adj_away = away_score
    else:
        adj_home = home_score
        adj_away = away_score - float(home_spread)

    if abs(adj_home - adj_away) < 1e-9:
        return 0.0  # push

    won = adj_home > adj_away if side == "HOME" else adj_away > adj_home

    if won:
        dec = american_to_decimal(price_american)
        return stake * (dec - 1.0)
    return -stake

