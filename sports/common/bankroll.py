import math
from typing import Optional

from sports.common.util import american_to_decimal, safe_float


DEFAULT_BANKROLL = 250.0
UNIT_PCT = 0.04  # 4% of bankroll per unit


def play_pass_rule(
    row,
    *,
    require_pick: bool = True,
    require_value_tier: str = "HIGH VALUE",
    min_confidence: str = "MEDIUM",  # LOW/MEDIUM/HIGH
    max_abs_moneyline: Optional[int] = 400,
) -> str:
    primary = str(row.get("primary_recommendation", ""))
    value_tier = str(row.get("value_tier", ""))
    conf = str(row.get("confidence", ""))

    if require_pick and ("PICK" not in primary):
        return "PASS"
    if require_value_tier and (value_tier != require_value_tier):
        return "PASS"

    conf_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if conf_rank.get(conf, 0) < conf_rank.get(min_confidence, 1):
        return "PASS"

    if max_abs_moneyline is not None and "ML" in primary:
        hm = safe_float(row.get("home_ml"))
        am = safe_float(row.get("away_ml"))
        if "HOME ML" in primary and hm is not None and not math.isnan(hm) and abs(hm) > max_abs_moneyline:
            return "PASS"
        if "AWAY ML" in primary and am is not None and not math.isnan(am) and abs(am) > max_abs_moneyline:
            return "PASS"

    return "PLAY"


def kelly_fraction(p: float, odds_american: float) -> float:
    d = american_to_decimal(odds_american)
    b = d - 1.0
    q = 1.0 - p
    frac = (b * p - q) / b
    return max(frac, 0.0)


def bet_size_flat(bankroll: float, flat_pct: float) -> float:
    return max(bankroll * float(flat_pct), 0.0)


def bet_size_kelly_ml(
    bankroll: float,
    p: float,
    odds_american: float,
    *,
    kelly_mult: float = 0.5,
    max_pct: float = 0.03,
) -> float:
    f = kelly_fraction(float(p), float(odds_american))
    f_adj = min(f * float(kelly_mult), float(max_pct))
    return max(bankroll * f_adj, 0.0)


def compute_bet_size(
    row,
    bankroll: float,
    *,
    sizing_mode: str = "flat",  # "flat" or "kelly"
    flat_pct: float = UNIT_PCT,
    kelly_mult: float = 0.5,
    kelly_max_pct: float = 0.03,
) -> float:
    if str(row.get("play_pass")) != "PLAY":
        return 0.0

    primary = str(row.get("primary_recommendation", ""))

    if sizing_mode == "flat":
        return bet_size_flat(bankroll, flat_pct)

    # Kelly: ML only
    if "HOME ML" in primary:
        ml = safe_float(row.get("home_ml"))
        if ml is None or (isinstance(ml, float) and math.isnan(ml)):
            return 0.0
        p = float(row.get("model_home_prob"))
        return bet_size_kelly_ml(bankroll, p, ml, kelly_mult=kelly_mult, max_pct=kelly_max_pct)

    if "AWAY ML" in primary:
        ml = safe_float(row.get("away_ml"))
        if ml is None or (isinstance(ml, float) and math.isnan(ml)):
            return 0.0
        p = 1.0 - float(row.get("model_home_prob"))
        return bet_size_kelly_ml(bankroll, p, ml, kelly_mult=kelly_mult, max_pct=kelly_max_pct)

    # ATS: fallback to flat
    return bet_size_flat(bankroll, flat_pct)

