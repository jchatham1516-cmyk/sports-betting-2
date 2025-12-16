import math
import re
from typing import Optional


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ").strip())


def normalize_team_name(s: str) -> str:
    s = normalize_spaces(s).lower()
    s = s.replace(".", "")
    return s


def american_to_implied_prob(odds) -> float:
    odds = float(odds)
    if odds < 0:
        p = (-odds) / ((-odds) + 100.0)
    else:
        p = 100.0 / (odds + 100.0)
    return max(min(p, 0.9999), 0.0001)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

