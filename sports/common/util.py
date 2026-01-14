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
    """One-sided implied probability with sane guarding.

    Notes
    -----
    We intentionally avoid over-aggressive clipping here because downstream
    calibration layers need unbiased probabilities. A small numeric guard is
    kept to avoid divide-by-zero but otherwise the output reflects the raw
    price.
    """

    try:
        odds = float(odds)
    except Exception:
        return float("nan")

    if odds == 0 or math.isinf(odds) or math.isnan(odds):
        return float("nan")

    if odds < 0:
        p = (-odds) / ((-odds) + 100.0)
    else:
        p = 100.0 / (odds + 100.0)

    return float(max(min(p, 0.999999), 0.000001))


def implied_prob_from_american(odds: float) -> float:
    """Alias kept for clarity in odds math helpers."""

    return american_to_implied_prob(odds)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def remove_vig_two_way(p_home: float, p_away: float) -> Optional[tuple[float, float]]:
    """Return de-vigged probabilities for two-sided markets.

    The returned pair is normalized to sum to 1 when inputs are finite.
    Returns None if inputs are unusable.
    """

    try:
        ph = float(p_home)
        pa = float(p_away)
    except Exception:
        return None

    if any(math.isnan(x) or math.isinf(x) or x <= 0 for x in (ph, pa)):
        return None

    s = ph + pa
    if s <= 0:
        return None

    return float(ph / s), float(pa / s)


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


def normalize_result_label(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in {"W", "WIN"}:
        return "WIN"
    if text in {"L", "LOSS"}:
        return "LOSS"
    if text in {"P", "PUSH"}:
        return "PUSH"
    return text


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
