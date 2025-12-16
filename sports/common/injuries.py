# sports/common/injuries.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Injury:
    player: str
    team: str
    status: str               # e.g. OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, DAY-TO-DAY
    detail: Optional[str] = None
    source: Optional[str] = None
    updated_at: Optional[str] = None

# Normalize statuses across sports
def normalize_status(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in {"out", "inactive", "ruled out"}:
        return "OUT"
    if s in {"doubtful"}:
        return "DOUBTFUL"
    if s in {"questionable", "q"}:
        return "QUESTIONABLE"
    if s in {"probable", "p"}:
        return "PROBABLE"
    if s in {"day-to-day", "dtd"}:
        return "DAY-TO-DAY"
    if s in {"ir", "injured reserve"}:
        return "IR"
    return raw.upper() if raw else "UNKNOWN"
