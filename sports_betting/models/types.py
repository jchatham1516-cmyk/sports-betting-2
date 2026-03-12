from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class MarketOdds:
    market: str
    side: str
    line: float | None
    american_odds: int
    decimal_odds: float
    sportsbook: str


@dataclass
class GameContext:
    event_id: str
    date: datetime
    sport: str
    home_team: str
    away_team: str
    markets: list[MarketOdds] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketPrediction:
    event_id: str
    date: str
    sport: str
    game: str
    market: str
    side: str
    line: float | None
    sportsbook_odds: int
    model_probability: float
    market_probability: float
    edge: float
    expected_value: float
    confidence: float
    model_quality: float
    explanation: list[str]
    flags: list[str]
    recommended_units: float = 0.0
    decision: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_summary"] = "; ".join(self.explanation)
        payload["flags"] = "; ".join(self.flags)
        return payload
