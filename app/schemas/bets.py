"""Pydantic schemas for tracked bets."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TrackedBetRead(BaseModel):
    id: str
    run_id: str
    bet_date: str
    sport: str
    market: str
    home: str
    away: str
    pick: str
    price: Optional[float] = None
    units: Optional[float] = None
    result: str
    settled_at: Optional[str] = None


class SettleResponse(BaseModel):
    message: str
    pending_count: int
    updated_count: int
