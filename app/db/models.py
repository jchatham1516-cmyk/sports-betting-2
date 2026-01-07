"""Database models."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column, JSON, Text
from sqlmodel import Field, SQLModel


class Run(SQLModel, table=True):
    """Represents a model run."""

    __tablename__ = "runs"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    sport: str
    game_date: date
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str
    settings_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    log: Optional[str] = Field(default=None, sa_column=Column(Text))
    artifacts_json: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class Prediction(SQLModel, table=True):
    """Prediction record for a run."""

    __tablename__ = "predictions"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    run_id: str = Field(foreign_key="runs.id")
    game_date: date
    home: str
    away: str
    market: Optional[str] = None
    pick: Optional[str] = None
    price: Optional[float] = None
    units: Optional[float] = None
    raw_row: dict = Field(sa_column=Column(JSON))


class TrackedBet(SQLModel, table=True):
    """Tracked bet for grading."""

    __tablename__ = "tracked_bets"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    run_id: str = Field(foreign_key="runs.id")
    bet_date: date
    sport: str
    market: str
    home: str
    away: str
    pick: str
    price: Optional[float] = None
    units: Optional[float] = None
    result: str = "PENDING"
    settled_at: Optional[datetime] = None
