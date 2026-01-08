"""Pydantic schemas for runs."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    sport: str
    game_date: str
    settings: Optional[dict[str, Any]] = None


class RunResponse(BaseModel):
    run_id: str = Field(..., alias="id")
    status: str
    predictions_count: int
    tracked_bets_count: int


class RunRead(BaseModel):
    id: str
    sport: str
    game_date: str
    status: str
    created_at: str
    settings_json: Optional[dict[str, Any]] = None
    log: Optional[str] = None
    artifacts_json: Optional[dict[str, Any]] = None


class RunStatusResponse(BaseModel):
    id: str
    status: str
    progress_percent: int = Field(..., alias="progress_percent")
    message: Optional[str] = None
    predictions_count: int
    tracked_bets_count: int
    error: Optional[str] = None
