"""Tracked bets endpoints."""
from __future__ import annotations

from datetime import datetime, date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from app.db.models import TrackedBet
from app.db.session import get_session
from app.schemas.bets import SettleResponse, TrackedBetRead

router = APIRouter(prefix="/bets", tags=["bets"])


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


@router.get("", response_model=list[TrackedBetRead])
def list_bets(
    date: str = Query(..., description="YYYY-MM-DD"),
    sport: str | None = None,
    session: Session = Depends(get_session),
) -> list[TrackedBetRead]:
    bet_date = _parse_date(date)
    statement = select(TrackedBet).where(TrackedBet.bet_date == bet_date)
    if sport:
        statement = statement.where(TrackedBet.sport == sport)
    bets = session.exec(statement).all()
    return [
        TrackedBetRead(
            id=bet.id,
            run_id=bet.run_id,
            bet_date=bet.bet_date.isoformat(),
            sport=bet.sport,
            market=bet.market,
            home=bet.home,
            away=bet.away,
            pick=bet.pick,
            price=bet.price,
            units=bet.units,
            result=bet.result,
            settled_at=bet.settled_at.isoformat() if bet.settled_at else None,
        )
        for bet in bets
    ]


@router.post("/settle", response_model=SettleResponse)
def settle_bets(
    date: str = Query(..., description="YYYY-MM-DD"),
    sport: str | None = None,
    session: Session = Depends(get_session),
) -> SettleResponse:
    bet_date = _parse_date(date)
    statement = select(TrackedBet).where(TrackedBet.bet_date == bet_date)
    if sport:
        statement = statement.where(TrackedBet.sport == sport)
    bets = session.exec(statement).all()
    if not bets:
        raise HTTPException(status_code=404, detail="No tracked bets found")

    # TODO: wire in scoring utilities to grade bets.
    pending_count = 0
    for bet in bets:
        bet.result = "PENDING"
        pending_count += 1
        session.add(bet)
    session.commit()

    return SettleResponse(
        message="Settlement not implemented yet; bets left as PENDING.",
        pending_count=pending_count,
        updated_count=len(bets),
    )
