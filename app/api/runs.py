"""Run endpoints."""
from __future__ import annotations

from datetime import datetime, date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.core import runner
from app.db.models import Prediction, Run, TrackedBet
from app.db.session import get_session
from app.schemas.runs import RunCreate, RunRead, RunResponse

router = APIRouter(prefix="/runs", tags=["runs"])


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _row_value(row: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _infer_market(row: dict[str, Any]) -> str:
    text = str(row.get("primary_recommendation") or row.get("market") or "").lower()
    if "total" in text or "over" in text or "under" in text:
        return "total"
    if "spread" in text or "ats" in text:
        return "spread"
    return "ml"


@router.post("", response_model=RunResponse)
def create_run(payload: RunCreate, session: Session = Depends(get_session)) -> RunResponse:
    run = Run(
        sport=payload.sport,
        game_date=_parse_date(payload.game_date),
        status="running",
        settings_json=payload.settings or {},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    try:
        result = runner.run_model(payload.sport, payload.game_date, payload.settings)
        run.artifacts_json = {
            "predictions_path": result.get("predictions_path"),
            "tracked_bets_path": result.get("tracked_bets_path"),
        }
        run.log = result.get("log")
        run.status = "done"
        session.add(run)
        session.commit()

        predictions_rows = result.get("predictions_rows", [])
        prediction_models = []
        for row in predictions_rows:
            game_date = row.get("date") or payload.game_date
            if "/" in str(game_date):
                game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
            else:
                game_date_obj = _parse_date(str(game_date))
            home = _row_value(row, ["home", "home_team", "homeTeam"], default="")
            away = _row_value(row, ["away", "away_team", "awayTeam"], default="")
            pick = _row_value(row, ["pick", "primary_recommendation"], default="")
            price = row.get("price")
            if price is None:
                if "HOME" in str(pick).upper():
                    price = row.get("home_ml")
                elif "AWAY" in str(pick).upper():
                    price = row.get("away_ml")
            units = row.get("units") or row.get("bet_size")
            prediction_models.append(
                Prediction(
                    run_id=run.id,
                    game_date=game_date_obj,
                    home=str(home),
                    away=str(away),
                    market=_infer_market(row),
                    pick=str(pick),
                    price=float(price) if price is not None else None,
                    units=float(units) if units is not None else None,
                    raw_row=row,
                )
            )
        if prediction_models:
            session.add_all(prediction_models)
            session.commit()

        tracked_rows = result.get("tracked_bets_rows", [])
        tracked_models = []
        for row in tracked_rows:
            bet_date = row.get("bet_date") or payload.game_date
            if "/" in str(bet_date):
                bet_date_obj = datetime.strptime(str(bet_date), "%m/%d/%Y").date()
            else:
                bet_date_obj = _parse_date(str(bet_date))
            tracked_models.append(
                TrackedBet(
                    run_id=run.id,
                    bet_date=bet_date_obj,
                    sport=row.get("sport", payload.sport),
                    market=row.get("market", "ml"),
                    home=str(row.get("home", "")),
                    away=str(row.get("away", "")),
                    pick=str(row.get("pick", "")),
                    price=float(row.get("price")) if row.get("price") is not None else None,
                    units=float(row.get("units")) if row.get("units") is not None else None,
                    result=str(row.get("result", "PENDING")),
                )
            )
        if tracked_models:
            session.add_all(tracked_models)
            session.commit()

        return RunResponse(
            id=run.id,
            status=run.status,
            predictions_count=len(prediction_models),
            tracked_bets_count=len(tracked_models),
        )
    except Exception as exc:  # noqa: BLE001
        run.status = "error"
        run.log = str(exc)
        session.add(run)
        session.commit()
        raise HTTPException(
            status_code=500,
            detail={"message": "Run failed", "run_id": run.id, "error": str(exc)},
        ) from exc


@router.get("/{run_id}", response_model=RunRead)
def get_run(run_id: str, session: Session = Depends(get_session)) -> RunRead:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunRead(
        id=run.id,
        sport=run.sport,
        game_date=run.game_date.isoformat(),
        status=run.status,
        created_at=run.created_at.isoformat(),
        settings_json=run.settings_json,
        log=run.log,
        artifacts_json=run.artifacts_json,
    )


@router.get("/{run_id}/predictions")
def get_predictions(run_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    statement = select(Prediction).where(Prediction.run_id == run_id)
    predictions = session.exec(statement).all()
    return [prediction.raw_row for prediction in predictions]


@router.get("/{run_id}/download/predictions.csv")
def download_predictions(run_id: str, session: Session = Depends(get_session)) -> FileResponse:
    run = session.get(Run, run_id)
    if not run or not run.artifacts_json or not run.artifacts_json.get("predictions_path"):
        raise HTTPException(status_code=404, detail="Predictions file not found")
    path = run.artifacts_json["predictions_path"]
    return FileResponse(path, filename="predictions.csv")


@router.get("/{run_id}/download/tracked_bets.csv")
def download_tracked_bets(run_id: str, session: Session = Depends(get_session)) -> FileResponse:
    run = session.get(Run, run_id)
    if not run or not run.artifacts_json or not run.artifacts_json.get("tracked_bets_path"):
        raise HTTPException(status_code=404, detail="Tracked bets file not found")
    path = run.artifacts_json["tracked_bets_path"]
    return FileResponse(path, filename="tracked_bets.csv")
