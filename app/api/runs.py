"""Run endpoints."""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.core import runner
from app.core.run_store import store
from app.db.models import Prediction, Run, TrackedBet
from app.db.session import get_engine, get_session
from app.schemas.runs import RunCreate, RunResponse, RunStatusResponse

router = APIRouter(prefix="/runs", tags=["runs"])


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _validate_game_date(date_str: str) -> date:
    try:
        return _parse_date(date_str)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": "game_date must be YYYY-MM-DD"},
        ) from exc


def _run_key(sport: str, game_date: str, settings: dict[str, Any] | None) -> str:
    return f"{sport}:{game_date}"


def _persist_run_results(
    run_id: str,
    payload: RunCreate,
    result: dict[str, Any],
) -> tuple[int, int]:
    engine = get_engine()
    prediction_models = []
    tracked_models = []
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            raise RuntimeError("Run not found in database")

        run.artifacts_json = {
            "predictions_path": result.get("predictions_path"),
            "tracked_bets_path": result.get("tracked_bets_path"),
        }
        run.log = result.get("log")
        run.status = "done"
        session.add(run)
        session.commit()

        predictions_rows = result.get("predictions_rows", [])
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
                    run_id=run_id,
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
        for row in tracked_rows:
            bet_date = row.get("bet_date") or payload.game_date
            if "/" in str(bet_date):
                bet_date_obj = datetime.strptime(str(bet_date), "%m/%d/%Y").date()
            else:
                bet_date_obj = _parse_date(str(bet_date))
            tracked_models.append(
                TrackedBet(
                    run_id=run_id,
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

    return (len(prediction_models), len(tracked_models))


def _run_worker(run_id: str, payload: RunCreate) -> None:
    store.update_run(run_id, status="running", progress=10, message="Starting model run...")
    with Session(get_engine()) as session:
        run = session.get(Run, run_id)
        if run:
            run.status = "running"
            session.add(run)
            session.commit()
    if not os.getenv("BALLDONTLIE_API_KEY"):
        store.update_run(
            run_id,
            status="failed",
            progress=10,
            message="Run failed",
            error="Missing BALLDONTLIE_API_KEY",
        )
        with Session(get_engine()) as session:
            run = session.get(Run, run_id)
            if run:
                run.status = "failed"
                run.log = "Missing BALLDONTLIE_API_KEY"
                session.add(run)
                session.commit()
        return
    if not os.getenv("ODDS_API_KEY"):
        store.update_run(
            run_id,
            status="failed",
            progress=10,
            message="Run failed",
            error="Missing ODDS_API_KEY",
        )
        with Session(get_engine()) as session:
            run = session.get(Run, run_id)
            if run:
                run.status = "failed"
                run.log = "Missing ODDS_API_KEY"
                session.add(run)
                session.commit()
        return

    start_time = time.monotonic()
    milestones = [
        (30, "Loading data..."),
        (60, "Computing predictions..."),
        (85, "Writing outputs..."),
    ]
    next_milestone = 0
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(runner.run_model, payload.sport, payload.game_date, payload.settings)
            while not future.done():
                elapsed = time.monotonic() - start_time
                estimated = min(85, int((elapsed / 330) * 85))
                if next_milestone < len(milestones):
                    target_progress, message = milestones[next_milestone]
                    if estimated >= target_progress:
                        store.update_run(run_id, progress=target_progress, message=message)
                        next_milestone += 1
                    else:
                        store.update_run(run_id, progress=max(10, estimated), message="Running model...")
                else:
                    store.update_run(run_id, progress=max(10, estimated), message="Running model...")
                time.sleep(5)
            result = future.result()
        _persist_run_results(run_id, payload, result)
    except Exception as exc:  # noqa: BLE001
        store.update_run(
            run_id,
            status="failed",
            progress=90,
            message="Run failed",
            error=str(exc),
        )
        with Session(get_engine()) as session:
            run = session.get(Run, run_id)
            if run:
                run.status = "failed"
                run.log = str(exc)
                session.add(run)
                session.commit()
        return

    store.update_run(
        run_id,
        status="done",
        progress=100,
        message="Run complete",
    )
    with Session(get_engine()) as session:
        run = session.get(Run, run_id)
        if run:
            run.status = "done"
            session.add(run)
            session.commit()


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
    _validate_game_date(payload.game_date)
    run_key = _run_key(payload.sport, payload.game_date, payload.settings)
    existing = store.find_active_by_key(run_key)
    if existing:
        return RunResponse(
            id=existing.id,
            status=existing.status,
            predictions_count=0,
            tracked_bets_count=0,
        )
    existing_db = session.exec(
        select(Run).where(
            Run.sport == payload.sport,
            Run.game_date == _parse_date(payload.game_date),
            Run.status.in_(["queued", "running"]),
        )
    ).first()
    if existing_db:
        return RunResponse(
            id=existing_db.id,
            status=existing_db.status,
            predictions_count=0,
            tracked_bets_count=0,
        )

    run_status = store.create_run(
        status="queued",
        progress=0,
        message="Queued",
        key=run_key,
    )
    run = Run(
        id=run_status.id,
        sport=payload.sport,
        game_date=_parse_date(payload.game_date),
        status="queued",
        settings_json=payload.settings or {},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    thread = threading.Thread(
        target=_run_worker,
        args=(run.id, payload),
        daemon=True,
    )
    thread.start()

    return RunResponse(
        id=run.id,
        status="queued",
        predictions_count=0,
        tracked_bets_count=0,
    )


def _normalize_status(raw_status: str) -> str:
    if raw_status == "error":
        return "failed"
    if raw_status in {"queued", "running", "done", "failed"}:
        return raw_status
    return raw_status


def _count_run_items(session: Session, run_id: str) -> tuple[int, int]:
    predictions = session.exec(select(Prediction).where(Prediction.run_id == run_id)).all()
    tracked = session.exec(select(TrackedBet).where(TrackedBet.run_id == run_id)).all()
    return (len(predictions), len(tracked))


def _build_run_status_response(run: Run, session: Session) -> RunStatusResponse:
    store_run = store.get_run(run.id)
    status = _normalize_status(store_run.status) if store_run else _normalize_status(run.status)
    if store_run:
        progress = store_run.progress
        message = store_run.message
        error = store_run.error
    else:
        progress = 100 if status == "done" else 0
        if status == "done":
            message = "Run complete"
        elif status == "failed":
            message = "Run failed"
        elif status == "queued":
            message = "Queued"
        else:
            message = "Running"
        error = run.log if status == "failed" else None

    predictions_count = 0
    tracked_bets_count = 0
    if status == "done":
        predictions_count, tracked_bets_count = _count_run_items(session, run.id)

    return RunStatusResponse(
        id=run.id,
        status=status,
        progress_percent=progress,
        message=message,
        predictions_count=predictions_count,
        tracked_bets_count=tracked_bets_count,
        error=error,
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
def get_run(run_id: str, session: Session = Depends(get_session)) -> RunStatusResponse:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _build_run_status_response(run, session)


@router.get("/{run_id}/status", response_model=RunStatusResponse)
def get_run_status(run_id: str, session: Session = Depends(get_session)) -> RunStatusResponse:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _build_run_status_response(run, session)


@router.get("/{run_id}/predictions")
def get_predictions(run_id: str, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    store_run = store.get_run(run_id)
    status = _normalize_status(store_run.status) if store_run else _normalize_status(run.status)
    progress = store_run.progress if store_run else (100 if status == "done" else 0)
    if status != "done":
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Run not finished",
                "status": status,
                "progress_percent": progress,
            },
        )
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
