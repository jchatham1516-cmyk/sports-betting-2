"""In-memory run status store."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Iterable
from uuid import uuid4


@dataclass
class RunStatus:
    id: str
    status: str
    progress: int
    message: str
    started_at: datetime
    updated_at: datetime
    error: str | None = None
    key: str | None = None


class RunStatusStore:
    """Thread-safe store for run status metadata."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[str, RunStatus] = {}

    def create_run(self, *, status: str, progress: int, message: str, key: str | None = None) -> RunStatus:
        now = datetime.utcnow()
        run_id = str(uuid4())
        run = RunStatus(
            id=run_id,
            status=status,
            progress=progress,
            message=message,
            started_at=now,
            updated_at=now,
            key=key,
        )
        with self._lock:
            self._runs[run_id] = run
        return run

    def update_run(self, run_id: str, **fields: object) -> RunStatus | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            for field_name, value in fields.items():
                if hasattr(run, field_name):
                    setattr(run, field_name, value)
            run.updated_at = datetime.utcnow()
            return run

    def get_run(self, run_id: str) -> RunStatus | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> Iterable[RunStatus]:
        with self._lock:
            return list(self._runs.values())

    def find_active_by_key(self, key: str) -> RunStatus | None:
        with self._lock:
            for run in self._runs.values():
                if run.key == key and run.status in {"queued", "running"}:
                    return run
        return None


store = RunStatusStore()
