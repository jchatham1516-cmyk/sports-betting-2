"""Database session utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings

_engine = None


def get_engine():
    """Return a SQLModel engine instance."""
    global _engine
    if _engine is not None:
        return _engine

    settings = get_settings()
    database_url = settings.database_url

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
        if database_url.startswith("sqlite:///"):
            db_path = Path(database_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(database_url, connect_args=connect_args)
    return _engine


def init_db() -> None:
    """Create database tables."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Yield a database session."""
    engine = get_engine()
    with Session(engine) as session:
        yield session
