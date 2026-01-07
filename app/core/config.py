"""Application configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    database_url: str
    results_dir: Path
    odds_dir: Path


def get_settings() -> Settings:
    """Return application settings."""
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        database_url = "sqlite:///app/db/app.db"

    return Settings(
        database_url=database_url,
        results_dir=Path(os.getenv("RESULTS_DIR", "results")),
        odds_dir=Path(os.getenv("ODDS_DIR", "odds")),
    )
