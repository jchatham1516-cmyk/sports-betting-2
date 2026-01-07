"""Runner that wraps the existing model CLI."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_SPORTS = {"nba", "nfl", "nhl"}


def _script_supports_flag(script_path: Path, flag: str) -> bool:
    try:
        contents = script_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return flag in contents


def _normalize_game_date(game_date: str) -> str:
    """Convert YYYY-MM-DD to MM/DD/YYYY if needed."""
    if "/" in game_date:
        return game_date
    return datetime.strptime(game_date, "%Y-%m-%d").strftime("%m/%d/%Y")


def _settings_to_args(settings: dict[str, Any] | None) -> list[str]:
    args: list[str] = []
    if not settings:
        return args
    for key, value in settings.items():
        if value is None:
            continue
        flag = key
        if not flag.startswith("--"):
            flag = f"--{flag}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])
    return args


def _latest_csv(results_dir: Path, pattern: str) -> Path | None:
    candidates = list(results_dir.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_market_from_row(row: dict[str, Any]) -> str:
    text = str(row.get("primary_recommendation") or row.get("market") or "").lower()
    if "total" in text or "over" in text or "under" in text:
        return "total"
    if "spread" in text or "ats" in text:
        return "spread"
    return "ml"


def _build_tracked_bets(predictions_df: pd.DataFrame, sport: str, game_date: str) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()

    plays_df = predictions_df.copy()
    if "play_pass" in plays_df.columns:
        plays_df = plays_df[plays_df["play_pass"].astype(str) == "PLAY"].copy()
    if plays_df.empty:
        return pd.DataFrame()

    bet_date = game_date

    records = []
    for _, row in plays_df.iterrows():
        row_dict = row.to_dict()
        market = _infer_market_from_row(row_dict)
        home = row_dict.get("home") or row_dict.get("home_team") or ""
        away = row_dict.get("away") or row_dict.get("away_team") or ""
        pick = row_dict.get("pick") or row_dict.get("primary_recommendation") or ""
        price = row_dict.get("price")
        if price is None:
            if "HOME" in str(pick).upper():
                price = row_dict.get("home_ml")
            elif "AWAY" in str(pick).upper():
                price = row_dict.get("away_ml")
        units = row_dict.get("units") or row_dict.get("bet_size") or 0.0
        records.append(
            {
                "bet_date": bet_date,
                "sport": sport,
                "market": market,
                "home": home,
                "away": away,
                "pick": pick,
                "price": price,
                "units": units,
                "result": "PENDING",
            }
        )

    return pd.DataFrame(records)


def run_model(sport: str, game_date: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the existing model CLI and return predictions/tracked bets."""
    if sport not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport: {sport}")

    script_path = Path("current_of_sports_betting_algorithm.py")
    if not script_path.exists():
        raise FileNotFoundError("current_of_sports_betting_algorithm.py not found")

    normalized_date = _normalize_game_date(game_date)

    cmd = [sys.executable, str(script_path), "--sport", sport]
    if _script_supports_flag(script_path, "--date"):
        cmd.extend(["--date", normalized_date])

    cmd.extend(_settings_to_args(settings))

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    log = "".join([stdout, "\n", stderr]).strip()

    if completed.returncode != 0:
        raise RuntimeError(f"Runner failed ({completed.returncode}): {log}")

    results_dir = Path("results")
    predictions_path = _latest_csv(results_dir, f"predictions_{sport}_*.csv")
    if predictions_path is None:
        raise FileNotFoundError("No predictions CSV found in results/")

    predictions_df = pd.read_csv(predictions_path)

    tracked_path = _latest_csv(results_dir, f"tracked_bets_{sport}_*.csv")
    tracked_df = pd.DataFrame()
    if tracked_path and tracked_path.exists():
        tracked_df = pd.read_csv(tracked_path)
    else:
        tracked_df = _build_tracked_bets(predictions_df, sport, game_date)
        if not tracked_df.empty:
            tracked_path = results_dir / f"tracked_bets_{sport}_{game_date}.csv"
            tracked_df.to_csv(tracked_path, index=False)
        else:
            tracked_path = None

    return {
        "predictions_path": str(predictions_path),
        "tracked_bets_path": str(tracked_path) if tracked_path else None,
        "predictions_rows": predictions_df.to_dict(orient="records"),
        "tracked_bets_rows": tracked_df.to_dict(orient="records"),
        "log": log,
    }
