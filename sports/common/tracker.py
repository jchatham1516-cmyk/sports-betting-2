from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, Tuple

import pandas as pd


# ---------------------------
# Helpers
# ---------------------------

def _american_to_decimal(american: float) -> float:
    a = float(american)
    if a == 0:
        return 1.0
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))


def _calc_profit(stake: float, price_american: float, result: str) -> float:
    """
    Returns profit (not payout). Profit excludes stake return.
    WIN: + stake*(decimal-1)
    LOSS: - stake
    PUSH: 0
    """
    stake = float(stake)
    if result == "WIN":
        dec = _american_to_decimal(price_american)
        return stake * (dec - 1.0)
    if result == "LOSS":
        return -stake
    return 0.0


def _find_latest_recs_csv(results_dir: str, target_date: date) -> Optional[str]:
    """
    Finds a recommendations file for the target date.
    Adjust pattern to match your repo's naming convention.
    """
    ds = target_date.strftime("%m-%d-%Y")
    patterns = [
        os.path.join(results_dir, f"*{ds}*.csv"),
        os.path.join(results_dir, f"*{target_date.isoformat()}*.csv"),
        os.path.join(results_dir, "*.csv"),
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    # Prefer files that look like outputs (you can tighten this)
    candidates = [c for c in candidates if "tracking" not in c.lower()]
    if not candidates:
        return None

    # newest file wins
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# ---------------------------
# Grading logic
# ---------------------------

def grade_moneyline(side: str, home_score: int, away_score: int) -> str:
    if home_score is None or away_score is None:
        return "MISSING_SCORE"
    if home_score == away_score:
        # NBA/NFL shouldn't tie often; handle anyway
        return "PUSH"
    winner = "HOME" if home_score > away_score else "AWAY"
    return "WIN" if side.upper() == winner else "LOSS"


def grade_spread(side: str, spread_home: float, home_score: int, away_score: int) -> str:
    if home_score is None or away_score is None:
        return "MISSING_SCORE"
    spread_home = float(spread_home)

    # HOME ATS uses home_score + spread_home
    if side.upper() == "HOME":
        lhs = home_score + spread_home
        rhs = away_score
    else:  # AWAY
        lhs = away_score - spread_home
        rhs = home_score

    if abs(lhs - rhs) < 1e-9:
        return "PUSH"
    return "WIN" if lhs > rhs else "LOSS"


def grade_total(side: str, total_line: float, home_score: int, away_score: int) -> str:
    if home_score is None or away_score is None:
        return "MISSING_SCORE"
    total = home_score + away_score
    line = float(total_line)
    if abs(total - line) < 1e-9:
        return "PUSH"
    if side.upper() == "OVER":
        return "WIN" if total > line else "LOSS"
    return "WIN" if total < line else "LOSS"


# ---------------------------
# Main entrypoint
# ---------------------------

def track_yesterday(
    sport: str,
    results_dir: str = "results",
    tracking_dir: str = "results/tracking",
    unit_dollars_default: float = 10.0,
) -> Tuple[Optional[str], Dict]:
    """
    Reads yesterday's recommendations CSV, fetches final scores using your existing score fetcher,
    grades bets, and writes tracking outputs.

    NOTE: You must wire in your repo's score fetch function below.
    """
    yday = date.today() - timedelta(days=1)

    os.makedirs(tracking_dir, exist_ok=True)

    recs_path = _find_latest_recs_csv(results_dir, yday)
    if not recs_path:
        return None, {"ok": False, "reason": "No recommendations CSV found for yesterday."}

    df = pd.read_csv(recs_path)

    # --- You MUST adapt these column names to your output ---
    # Expected minimum:
    # date, home, away, plus something that indicates recommended bets and lines/prices/units
    required_cols = {"date", "home", "away"}
    missing = required_cols - set(df.columns)
    if missing:
        return recs_path, {"ok": False, "reason": f"Missing required columns: {sorted(missing)}"}

    # -------------------------------------------------------
    # Pull final scores for yesterday (WIRE THIS TO YOUR REPO)
    # -------------------------------------------------------
    # Replace this import with whatever you already use for scores
    # Example idea:
    # from sports.common.scores_sources import fetch_scores_history_by_day
    # scores_df = fetch_scores_history_by_day("nba", yday)  # must return home/away + scores
    #
    # For now, we assume you will produce a DataFrame:
    # columns: home, away, home_score, away_score
    #
    raise NotImplementedError(
        "Wire track_yesterday() into your existing score fetcher. "
        "Return scores_df with columns: home, away, home_score, away_score."
    )

    # Example merge logic once you have scores_df:
    # merged = df.merge(scores_df, on=["home", "away"], how="left")

    # Then: build rows of bets from your recommendation columns.
    # If your recs CSV has columns like:
    # - primary_recommendation or ml_recommendation/spread_recommendation/total_recommendation
    # - home_ml, away_ml, home_spread, total_points
    # - units / unit_dollars
    #
    # Create a normalized "bets" table, one row per bet.
