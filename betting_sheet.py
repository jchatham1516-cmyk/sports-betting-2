"""Utilities to turn raw model spreadsheets into standardized betting sheets.

This module accepts a user-provided spreadsheet (CSV/XLSX) with model outputs
and augments it with recommendations, play/passes, and bet sizing.  It is
intended to be robust to light schema drift (e.g., column name casing) and to
fill any missing columns with `NaN` so the downstream helpers can operate.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from recommendations import Thresholds, add_recommendations_to_df
from sports.common.bet_rules import (
    MIN_PLAY_EDGE_ABS,
    DEFAULT_UNIT_DOLLARS,
    add_betting_outputs,
)


# Columns we attempt to coerce/ensure for recommendations + bet sizing
CANONICAL_COLUMNS: Sequence[str] = (
    "date",
    "home",
    "away",
    "home_ml",
    "away_ml",
    "home_spread",
    "spread_price",
    "model_home_prob",
    "market_home_prob",
    "model_spread_home",
    "spread_edge_home",
    "ats_edge_vs_be",
    # Totals (optional)
    "total_points",
    "total_over_price",
    "total_under_price",
    "model_total",
    "total_edge_points",
    "total_pick_side",
    "total_edge_vs_be",
    "total_recommendation",
    "total_sd",
    "p_home_cover",
    "win_prob_home",
)


def _canonicalize_columns(df: pd.DataFrame, expected: Iterable[str]) -> pd.DataFrame:
    """Rename columns case-insensitively and ensure missing columns exist."""
    out = df.copy()
    existing_lower = {str(c).lower(): c for c in out.columns}

    rename_map = {}
    for name in expected:
        lower = str(name).lower()
        if lower in existing_lower and name != existing_lower[lower]:
            rename_map[existing_lower[lower]] = name
    if rename_map:
        out = out.rename(columns=rename_map)

    for name in expected:
        if name not in out.columns:
            out[name] = np.nan
    return out


def _load_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    # Fall back to CSV (works for .csv and most delimited files)
    return pd.read_csv(path)


def convert_to_betting_sheet(
    input_path: str,
    *,
    sport: str = "nba",
    output_path: Optional[str] = None,
    thresholds: Optional[Thresholds] = None,
    unit_dollars: float = DEFAULT_UNIT_DOLLARS,
    min_play_edge_abs: float = MIN_PLAY_EDGE_ABS,
) -> pd.DataFrame:
    """
    Convert a user-provided spreadsheet to a standardized betting sheet.

    Steps:
      1. Load CSV/XLSX input.
      2. Normalize/ensure the key columns exist for recommendations + betting.
      3. Add recommendations (ML/ATS/TOTAL) using sport-aware preferences.
      4. Recompute primary recommendation and size bets.
      5. Write the enriched sheet to disk (if `output_path` is provided).
    """

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    df = _load_sheet(src)
    df = _canonicalize_columns(df, CANONICAL_COLUMNS)

    thresholds = thresholds or Thresholds()
    enriched, _ = add_recommendations_to_df(df, thresholds=thresholds, sport=sport)
    betting_df = add_betting_outputs(
        enriched,
        unit_dollars=float(unit_dollars),
        min_play_edge_abs=float(min_play_edge_abs),
    )

    if output_path:
        dest = Path(output_path)
    else:
        dest = src.with_name(f"{src.stem}_betting_sheet.csv")

    os.makedirs(dest.parent, exist_ok=True)
    betting_df.to_csv(dest, index=False)
    print(f"[betting_sheet] wrote {len(betting_df)} rows to {dest}")
    return betting_df


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a spreadsheet into a betting sheet with picks and bet sizes.")
    parser.add_argument("input_path", type=str, help="Path to CSV or XLSX file with model outputs.")
    parser.add_argument("--sport", type=str, default="nba", choices=["nba", "nfl", "nhl"], help="Sport (affects primary preference).")
    parser.add_argument("--output", type=str, default=None, help="Optional output path (defaults to <input>_betting_sheet.csv).")
    parser.add_argument("--unit-dollars", type=float, default=DEFAULT_UNIT_DOLLARS, help="Dollar value of 1 unit for bet sizing.")
    parser.add_argument("--min-play-edge", type=float, default=MIN_PLAY_EDGE_ABS, help="Minimum absolute edge to qualify as a PLAY.")
    args = parser.parse_args(argv)

    convert_to_betting_sheet(
        args.input_path,
        sport=args.sport,
        output_path=args.output,
        unit_dollars=args.unit_dollars,
        min_play_edge_abs=args.min_play_edge,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
