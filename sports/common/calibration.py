# sports/common/calibration.py
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


CALIBRATION_DIR = "results"
NBA_CALIBRATION_PATH = os.path.join(CALIBRATION_DIR, "calibration_nba.json")


@dataclass
class LinearCalibrator:
    """
    Maps elo_diff -> spread via: spread = a + b * elo_diff
    """
    a: float = 0.0
    b: float = -1.0 / 30.0  # default slope (tunable)

    min_points: int = 40
    max_files: int = 60  # most recent files to scan

    def predict_spread(self, elo_diff: float) -> float:
        return float(self.a + self.b * float(elo_diff))

    def to_dict(self) -> dict:
        return {"a": float(self.a), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearCalibrator":
        return cls(a=float(d.get("a", 0.0)), b=float(d.get("b", -1.0 / 30.0)))


def load_nba_calibrator() -> LinearCalibrator:
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    if not os.path.exists(NBA_CALIBRATION_PATH):
        return LinearCalibrator()
    try:
        with open(NBA_CALIBRATION_PATH, "r", encoding="utf-8") as f:
            return LinearCalibrator.from_dict(json.load(f))
    except Exception:
        return LinearCalibrator()


def save_nba_calibrator(cal: LinearCalibrator) -> None:
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    with open(NBA_CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(cal.to_dict(), f, indent=2)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def fit_nba_calibration_from_history(
    *,
    pattern: str = os.path.join(CALIBRATION_DIR, "predictions_nba_*.csv"),
    min_points: int = 40,
    max_files: int = 60,
) -> Tuple[LinearCalibrator, int]:
    """
    Fits spread = a + b * elo_diff using your historical saved predictions CSVs.
    Requires columns: elo_diff, home_spread
    """
    files = sorted(glob.glob(pattern))[-max_files:]
    if not files:
        return LinearCalibrator(min_points=min_points, max_files=max_files), 0

    xs = []
    ys = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue

        if "elo_diff" not in df.columns or "home_spread" not in df.columns:
            continue

        for _, r in df.iterrows():
            xd = _safe_float(r.get("elo_diff"))
            yd = _safe_float(r.get("home_spread"))
            if xd is None or yd is None:
                continue

            # Ignore wild market spreads (data glitch)
            if abs(yd) > 25:
                continue

            xs.append(xd)
            ys.append(yd)

    n = len(xs)
    if n < min_points:
        return LinearCalibrator(min_points=min_points, max_files=max_files), n

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)

    # Linear regression with intercept: y = a + b*x
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    cal = LinearCalibrator(a=a, b=b, min_points=min_points, max_files=max_files)
    return cal, n


def update_and_save_nba_calibration() -> LinearCalibrator:
    cal, n = fit_nba_calibration_from_history()
    if n >= cal.min_points:
        save_nba_calibrator(cal)
    return cal
