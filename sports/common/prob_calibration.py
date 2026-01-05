"""
sports/common/prob_calibration.py

Probability calibration for sports betting models.

Supports:
1) "Per-bet-type Platt params" format used by ProbabilityCalibrator:
   {
     "moneyline": {"A": 1.0, "B": 0.0},
     "spread":    {"A": 1.0, "B": 0.0},
     "total":     {"A": 1.0, "B": 0.0}
   }

2) "Single calibrator object" expected by older code paths:
   - load(path) returns an object with .predict(p)
   - save(path, calibrator) persists it
   - calibrate_prob(ps, ys) fits a simple Platt scaler and returns (calibrator, label)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# -------------------------
# Numerically stable helpers
# -------------------------
def _clip_prob(p: float) -> float:
    try:
        return float(np.clip(float(p), 0.001, 0.999))
    except Exception:
        return 0.5


def _logit(p: float) -> float:
    p = _clip_prob(p)
    return float(math.log(p / (1.0 - p)))


def _sigmoid(x: float) -> float:
    # stable sigmoid
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# -------------------------
# Simple Platt Calibrator object (legacy API)
# -------------------------
@dataclass
class PlattCalibrator:
    """Calibrator with .predict(p) used by existing model code."""
    A: float = 1.0
    B: float = 0.0

    def predict(self, p: float) -> float:
        p = _clip_prob(p)
        x = _logit(p)
        y = _sigmoid(self.A * x + self.B)
        return float(np.clip(y, 0.01, 0.99))

    def to_dict(self) -> Dict[str, float]:
        return {"A": float(self.A), "B": float(self.B)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PlattCalibrator":
        A = float(d.get("A", 1.0))
        B = float(d.get("B", 0.0))
        return PlattCalibrator(A=A, B=B)


# -------------------------
# New-style per bet-type calibrator (your class)
# -------------------------
class ProbabilityCalibrator:
    """
    Platt scaling-based calibration for sports betting probabilities.
    Maintains separate calibrators for ML, ATS, and Totals per sport.

    File format:
      { "moneyline": {"A":..,"B":..}, "spread": {...}, "total": {...} }
    """

    def __init__(self, sport: str, calibration_path: Optional[str] = None):
        self.sport = str(sport)
        self.calibrators: Dict[str, Dict[str, float]] = {}
        self.calibration_path = calibration_path or self._default_path()
        self.load_calibration()

    def _default_path(self) -> str:
        # Prefer "results/" (common in your repo), but keep backward compat with the original path too.
        # You can override with calibration_path.
        cal_dir = os.getenv("CALIBRATION_DIR", "results")
        return os.path.join(cal_dir, f"calibration_params_{self.sport}.json")

    def load_calibration(self) -> None:
        """Load pre-fitted calibration parameters (per bet type)."""
        path = self.calibration_path

        # Backward compatible fallback:
        fallback = os.path.join("sports", "common", f"calibration_params_{self.sport}.json")

        load_path = path if os.path.exists(path) else (fallback if os.path.exists(fallback) else None)

        if load_path:
            try:
                with open(load_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # ensure structure
                if isinstance(data, dict):
                    self.calibrators = data
                else:
                    self.calibrators = {}
            except Exception:
                self.calibrators = {}
        else:
            self.calibrators = {}

        # Defaults if missing
        for bt in ("moneyline", "spread", "total"):
            if bt not in self.calibrators or not isinstance(self.calibrators.get(bt), dict):
                self.calibrators[bt] = {"A": 1.0, "B": 0.0}
            else:
                # sanitize
                self.calibrators[bt]["A"] = float(self.calibrators[bt].get("A", 1.0))
                self.calibrators[bt]["B"] = float(self.calibrators[bt].get("B", 0.0))

    def save_calibration(self) -> None:
        """Persist calibration parameters."""
        try:
            os.makedirs(os.path.dirname(self.calibration_path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(self.calibration_path, "w", encoding="utf-8") as f:
                json.dump(self.calibrators, f, indent=2)
        except Exception:
            pass

    def calibrate(self, prob: float, bet_type: str) -> float:
        """
        Apply Platt scaling: sigmoid(A * logit(prob) + B)
        """
        bt = str(bet_type)
        if bt not in self.calibrators:
            return float(_clip_prob(prob))

        params = self.calibrators[bt]
        A = float(params.get("A", 1.0))
        B = float(params.get("B", 0.0))

        p = _clip_prob(prob)
        x = _logit(p)
        y = _sigmoid(A * x + B)
        return float(np.clip(y, 0.01, 0.99))


# -------------------------
# Legacy functions expected by your NBA model
# -------------------------
def load(path: str) -> Optional[PlattCalibrator]:
    """
    Load a calibrator saved by save() or compatible dict.
    Returns an object with .predict(p).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # If someone saved the per-bet-type dict here by mistake, try to use moneyline.
        if isinstance(d, dict) and "A" in d and "B" in d:
            return PlattCalibrator.from_dict(d)
        if isinstance(d, dict) and "moneyline" in d and isinstance(d["moneyline"], dict):
            return PlattCalibrator.from_dict(d["moneyline"])
        return None
    except Exception:
        return None


def save(path: str, calibrator: Any) -> None:
    """
    Save a calibrator. Accepts:
      - PlattCalibrator
      - dict with A/B
      - object with attributes A/B
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

    out: Dict[str, float]
    if isinstance(calibrator, PlattCalibrator):
        out = calibrator.to_dict()
    elif isinstance(calibrator, dict):
        out = {"A": float(calibrator.get("A", 1.0)), "B": float(calibrator.get("B", 0.0))}
    else:
        out = {"A": float(getattr(calibrator, "A", 1.0)), "B": float(getattr(calibrator, "B", 0.0))}

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass


def fit_platt(ps: np.ndarray, ys: np.ndarray, *, l2: float = 1e-3, max_iter: int = 500) -> PlattCalibrator:
    """
    Fit Platt scaling on (ps -> ys) using a tiny Newton method on logistic regression over logit(ps).
    ps: predicted probabilities (0..1)
    ys: outcomes (0/1)
    """
    p = np.clip(np.asarray(ps, dtype=float), 0.001, 0.999)
    y = np.asarray(ys, dtype=float)

    x = np.log(p / (1.0 - p))  # logit

    # Initialize A,B
    A = 1.0
    B = 0.0

    for _ in range(int(max_iter)):
        z = A * x + B
        # sigmoid(z) stable enough for numpy if we clip
        z_clip = np.clip(z, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-z_clip))

        # gradients
        gA = np.sum((mu - y) * x) + l2 * A
        gB = np.sum(mu - y)

        # Hessian components
        w = mu * (1.0 - mu)
        hAA = np.sum(w * x * x) + l2
        hAB = np.sum(w * x)
        hBB = np.sum(w)

        det = hAA * hBB - hAB * hAB
        if det <= 1e-12:
            break

        # Newton step
        dA = (hBB * gA - hAB * gB) / det
        dB = (-hAB * gA + hAA * gB) / det

        A_new = A - dA
        B_new = B - dB

        # small convergence check
        if abs(A_new - A) + abs(B_new - B) < 1e-8:
            A, B = A_new, B_new
            break

        A, B = A_new, B_new

    return PlattCalibrator(A=float(A), B=float(B))


def calibrate_prob(ps: np.ndarray, ys: np.ndarray) -> Tuple[PlattCalibrator, str]:
    """
    Compatibility wrapper used by your NBA model.
    Returns (calibrator, label).
    """
    cal = fit_platt(ps, ys)
    return cal, "platt"


# Backward compatibility helper from your snippet
def calibrate_probability(prob: float, sport: str, bet_type: str = "moneyline") -> float:
    calibrator = ProbabilityCalibrator(sport)
    return calibrator.calibrate(prob, bet_type)
