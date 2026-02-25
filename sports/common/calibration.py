from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sports.common.normalization import bounded_probability

CalibrationMethod = Literal["isotonic", "platt"]


class ProbabilityCalibrator:
    def __init__(self, method: CalibrationMethod = "isotonic") -> None:
        self.method = method
        self.model: IsotonicRegression | LogisticRegression | None = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "ProbabilityCalibrator":
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        if self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(probs, outcomes)
        else:
            model = LogisticRegression(max_iter=200)
            model.fit(probs.reshape(-1, 1), outcomes)
        self.model = model
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([bounded_probability(float(p)) for p in probs], dtype=float)
        probs = np.asarray(probs, dtype=float)
        if self.method == "isotonic":
            calibrated = self.model.predict(probs)
        else:
            calibrated = self.model.predict_proba(probs.reshape(-1, 1))[:, 1]
        return np.array([bounded_probability(float(p)) for p in calibrated], dtype=float)

    def to_payload(self) -> dict:
        if self.model is None:
            raise ValueError("Calibrator has not been fitted.")
        if self.method == "isotonic":
            return {
                "method": self.method,
                "x_thresholds": self.model.X_thresholds_.tolist(),
                "y_thresholds": self.model.y_thresholds_.tolist(),
            }
        return {
            "method": self.method,
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
        }

    @staticmethod
    def from_payload(payload: dict) -> "ProbabilityCalibrator":
        method = payload["method"]
        calibrator = ProbabilityCalibrator(method=method)
        if method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.X_thresholds_ = np.asarray(payload["x_thresholds"], dtype=float)
            model.y_thresholds_ = np.asarray(payload["y_thresholds"], dtype=float)
            calibrator.model = model
        else:
            model = LogisticRegression(max_iter=200)
            model.coef_ = np.asarray(payload["coef"], dtype=float)
            model.intercept_ = np.asarray(payload["intercept"], dtype=float)
            model.classes_ = np.array([0.0, 1.0], dtype=float)
            calibrator.model = model
        return calibrator


def calibrator_path(base_dir: Path, sport: str, market: str) -> Path:
    return base_dir / sport / f"{market}_calibrator.json"


def save_calibrator(base_dir: Path, sport: str, market: str, calibrator: ProbabilityCalibrator) -> Path:
    path = calibrator_path(base_dir, sport, market)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibrator.to_payload(), indent=2), encoding="utf-8")
    return path


def load_calibrator(base_dir: Path, sport: str, market: str) -> ProbabilityCalibrator | None:
    path = calibrator_path(base_dir, sport, market)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ProbabilityCalibrator.from_payload(payload)
