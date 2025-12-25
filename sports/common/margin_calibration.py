from __future__ import annotations
from dataclasses import dataclass
import json
import os
import numpy as np

CAL_DIR = "results"


@dataclass
class MarginCalibrator:
    a: float = 0.0
    b: float = 0.0

    def predict(self, elo_diff: float) -> float:
        return float(self.a + self.b * elo_diff)


def load(path: str) -> MarginCalibrator:
    if not os.path.exists(path):
        return MarginCalibrator()
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return MarginCalibrator(**d)


def save(path: str, cal: MarginCalibrator) -> None:
    os.makedirs(CAL_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cal.__dict__, f, indent=2)


def fit(xs: np.ndarray, ys: np.ndarray, lam: float = 10.0) -> MarginCalibrator:
    if len(xs) < 20:
        return MarginCalibrator()

    X = np.column_stack([np.ones_like(xs), xs])
    I = np.eye(2)
    I[0, 0] = 0.0

    beta = np.linalg.inv(X.T @ X + lam * I) @ (X.T @ ys)
    return MarginCalibrator(float(beta[0]), float(beta[1]))
