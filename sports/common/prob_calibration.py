# sports/common/prob_calibration.py
from __future__ import annotations
from dataclasses import dataclass
import json, os
import numpy as np

CAL_DIR = "results"
PATH_NBA = os.path.join(CAL_DIR, "prob_cal_nba.json")

@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0

    def predict(self, p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        logit = np.log(p / (1 - p))
        z = self.a * logit + self.b
        return float(1.0 / (1.0 + np.exp(-z)))

def fit_platt(ps: np.ndarray, ys: np.ndarray) -> PlattCalibrator:
    # simple gradient descent (no sklearn dependency)
    ps = np.clip(ps.astype(float), 1e-6, 1 - 1e-6)
    ys = ys.astype(float)

    x = np.log(ps / (1 - ps))
    a, b = 1.0, 0.0
    lr = 0.05

    for _ in range(2000):
        z = a * x + b
        p2 = 1.0 / (1.0 + np.exp(-z))
        # gradients for log loss
        da = np.mean((p2 - ys) * x)
        db = np.mean(p2 - ys)
        a -= lr * da
        b -= lr * db

    return PlattCalibrator(a=float(a), b=float(b))

def load(path: str) -> PlattCalibrator:
    os.makedirs(CAL_DIR, exist_ok=True)
    if not os.path.exists(path):
        return PlattCalibrator()
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return PlattCalibrator(a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)))

def save(path: str, cal: PlattCalibrator) -> None:
    os.makedirs(CAL_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"a": cal.a, "b": cal.b}, f)
