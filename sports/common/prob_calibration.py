from __future__ import annotations
from dataclasses import dataclass
import json
import os
import numpy as np

CAL_DIR = "results"


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0

    def predict(self, p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        x = np.log(p / (1.0 - p))
        z = self.a * x + self.b
        return float(1.0 / (1.0 + np.exp(-z)))


def load(path: str) -> PlattCalibrator:
    if not os.path.exists(path):
        return PlattCalibrator()
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return PlattCalibrator(**d)


def save(path: str, cal: PlattCalibrator) -> None:
    os.makedirs(CAL_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cal.__dict__, f, indent=2)


def fit_platt(ps: np.ndarray, ys: np.ndarray) -> PlattCalibrator:
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    ys = ys.astype(float)

    x = np.log(ps / (1 - ps))
    a, b = 1.0, 0.0
    lr = 0.05

    for _ in range(2000):
        z = a * x + b
        p2 = 1.0 / (1.0 + np.exp(-z))
        a -= lr * np.mean((p2 - ys) * x)
        b -= lr * np.mean(p2 - ys)

    return PlattCalibrator(float(a), float(b))
