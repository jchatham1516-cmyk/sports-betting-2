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

    # Alias for compatibility
    def predict_proba(self, p: float) -> float:
        return self.predict(p)


@dataclass
class IsotonicCalibrator:
    """Lightweight isotonic-like calibrator without external deps."""

    thresholds: list = None
    values: list = None

    def predict(self, p: float) -> float:
        if not self.thresholds or not self.values:
            return float(p)
        p = float(np.clip(p, 0.0, 1.0))
        for t, v in zip(self.thresholds, self.values):
            if p <= t:
                return float(v)
        return float(self.values[-1])

    def predict_proba(self, p: float) -> float:
        return self.predict(p)


def load(path: str):
    if not os.path.exists(path):
        return PlattCalibrator()
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("type") == "isotonic":
        return IsotonicCalibrator(thresholds=d.get("thresholds"), values=d.get("values"))
    return PlattCalibrator(a=d.get("a", 1.0), b=d.get("b", 0.0))


def save(path: str, cal) -> None:
    os.makedirs(CAL_DIR, exist_ok=True)
    payload = getattr(cal, "__dict__", {})
    if isinstance(cal, IsotonicCalibrator):
        payload = {"type": "isotonic", "thresholds": cal.thresholds, "values": cal.values}
    else:
        payload = {"type": "platt", "a": getattr(cal, "a", 1.0), "b": getattr(cal, "b", 0.0)}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def fit_isotonic(ps: np.ndarray, ys: np.ndarray, min_samples: int = 50) -> IsotonicCalibrator:
    """Simple, monotone calibration via pooled adjacent violators.

    This avoids external dependencies while still dampening overconfidence in
    regions with sparse data.
    """

    if len(ps) < int(min_samples):
        return IsotonicCalibrator()

    order = np.argsort(ps)
    ps_sorted = ps[order]
    ys_sorted = ys[order]

    # pooled adjacent violators
    buckets = []
    for p, y in zip(ps_sorted, ys_sorted):
        buckets.append([p, y, 1])  # p_sum, y_sum, count
        while len(buckets) >= 2 and buckets[-2][1] / max(buckets[-2][2], 1) > buckets[-1][1] / max(buckets[-1][2], 1):
            b1 = buckets.pop()
            b0 = buckets.pop()
            buckets.append([b0[0] + b1[0], b0[1] + b1[1], b0[2] + b1[2]])

    thresholds = []
    values = []
    cum = 0
    for p_sum, y_sum, cnt in buckets:
        cum += cnt
        thresholds.append(float(ps_sorted[min(len(ps_sorted) - 1, cum - 1)]))
        values.append(float(y_sum / max(cnt, 1)))

    return IsotonicCalibrator(thresholds=thresholds, values=values)


def calibrate_prob(ps: np.ndarray, ys: np.ndarray) -> tuple[object, str]:
    """Return a calibrator (isotonic preferred) and label."""

    try:
        iso = fit_isotonic(ps, ys)
        if iso.thresholds:
            return iso, "isotonic"
    except Exception:
        pass

    return fit_platt(ps, ys), "platt"
