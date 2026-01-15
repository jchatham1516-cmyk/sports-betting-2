from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sports.common.util import clamp, normalize_result_label


MIN_SAMPLES = 300
_CALIBRATOR_CACHE: Dict[str, "PlattCalibrator"] = {}


@dataclass
class PlattCalibrator:
    a: float
    b: float
    n_samples: int = 0

    def predict(self, p_raw: float) -> float:
        return apply_platt(p_raw, self.a, self.b)

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b), "n_samples": int(self.n_samples)}

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "PlattCalibrator":
        return PlattCalibrator(
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            n_samples=int(data.get("n_samples", 0)),
        )


def _calibration_path(sport: str) -> str:
    return os.path.join("results", "calibration", f"platt_{sport}.json")


def _fit_platt_scaling(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    if x.size == 0:
        return None

    a, b = 1.0, 0.0
    reg = 1e-4

    for _ in range(50):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        w = np.clip(w, 1e-6, None)
        z_adj = z + (y - p) / w

        xw = x * w
        sxx = float((xw * x).sum()) + reg
        sx = float(xw.sum())
        sw = float(w.sum()) + reg
        szx = float((w * z_adj * x).sum())
        sz = float((w * z_adj).sum())

        det = sxx * sw - sx * sx
        if abs(det) < 1e-12:
            break

        new_a = (szx * sw - sx * sz) / det
        new_b = (sxx * sz - sx * szx) / det

        if abs(new_a - a) < 1e-6 and abs(new_b - b) < 1e-6:
            a, b = new_a, new_b
            break
        a, b = new_a, new_b

    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return float(a), float(b)


def fit_platt(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p_raw = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p_raw) & np.isfinite(y)
    p_raw = np.clip(p_raw[mask], 1e-6, 1 - 1e-6)
    y = y[mask]
    x = np.log(p_raw / (1.0 - p_raw))
    fitted = _fit_platt_scaling(x, y)
    if fitted is None:
        return {"a": 1.0, "b": 0.0}
    a, b = fitted
    return {"a": float(a), "b": float(b)}


def apply_platt(p_raw: float, a: float, b: float) -> float:
    if p_raw is None or not np.isfinite(p_raw):
        return float("nan")
    p_raw = float(clamp(float(p_raw), 1e-6, 1 - 1e-6))
    logit = np.log(p_raw / (1 - p_raw))
    z = float(a) * logit + float(b)
    return float(1.0 / (1.0 + np.exp(-z)))


def save_calibrator(sport: str, params: Dict[str, float], n_samples: int = 0) -> None:
    os.makedirs(os.path.join("results", "calibration"), exist_ok=True)
    payload = {
        "a": float(params.get("a", 1.0)),
        "b": float(params.get("b", 0.0)),
        "n_samples": int(n_samples),
    }
    with open(_calibration_path(str(sport).lower()), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_calibrator(sport: str) -> Optional[Dict[str, float]]:
    sport_key = str(sport).lower()
    if sport_key in _CALIBRATOR_CACHE:
        cal = _CALIBRATOR_CACHE[sport_key]
        return {"a": cal.a, "b": cal.b}

    path = _calibration_path(sport_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    cal = PlattCalibrator.from_dict(payload)
    _CALIBRATOR_CACHE[sport_key] = cal
    return {"a": cal.a, "b": cal.b}


def fit_calibrator(sport: str, rows_df: pd.DataFrame) -> Optional[PlattCalibrator]:
    sport_key = str(sport).lower()
    work = rows_df.copy()

    if "sport" in work.columns:
        work = work[work["sport"].astype(str).str.lower() == sport_key]

    if work.empty:
        return None

    result_col = None
    if "home_win" in work.columns:
        result_col = "home_win"
    elif "result" in work.columns:
        work["result_norm"] = work.get("result", "").apply(normalize_result_label)
        work = work[work["result_norm"].isin(["WIN", "LOSS"])]
        result_col = "result_norm"

    if result_col is None or work.empty:
        return None

    if "model_home_prob_raw" in work.columns:
        p_raw = pd.to_numeric(work["model_home_prob_raw"], errors="coerce").astype(float)
    elif "p_model_raw" in work.columns:
        p_raw = pd.to_numeric(work["p_model_raw"], errors="coerce").astype(float)
    elif "market_home_prob" in work.columns:
        print("[calibration] WARNING: using market_home_prob as proxy for model probs.")
        p_raw = pd.to_numeric(work["market_home_prob"], errors="coerce").astype(float)
    else:
        return None

    if result_col == "result_norm":
        y = (work[result_col] == "WIN").astype(float)
    else:
        y = pd.to_numeric(work[result_col], errors="coerce").astype(float)

    mask = np.isfinite(p_raw.to_numpy()) & np.isfinite(y.to_numpy())
    p_raw = p_raw.to_numpy()[mask]
    y = y.to_numpy()[mask]

    if p_raw.size < MIN_SAMPLES:
        print(f"[calibration] WARNING: {sport_key} only {p_raw.size} samples; skipping calibration.")
        return None

    params = fit_platt(p_raw, y)
    calibrator = PlattCalibrator(a=params["a"], b=params["b"], n_samples=int(p_raw.size))
    _CALIBRATOR_CACHE[sport_key] = calibrator
    save_calibrator(sport_key, params, n_samples=int(p_raw.size))
    return calibrator


def calibrate_prob(
    sport_or_probs: Union[str, np.ndarray, list],
    p_raw_or_y: Optional[Union[float, np.ndarray, list]] = None,
    market_type: Optional[str] = None,
) -> Union[float, Tuple[PlattCalibrator, str]]:
    if isinstance(sport_or_probs, str):
        if market_type and str(market_type).upper() != "ML":
            return float(p_raw_or_y) if p_raw_or_y is not None else float("nan")
        params = load_calibrator(sport_or_probs)
        if params is None:
            return float(p_raw_or_y) if p_raw_or_y is not None else float("nan")
        return apply_platt(float(p_raw_or_y), params["a"], params["b"])

    if p_raw_or_y is None:
        return float("nan")

    probs = np.asarray(sport_or_probs, dtype=float)
    ys = np.asarray(p_raw_or_y, dtype=float)
    params = fit_platt(probs, ys)
    return PlattCalibrator(a=params["a"], b=params["b"]), "platt"
