from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sports.common.util import clamp, normalize_result_label


MIN_SAMPLES = 200
_CALIBRATOR_CACHE: Dict[str, "Calibrator"] = {}


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def predict(self, p_raw: float) -> float:
        if p_raw is None or not np.isfinite(p_raw):
            return float("nan")
        p_raw = float(clamp(float(p_raw), 1e-6, 1 - 1e-6))
        logit = np.log(p_raw / (1 - p_raw))
        z = self.a * logit + self.b
        return float(1.0 / (1.0 + np.exp(-z)))

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b)}

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "PlattCalibrator":
        return PlattCalibrator(a=float(data.get("a", 1.0)), b=float(data.get("b", 0.0)))


@dataclass
class MarketCalibrator:
    a: float
    b: float
    n_samples: int
    status: str


@dataclass
class Calibrator:
    sport: str
    markets: Dict[str, MarketCalibrator]
    min_samples: int = MIN_SAMPLES

    def calibrate(self, p_raw: float, market_type: Optional[str] = None) -> float:
        if p_raw is None or not np.isfinite(p_raw):
            return float("nan")
        p_raw = float(clamp(float(p_raw), 1e-6, 1 - 1e-6))
        market_key = (market_type or "ML").upper()
        market_cal = self.markets.get(market_key)
        if market_cal is None or market_cal.status != "fit":
            return p_raw
        logit = np.log(p_raw / (1 - p_raw))
        z = market_cal.a * logit + market_cal.b
        return float(1.0 / (1.0 + np.exp(-z)))


def _calibration_path(sport: str) -> str:
    return os.path.join("results", "calibration", f"prob_cal_{sport}.json")


def load_calibrator(sport: str) -> Optional[Calibrator]:
    sport_key = str(sport).lower()
    if sport_key in _CALIBRATOR_CACHE:
        return _CALIBRATOR_CACHE[sport_key]

    path = _calibration_path(sport_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    markets: Dict[str, MarketCalibrator] = {}
    for key, data in (payload.get("markets") or {}).items():
        markets[key.upper()] = MarketCalibrator(
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            n_samples=int(data.get("n_samples", 0)),
            status=str(data.get("status", "fallback")),
        )

    calibrator = Calibrator(
        sport=payload.get("sport", sport_key),
        markets=markets,
        min_samples=int(payload.get("min_samples", MIN_SAMPLES)),
    )
    _CALIBRATOR_CACHE[sport_key] = calibrator
    return calibrator


def _fit_platt_scaling(x: np.ndarray, y: np.ndarray) -> Optional[MarketCalibrator]:
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
    return MarketCalibrator(a=float(a), b=float(b), n_samples=int(x.size), status="fit")


def fit_platt(p_raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p_raw) & np.isfinite(y)
    p_raw = np.clip(p_raw[mask], 1e-6, 1 - 1e-6)
    y = y[mask]
    x = np.log(p_raw / (1.0 - p_raw))
    fitted = _fit_platt_scaling(x, y)
    if fitted is None:
        return PlattCalibrator(a=1.0, b=0.0)
    return PlattCalibrator(a=fitted.a, b=fitted.b)


def save(path: str, calibrator: PlattCalibrator) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calibrator.to_dict(), f, indent=2)


def load(path: str) -> Optional[PlattCalibrator]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PlattCalibrator.from_dict(data)
    except Exception:
        return None


def fit_calibrator(sport: str, rows_df: pd.DataFrame) -> Calibrator:
    sport_key = str(sport).lower()
    work = rows_df.copy()

    if "sport" in work.columns:
        work = work[work["sport"].astype(str).str.lower() == sport_key]

    if work.empty:
        calibrator = Calibrator(sport=sport_key, markets={}, min_samples=MIN_SAMPLES)
        _CALIBRATOR_CACHE[sport_key] = calibrator
        return calibrator

    work = work.copy()
    work["result_norm"] = work.get("result", "").apply(normalize_result_label)
    work = work[work["result_norm"].isin(["WIN", "LOSS"])]

    markets: Dict[str, MarketCalibrator] = {}
    for market, sub in work.groupby(work.get("market_type", "")):
        market_key = str(market).lower()
        if market_key in {"moneyline", "ml"}:
            market_key = "ML"
        elif market_key in {"spread", "ats"}:
            market_key = "ATS"
        elif market_key in {"total", "totals"}:
            market_key = "TOTAL"
        else:
            market_key = str(market).upper() or "UNKNOWN"

        p_raw = pd.to_numeric(
            sub.get("p_model_raw", sub.get("model_prob", sub.get("p_model_used"))),
            errors="coerce",
        ).astype(float)
        y = (sub["result_norm"] == "WIN").astype(float).to_numpy()

        mask = np.isfinite(p_raw.to_numpy())
        p_raw = p_raw.to_numpy()[mask]
        y = y[mask]

        if p_raw.size < MIN_SAMPLES:
            print(
                f"[calibration] WARNING: {sport_key} {market_key} only {p_raw.size} samples; "
                "skipping calibration"
            )
            markets[market_key] = MarketCalibrator(
                a=1.0,
                b=0.0,
                n_samples=int(p_raw.size),
                status="fallback",
            )
            continue

        p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)
        x = np.log(p_raw / (1.0 - p_raw))
        fitted = _fit_platt_scaling(x, y)
        if fitted is None:
            markets[market_key] = MarketCalibrator(a=1.0, b=0.0, n_samples=int(p_raw.size), status="fallback")
        else:
            markets[market_key] = fitted

    calibrator = Calibrator(sport=sport_key, markets=markets, min_samples=MIN_SAMPLES)
    _CALIBRATOR_CACHE[sport_key] = calibrator

    os.makedirs(os.path.join("results", "calibration"), exist_ok=True)
    payload = {
        "sport": sport_key,
        "min_samples": MIN_SAMPLES,
        "markets": {
            key: {
                "a": cal.a,
                "b": cal.b,
                "n_samples": cal.n_samples,
                "status": cal.status,
            }
            for key, cal in markets.items()
        },
    }
    with open(_calibration_path(sport_key), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return calibrator


def calibrate_prob(
    sport_or_probs: Union[str, np.ndarray, list],
    p_raw_or_y: Optional[Union[float, np.ndarray, list]] = None,
    market_type: Optional[str] = None,
) -> Union[float, Tuple[PlattCalibrator, str]]:
    if isinstance(sport_or_probs, str):
        calibrator = load_calibrator(sport_or_probs)
        if calibrator is None:
            return float(p_raw_or_y) if p_raw_or_y is not None else float("nan")
        return calibrator.calibrate(float(p_raw_or_y), market_type=market_type)

    if p_raw_or_y is None:
        return float("nan")

    probs = np.asarray(sport_or_probs, dtype=float)
    ys = np.asarray(p_raw_or_y, dtype=float)
    calibrator = fit_platt(probs, ys)
    return calibrator, "platt"
