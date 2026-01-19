from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sports.common.util import clamp, normalize_result_label, safe_float


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
    return os.path.join("results", f"prob_cal_{sport}.json")


def _legacy_calibration_path(sport: str) -> str:
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


def save_calibrator(
    sport: str,
    params: Dict[str, float],
    n_samples: int = 0,
    window: Optional[int] = None,
    method: str = "platt",
) -> None:
    os.makedirs("results", exist_ok=True)
    payload = {
        "a": float(params.get("a", 1.0)),
        "b": float(params.get("b", 0.0)),
        "n_samples": int(n_samples),
        "window": int(window) if window is not None else None,
        "method": str(method),
    }
    with open(_calibration_path(str(sport).lower()), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_calibrator(sport: str) -> Optional[Dict[str, float]]:
    sport_key = str(sport).lower()
    if sport_key in _CALIBRATOR_CACHE:
        cal = _CALIBRATOR_CACHE[sport_key]
        return {"a": cal.a, "b": cal.b}

    path = _calibration_path(sport_key)
    legacy_path = _legacy_calibration_path(sport_key)
    if not os.path.exists(path):
        if not os.path.exists(legacy_path):
            return None
        path = legacy_path
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


def update_prob_calibration(
    sport: str,
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    window: int = 300,
    min_samples: int = MIN_SAMPLES,
) -> Optional[PlattCalibrator]:
    p_raw = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p_raw) & np.isfinite(y)
    p_raw = p_raw[mask]
    y = y[mask]

    if window and window > 0 and p_raw.size > window:
        p_raw = p_raw[-window:]
        y = y[-window:]

    if p_raw.size < int(min_samples):
        return None

    params = fit_platt(p_raw, y)
    calibrator = PlattCalibrator(a=params["a"], b=params["b"], n_samples=int(p_raw.size))
    _CALIBRATOR_CACHE[str(sport).lower()] = calibrator
    save_calibrator(sport, params, n_samples=int(p_raw.size), window=int(window), method="platt")
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


def _load_bet_log_samples(
    sport: str,
    *,
    days_back: int,
    bet_log_paths: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    frames: List[pd.DataFrame] = []
    for path in bet_log_paths:
        if not os.path.exists(path):
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue

    if not frames:
        return np.array([]), np.array([])

    bets = pd.concat(frames, ignore_index=True)
    if bets.empty:
        return np.array([]), np.array([])

    bets = bets.copy()
    bets["sport"] = bets.get("sport", "").astype(str).str.lower()
    bets = bets[bets["sport"] == str(sport).lower()]
    bets = bets[bets.get("market_type", "").astype(str).str.lower() == "moneyline"]

    if "date" in bets.columns:
        bets["date_parsed"] = pd.to_datetime(bets["date"], errors="coerce")
        cutoff = datetime.utcnow() - timedelta(days=int(days_back))
        bets = bets[bets["date_parsed"] >= cutoff]

    bets["result_norm"] = bets.get("result", "").apply(normalize_result_label)
    bets = bets[bets["result_norm"].isin(["WIN", "LOSS"])]
    if bets.empty:
        return np.array([]), np.array([])

    def _pick_prob(row: pd.Series) -> float:
        for col in ("p_model_raw", "model_prob", "p_model_cal", "p_model_final"):
            val = safe_float(row.get(col))
            if val is not None and np.isfinite(val):
                return float(val)
        return float("nan")

    p_raw = bets.apply(_pick_prob, axis=1).astype(float)
    y = (bets["result_norm"] == "WIN").astype(float)

    mask = np.isfinite(p_raw.to_numpy()) & np.isfinite(y.to_numpy())
    return p_raw.to_numpy()[mask], y.to_numpy()[mask]


def _load_prediction_samples(
    sport: str,
    *,
    days_back: int,
    preds_dir: str,
    eval_history_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sports.common.eval import update_eval_history_with_scores
    except Exception:
        return np.array([]), np.array([])

    merged = update_eval_history_with_scores(
        sport=sport,
        preds_dir=preds_dir,
        out_path=eval_history_path,
        days_back=days_back,
    )
    if merged.empty:
        return np.array([]), np.array([])

    if "score_date" in merged.columns:
        merged = merged.copy()
        merged["score_date"] = pd.to_datetime(merged["score_date"], errors="coerce")
        cutoff = datetime.utcnow() - timedelta(days=int(days_back))
        merged = merged[merged["score_date"] >= cutoff]

    p_raw = pd.to_numeric(merged.get("model_home_prob", np.nan), errors="coerce")
    y = pd.to_numeric(merged.get("actual_home_win", np.nan), errors="coerce")
    mask = np.isfinite(p_raw.to_numpy()) & np.isfinite(y.to_numpy())
    return p_raw.to_numpy()[mask], y.to_numpy()[mask]


def update_daily_ml_calibration(
    sport: str = "nba",
    *,
    days_back: int = 45,
    min_samples: int = MIN_SAMPLES,
    bet_log_paths: Optional[List[str]] = None,
    preds_dir: str = "results",
    eval_history_path: Optional[str] = None,
) -> Optional[PlattCalibrator]:
    sport_key = str(sport).lower()
    bet_log_paths = bet_log_paths or ["results/tracking/bet_log.csv", "results/bet_log.csv"]
    eval_history_path = eval_history_path or f"results/eval_history_{sport_key}.csv"

    p_raw_log, y_log = _load_bet_log_samples(sport_key, days_back=days_back, bet_log_paths=bet_log_paths)
    p_raw_pred, y_pred = _load_prediction_samples(
        sport_key,
        days_back=days_back,
        preds_dir=preds_dir,
        eval_history_path=eval_history_path,
    )

    if p_raw_log.size and p_raw_pred.size:
        p_raw = np.concatenate([p_raw_log, p_raw_pred])
        y = np.concatenate([y_log, y_pred])
    elif p_raw_log.size:
        p_raw, y = p_raw_log, y_log
    else:
        p_raw, y = p_raw_pred, y_pred

    if p_raw.size < int(min_samples):
        print(f"[calibration] {sport_key} has only {p_raw.size} samples; skipping ML calibration update.")
        return None

    return update_prob_calibration(
        sport_key,
        p_raw,
        y,
        window=int(p_raw.size),
        min_samples=int(min_samples),
    )
