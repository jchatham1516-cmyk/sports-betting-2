from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sports.common.util import clamp, normalize_result_label, safe_float


MIN_SAMPLES = 300
_CALIBRATOR_CACHE: Dict[str, "PlattCalibrator"] = {}
ML_ODDS_BUCKETS = ("favorite", "medium", "dog", "longshot")


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


def _calibration_key(sport: str, market_type: Optional[str], bucket: Optional[str]) -> str:
    sport_key = str(sport).lower()
    if market_type:
        market_key = str(market_type).lower()
        suffix = f"{market_key}"
        if bucket:
            suffix = f"{suffix}_{bucket}"
        return f"{sport_key}:{suffix}"
    return f"{sport_key}:ml"


def _calibration_path(sport: str, market_type: Optional[str] = None, bucket: Optional[str] = None) -> str:
    sport_key = str(sport).lower()
    if market_type:
        market_key = str(market_type).lower()
        suffix = f"{market_key}"
        if bucket:
            suffix = f"{suffix}_{bucket}"
        return os.path.join("results", f"prob_cal_{sport_key}_{suffix}.json")
    return os.path.join("results", f"prob_cal_{sport_key}.json")


def _legacy_calibration_path(sport: str) -> str:
    return os.path.join("results", "calibration", f"platt_{sport}.json")


def _history_dir() -> str:
    return os.path.join("results", "calibration_history")


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
    *,
    market_type: Optional[str] = None,
    bucket: Optional[str] = None,
    persist_snapshot: bool = True,
    days_back: Optional[int] = None,
) -> None:
    os.makedirs("results", exist_ok=True)
    payload = {
        "a": float(params.get("a", 1.0)),
        "b": float(params.get("b", 0.0)),
        "n_samples": int(n_samples),
        "window": int(window) if window is not None else None,
        "method": str(method),
        "market_type": str(market_type).upper() if market_type else None,
        "bucket": str(bucket) if bucket else None,
        "days_back": int(days_back) if days_back is not None else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(_calibration_path(str(sport).lower(), market_type=market_type, bucket=bucket), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if persist_snapshot:
        os.makedirs(_history_dir(), exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"{str(sport).lower()}"
        if market_type:
            snapshot_name = f"{snapshot_name}_{str(market_type).lower()}"
        if bucket:
            snapshot_name = f"{snapshot_name}_{bucket}"
        snapshot_path = os.path.join(_history_dir(), f"{snapshot_name}_{ts}.json")
        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            return


def load_calibrator(
    sport: str,
    *,
    market_type: Optional[str] = None,
    bucket: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    cache_key = _calibration_key(sport, market_type, bucket)
    if cache_key in _CALIBRATOR_CACHE:
        cal = _CALIBRATOR_CACHE[cache_key]
        return {"a": cal.a, "b": cal.b, "n_samples": int(cal.n_samples)}

    sport_key = str(sport).lower()
    path = _calibration_path(sport_key, market_type=market_type, bucket=bucket)
    legacy_path = _legacy_calibration_path(sport_key) if not market_type and not bucket else None
    if not os.path.exists(path):
        if legacy_path is None or not os.path.exists(legacy_path):
            return None
        path = legacy_path
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    cal = PlattCalibrator.from_dict(payload)
    _CALIBRATOR_CACHE[cache_key] = cal
    return {
        "a": cal.a,
        "b": cal.b,
        "n_samples": int(payload.get("n_samples", cal.n_samples)),
        "window": payload.get("window"),
        "method": payload.get("method"),
        "market_type": payload.get("market_type"),
        "bucket": payload.get("bucket"),
    }


def fit_calibrator(
    sport: str,
    rows_df: pd.DataFrame,
    *,
    market_type: Optional[str] = None,
) -> Optional[PlattCalibrator]:
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

    if market_type:
        market_label = str(market_type).lower()
        market_cols = [c for c in ("market_type", "market", "market_label") if c in work.columns]
        if market_cols:
            for col in market_cols:
                work = work[work[col].astype(str).str.lower().str.contains(market_label)]

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
    _CALIBRATOR_CACHE[_calibration_key(sport_key, market_type, None)] = calibrator
    save_calibrator(sport_key, params, n_samples=int(p_raw.size), market_type=market_type)
    return calibrator


def update_prob_calibration(
    sport: str,
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    window: int = 300,
    min_samples: int = MIN_SAMPLES,
    market_type: Optional[str] = None,
    bucket: Optional[str] = None,
    days_back: Optional[int] = None,
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
    _CALIBRATOR_CACHE[_calibration_key(str(sport).lower(), market_type, bucket)] = calibrator
    save_calibrator(
        sport,
        params,
        n_samples=int(p_raw.size),
        window=int(window),
        method="platt",
        market_type=market_type,
        bucket=bucket,
        days_back=days_back,
    )
    return calibrator


def odds_bucket_from_price(price: Optional[float]) -> Optional[str]:
    if price is None or not np.isfinite(price):
        return None
    price_val = float(price)
    if price_val <= -170:
        return "favorite"
    if price_val <= -115:
        return "medium"
    if price_val < 120:
        return "dog"
    if price_val >= 250:
        return "longshot"
    return "dog"


def calibrate_prob(
    sport_or_probs: Union[str, np.ndarray, list],
    p_raw_or_y: Optional[Union[float, np.ndarray, list]] = None,
    market_type: Optional[str] = None,
    price: Optional[float] = None,
    odds_bucket: Optional[str] = None,
) -> Union[float, Tuple[PlattCalibrator, str]]:
    if isinstance(sport_or_probs, str):
        bucket = odds_bucket
        if bucket is None and market_type and str(market_type).upper() == "ML":
            bucket = odds_bucket_from_price(price)
        params = load_calibrator(sport_or_probs, market_type=market_type, bucket=bucket)
        if params is None and bucket is not None:
            params = load_calibrator(sport_or_probs, market_type=market_type, bucket=None)
        if params is None:
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
    market_type: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames: List[pd.DataFrame] = []
    for path in bet_log_paths:
        if not os.path.exists(path):
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue

    if not frames:
        return np.array([]), np.array([]), np.array([])

    bets = pd.concat(frames, ignore_index=True)
    if bets.empty:
        return np.array([]), np.array([]), np.array([])

    bets = bets.copy()
    bets["sport"] = bets.get("sport", "").astype(str).str.lower()
    bets = bets[bets["sport"] == str(sport).lower()]
    if market_type:
        bets = bets[bets.get("market_type", "").astype(str).str.lower() == str(market_type).lower()]

    if "date" in bets.columns:
        bets["date_parsed"] = pd.to_datetime(bets["date"], errors="coerce")
        cutoff = datetime.utcnow() - timedelta(days=int(days_back))
        bets = bets[bets["date_parsed"] >= cutoff]

    bets["result_norm"] = bets.get("result", "").apply(normalize_result_label)
    bets = bets[bets["result_norm"].isin(["WIN", "LOSS"])]
    if bets.empty:
        return np.array([]), np.array([]), np.array([])

    def _pick_prob(row: pd.Series) -> float:
        for col in ("p_model_raw", "model_prob", "p_model_cal", "p_model_final"):
            val = safe_float(row.get(col))
            if val is not None and np.isfinite(val):
                return float(val)
        return float("nan")

    p_raw = bets.apply(_pick_prob, axis=1).astype(float)
    y = (bets["result_norm"] == "WIN").astype(float)

    mask = np.isfinite(p_raw.to_numpy()) & np.isfinite(y.to_numpy())
    price = pd.to_numeric(bets.get("price", np.nan), errors="coerce").astype(float)
    return p_raw.to_numpy()[mask], y.to_numpy()[mask], price.to_numpy()[mask]


def _load_eval_history_samples(
    sport: str,
    *,
    days_back: int,
    eval_history_path: str,
    market_type: Optional[str] = None,
    bucket: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(eval_history_path):
        return np.array([]), np.array([]), np.array([])
    try:
        hist = pd.read_csv(eval_history_path)
    except Exception:
        return np.array([]), np.array([]), np.array([])
    if hist.empty:
        return np.array([]), np.array([]), np.array([])
    hist = hist.copy()
    if "sport" in hist.columns:
        hist = hist[hist["sport"].astype(str).str.lower() == str(sport).lower()]
    if market_type:
        hist = hist[hist.get("market_type", "").astype(str).str.upper() == str(market_type).upper()]
    if "date" in hist.columns:
        hist["date_parsed"] = pd.to_datetime(hist["date"], errors="coerce")
        cutoff = datetime.utcnow() - timedelta(days=int(days_back))
        hist = hist[hist["date_parsed"] >= cutoff]
    p_raw = pd.to_numeric(hist.get("model_prob_raw", hist.get("model_prob")), errors="coerce").astype(float)
    y = pd.to_numeric(hist.get("actual_result"), errors="coerce").astype(float)
    price = pd.to_numeric(hist.get("price"), errors="coerce").astype(float)
    if bucket:
        bucket_mask = price.apply(lambda val: odds_bucket_from_price(val) == bucket)
        hist = hist[bucket_mask]
        p_raw = p_raw[bucket_mask]
        y = y[bucket_mask]
        price = price[bucket_mask]
    mask = np.isfinite(p_raw.to_numpy()) & np.isfinite(y.to_numpy())
    return p_raw.to_numpy()[mask], y.to_numpy()[mask], price.to_numpy()[mask]


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


def update_daily_market_calibration(
    sport: str,
    *,
    market_type: str,
    days_back: int = 45,
    min_samples: int = MIN_SAMPLES,
    bet_log_paths: Optional[List[str]] = None,
    eval_history_path: Optional[str] = None,
    bucket: Optional[str] = None,
    window: Optional[int] = None,
) -> Optional[PlattCalibrator]:
    sport_key = str(sport).lower()
    bet_log_paths = bet_log_paths or ["results/tracking/bet_log.csv", "results/bet_log.csv"]
    eval_history_path = eval_history_path or "results/eval_history.csv"

    p_raw_log, y_log, _ = _load_bet_log_samples(
        sport_key,
        days_back=days_back,
        bet_log_paths=bet_log_paths,
        market_type=market_type,
    )
    p_raw_hist, y_hist, _ = _load_eval_history_samples(
        sport_key,
        days_back=days_back,
        eval_history_path=eval_history_path,
        market_type=market_type,
        bucket=bucket,
    )

    if p_raw_log.size and p_raw_hist.size:
        p_raw = np.concatenate([p_raw_log, p_raw_hist])
        y = np.concatenate([y_log, y_hist])
    elif p_raw_log.size:
        p_raw, y = p_raw_log, y_log
    else:
        p_raw, y = p_raw_hist, y_hist

    if p_raw.size < int(min_samples):
        print(
            f"[calibration] {sport_key} {market_type} has only {p_raw.size} samples; skipping update."
        )
        return None

    use_window = int(window or p_raw.size)
    return update_prob_calibration(
        sport_key,
        p_raw,
        y,
        window=use_window,
        min_samples=int(min_samples),
        market_type=market_type,
        bucket=bucket,
        days_back=days_back,
    )


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
    eval_history_path = eval_history_path or "results/eval_history.csv"

    best = update_daily_market_calibration(
        sport_key,
        market_type="ML",
        days_back=days_back,
        min_samples=min_samples,
        bet_log_paths=bet_log_paths,
        eval_history_path=eval_history_path,
    )
    for bucket in ML_ODDS_BUCKETS:
        update_daily_market_calibration(
            sport_key,
            market_type="ML",
            days_back=days_back,
            min_samples=min_samples,
            bet_log_paths=bet_log_paths,
            eval_history_path=eval_history_path,
            bucket=bucket,
        )
    return best
