from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sports.common.eval import build_game_key


ATS_MIN_SAMPLES = 200
_ATS_CAL_CACHE: Dict[str, "ATSCalibrator"] = {}


@dataclass
class ATSCalibrator:
    a: float
    b: float
    n_samples: int = 0

    def predict_prob_cover(self, x: float) -> float:
        if x is None or not np.isfinite(x):
            return float("nan")
        z = float(self.a) * float(x) + float(self.b)
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b), "n_samples": int(self.n_samples)}

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ATSCalibrator":
        return ATSCalibrator(
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            n_samples=int(data.get("n_samples", 0)),
        )


def _ats_calibration_path(sport: str) -> str:
    sport_key = str(sport).lower()
    return os.path.join("results", f"ats_cal_{sport_key}.json")


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    if x.size == 0:
        return None

    a, b = 0.0, 0.0
    reg = 1e-4

    for _ in range(60):
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


def fit_ats_calibrator(samples: Sequence[Tuple[float, float]]) -> Optional[ATSCalibrator]:
    if not samples:
        return None
    xs = np.asarray([s[0] for s in samples], dtype=float)
    ys = np.asarray([s[1] for s in samples], dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    fitted = _fit_logistic(xs, ys)
    if fitted is None:
        return None
    a, b = fitted
    return ATSCalibrator(a=a, b=b, n_samples=int(xs.size))


def save_ats_calibrator(sport: str, calibrator: ATSCalibrator) -> None:
    os.makedirs("results", exist_ok=True)
    with open(_ats_calibration_path(sport), "w", encoding="utf-8") as f:
        json.dump(calibrator.to_dict(), f, indent=2)


def load_ats_calibrator(sport: str) -> Optional[ATSCalibrator]:
    sport_key = str(sport).lower()
    if sport_key in _ATS_CAL_CACHE:
        return _ATS_CAL_CACHE[sport_key]
    path = _ats_calibration_path(sport_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    cal = ATSCalibrator.from_dict(payload)
    if int(cal.n_samples) < int(ATS_MIN_SAMPLES):
        return None
    _ATS_CAL_CACHE[sport_key] = cal
    return cal


def _load_predictions_history(sport: str, preds_dirs: Iterable[str]) -> pd.DataFrame:
    sport_key = str(sport).lower()
    frames: List[pd.DataFrame] = []
    for preds_dir in preds_dirs:
        if not preds_dir:
            continue
        for path in Path(preds_dir).glob(f"predictions_{sport_key}_*.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df = df.copy()
            df["__source_file"] = str(path)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    if "game_key" not in out.columns:
        out["game_key"] = out.apply(
            lambda r: build_game_key(r.get("event_id"), r.get("date"), r.get("home"), r.get("away")),
            axis=1,
        )
    else:
        out["game_key"] = out["game_key"].fillna(
            out.apply(
                lambda r: build_game_key(r.get("event_id"), r.get("date"), r.get("home"), r.get("away")),
                axis=1,
            )
        )

    if "model_margin_home" in out.columns:
        out["model_margin_home_used"] = pd.to_numeric(out["model_margin_home"], errors="coerce")
    elif "model_spread_home" in out.columns:
        out["model_margin_home_used"] = -pd.to_numeric(out["model_spread_home"], errors="coerce")
    else:
        out["model_margin_home_used"] = np.nan

    out["home_spread_used"] = pd.to_numeric(out.get("home_spread"), errors="coerce")

    return out[["game_key", "model_margin_home_used", "home_spread_used"]]


def _load_bet_log_history(paths: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_ats_training_samples(
    *,
    sport: str = "nba",
    bet_log_paths: Optional[Iterable[str]] = None,
    preds_dirs: Optional[Iterable[str]] = None,
) -> List[Tuple[float, float]]:
    bet_log_paths = bet_log_paths or ["results/tracking/bet_log.csv", "results/bet_log.csv"]
    preds_dirs = preds_dirs or ["results", "results/tracking"]

    bet_log = _load_bet_log_history(bet_log_paths)
    if bet_log.empty:
        return []

    market_col = "market_type" if "market_type" in bet_log.columns else "market"
    bet_log = bet_log.copy()
    bet_log["sport_lower"] = bet_log.get("sport", "").astype(str).str.lower()
    bet_log["market_lower"] = bet_log.get(market_col, "").astype(str).str.lower()
    bet_log = bet_log[(bet_log["sport_lower"] == str(sport).lower()) & (bet_log["market_lower"] == "spread")]
    if bet_log.empty:
        return []

    bet_log["game_key"] = bet_log.apply(
        lambda r: build_game_key(r.get("event_id"), r.get("date"), r.get("home"), r.get("away")),
        axis=1,
    )

    preds = _load_predictions_history(sport, preds_dirs)
    if preds.empty:
        return []

    merged = bet_log.merge(preds, on="game_key", how="left")

    line_col = "line_at_bet" if "line_at_bet" in merged.columns else "line"
    merged["market_spread"] = pd.to_numeric(merged.get(line_col), errors="coerce")
    merged["market_spread"] = merged["market_spread"].fillna(merged["home_spread_used"])
    merged["model_margin"] = pd.to_numeric(merged["model_margin_home_used"], errors="coerce")

    home_score = pd.to_numeric(merged.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(merged.get("away_score"), errors="coerce")
    margin = home_score - away_score

    samples: List[Tuple[float, float]] = []
    for x, hs, m in zip(
        (merged["model_margin"] + merged["market_spread"]).to_numpy(),
        (margin + merged["market_spread"]).to_numpy(),
        merged["market_spread"].to_numpy(),
    ):
        if not np.isfinite(x) or not np.isfinite(hs) or not np.isfinite(m):
            continue
        if abs(float(hs)) < 1e-6:
            continue
        y = 1.0 if float(hs) > 0 else 0.0
        samples.append((float(x), float(y)))

    return samples


def train_ats_calibrator_from_history(
    *,
    sport: str = "nba",
    bet_log_paths: Optional[Iterable[str]] = None,
    preds_dirs: Optional[Iterable[str]] = None,
    min_samples: int = ATS_MIN_SAMPLES,
) -> Optional[ATSCalibrator]:
    samples = build_ats_training_samples(
        sport=sport, bet_log_paths=bet_log_paths, preds_dirs=preds_dirs
    )
    if len(samples) < int(min_samples):
        print(
            f"[ats calibration] {sport} only {len(samples)} samples; "
            f"need {int(min_samples)} to train."
        )
        return None
    calibrator = fit_ats_calibrator(samples)
    if calibrator is None:
        return None
    save_ats_calibrator(sport, calibrator)
    _ATS_CAL_CACHE[str(sport).lower()] = calibrator
    return calibrator


def get_ats_calibrator(
    *,
    sport: str = "nba",
    min_samples: int = ATS_MIN_SAMPLES,
    bet_log_paths: Optional[Iterable[str]] = None,
    preds_dirs: Optional[Iterable[str]] = None,
) -> Optional[ATSCalibrator]:
    existing = load_ats_calibrator(sport)
    if existing is not None:
        return existing
    return train_ats_calibrator_from_history(
        sport=sport, bet_log_paths=bet_log_paths, preds_dirs=preds_dirs, min_samples=min_samples
    )
