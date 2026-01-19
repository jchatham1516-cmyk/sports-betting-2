from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np


_UNCERTAINTY_CACHE: Dict[str, Dict[str, float]] = {}


def _uncertainty_path(sport: str) -> str:
    return os.path.join("results", f"prob_uncertainty_{sport}.json")


def compute_uncertainty(
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    window: int = 120,
    min_samples: int = 30,
) -> Optional[Dict[str, float]]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]

    if p.size == 0:
        return None

    if window and window > 0 and p.size > window:
        p = p[-window:]
        y = y[-window:]

    if p.size < min_samples:
        return None

    errors = y - p
    brier = float(np.mean((p - y) ** 2))
    error_std = float(np.std(errors))
    uncertainty = float(max(error_std, math.sqrt(brier)))

    return {
        "brier": brier,
        "error_std": error_std,
        "uncertainty": uncertainty,
        "n_samples": int(p.size),
        "window": int(window),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_uncertainty(sport: str, data: Dict[str, float]) -> None:
    os.makedirs("results", exist_ok=True)
    path = _uncertainty_path(str(sport).lower())
    payload = dict(data or {})
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        return
    _UNCERTAINTY_CACHE[str(sport).lower()] = payload


def update_uncertainty(
    sport: str,
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    window: int = 120,
    min_samples: int = 30,
) -> Optional[Dict[str, float]]:
    data = compute_uncertainty(probs, outcomes, window=window, min_samples=min_samples)
    if data is None:
        return None
    save_uncertainty(sport, data)
    return data


def load_uncertainty(sport: str) -> Optional[Dict[str, float]]:
    sport_key = str(sport).lower()
    if sport_key in _UNCERTAINTY_CACHE:
        return _UNCERTAINTY_CACHE[sport_key]

    path = _uncertainty_path(sport_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    _UNCERTAINTY_CACHE[sport_key] = payload
    return payload
