from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np


_UNCERTAINTY_CACHE: Dict[str, Dict[str, float]] = {}


def _uncertainty_path(sport: str, market: Optional[str] = None) -> str:
    sport_key = str(sport).lower()
    if market:
        market_key = str(market).lower()
        return os.path.join("results", f"prob_uncertainty_{sport_key}_{market_key}.json")
    return os.path.join("results", f"prob_uncertainty_{sport_key}.json")


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
        "n": int(p.size),
        "n_samples": int(p.size),
        "window": int(window),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_uncertainty(sport: str, data: Dict[str, float], *, market: Optional[str] = None) -> None:
    os.makedirs("results", exist_ok=True)
    path = _uncertainty_path(str(sport).lower(), market=market)
    payload = dict(data or {})
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        return
    cache_key = f"{str(sport).lower()}:{str(market).lower()}" if market else str(sport).lower()
    _UNCERTAINTY_CACHE[cache_key] = payload


def update_uncertainty(
    sport: str,
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    window: int = 120,
    min_samples: int = 30,
    market: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    return update_uncertainty_json(
        sport,
        probs,
        outcomes,
        window=window,
        min_samples=min_samples,
        market=market,
    )


def update_uncertainty_json(
    sport: str,
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    window: int = 120,
    min_samples: int = 30,
    market: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    data = compute_uncertainty(probs, outcomes, window=window, min_samples=min_samples)
    if data is None:
        return None
    if market:
        data["market"] = str(market).upper()
    data["sport"] = str(sport).lower()
    save_uncertainty(sport, data, market=market)
    return data


def load_uncertainty(sport: str, *, market: Optional[str] = None) -> Optional[Dict[str, float]]:
    sport_key = str(sport).lower()
    cache_key = f"{sport_key}:{str(market).lower()}" if market else sport_key
    if cache_key in _UNCERTAINTY_CACHE:
        return _UNCERTAINTY_CACHE[cache_key]

    path = _uncertainty_path(sport_key, market=market)
    if not os.path.exists(path):
        if market:
            path = _uncertainty_path(sport_key, market=None)
        if not os.path.exists(path):
            return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    _UNCERTAINTY_CACHE[cache_key] = payload
    return payload
