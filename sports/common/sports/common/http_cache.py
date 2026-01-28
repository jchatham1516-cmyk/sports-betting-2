# sports/common/http_cache.py
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

CACHE_DIR = os.getenv("ODDS_CACHE_DIR", "results/odds_cache")
DEFAULT_TTL_S = int(os.getenv("ODDS_CACHE_TTL_S", "21600"))  # 6 hours
SHORT_TTL_S = int(os.getenv("ODDS_CACHE_SHORT_TTL_S", "900"))  # 15 minutes
SHORT_TTL_WINDOW_S = int(os.getenv("ODDS_CACHE_SHORT_TTL_WINDOW_S", "10800"))  # 3 hours


def _parse_event_start(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return float(value.replace(tzinfo=timezone.utc).timestamp())
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _dynamic_ttl(ttl_s: int, *, event_start: Optional[object]) -> int:
    if not event_start:
        return int(ttl_s)
    event_ts = _parse_event_start(event_start)
    if event_ts is None:
        return int(ttl_s)
    now = time.time()
    seconds_to_start = event_ts - now
    if 0 <= seconds_to_start <= float(SHORT_TTL_WINDOW_S):
        return int(min(ttl_s, SHORT_TTL_S))
    return int(ttl_s)

def _key(url: str, params: Dict[str, Any]) -> str:
    raw = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def cached_get_json(
    url: str,
    params: Dict[str, Any],
    *,
    ttl_s: int = DEFAULT_TTL_S,
    timeout: int = 20,
    allow_stale_on_error: bool = True,
    event_start: Optional[object] = None,
) -> Any:
    os.makedirs(CACHE_DIR, exist_ok=True)
    k = _key(url, params)
    path = os.path.join(CACHE_DIR, f"{k}.json")

    now = time.time()
    ttl_s = _dynamic_ttl(ttl_s, event_start=event_start)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ts = float(payload.get("_ts", 0))
            if now - ts <= ttl_s:
                return payload.get("data")
        except Exception:
            pass

    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_ts": now, "data": data}, f)
        return data
    except Exception:
        if allow_stale_on_error and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                return payload.get("data")
            except Exception:
                pass
        raise
