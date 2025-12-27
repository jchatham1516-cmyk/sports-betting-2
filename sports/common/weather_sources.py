# sports/common/weather_sources.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests

# ----------------------------
# Config / budgets
# ----------------------------
DEFAULT_TIMEOUT = 15

WEATHER_MAX_REQUESTS = int(os.getenv("WEATHER_MAX_REQUESTS", "20"))  # hard cap per run
WEATHER_SLEEP_S = float(os.getenv("WEATHER_SLEEP_S", "0.10"))        # spacing between calls
WEATHER_CACHE_PATH = os.getenv("WEATHER_CACHE_PATH", "results/weather_cache.json")

# If True, weather calls are skipped entirely (returns None)
WEATHER_DISABLE = os.getenv("WEATHER_DISABLE", "0") == "1"


@dataclass
class _Budget:
    limit: int
    used: int = 0
    hard_stop: bool = False

    def bump(self) -> None:
        self.used += 1
        if self.used > self.limit:
            self.hard_stop = True
            raise RuntimeError(f"[weather] Request budget exceeded: {self.used}>{self.limit}")


_BUDGET = _Budget(limit=WEATHER_MAX_REQUESTS)


def _load_cache() -> Dict[str, Any]:
    try:
        if not os.path.exists(WEATHER_CACHE_PATH):
            return {}
        with open(WEATHER_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(WEATHER_CACHE_PATH) or "results", exist_ok=True)
        with open(WEATHER_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _dt_from_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        # Odds API style ISO with Z
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _cache_key(home_team: str, commence_iso: str) -> str:
    return f"{home_team}__{commence_iso}".strip()


def _http_get_json(url: str, params: dict) -> Any:
    if WEATHER_DISABLE:
        return None
    if _BUDGET.hard_stop:
        return None

    _BUDGET.bump()
    r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    if r.status_code == 429:
        # quick backoff
        time.sleep(1.0)
        r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)

    r.raise_for_status()
    time.sleep(WEATHER_SLEEP_S)
    return r.json()


def _geocode_stadium(query: str) -> Optional[Tuple[float, float, str]]:
    """
    Uses Open-Meteo geocoding to get lat/lon from a text query.
    Returns (lat, lon, name) or None.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 1, "language": "en", "format": "json"}
    data = _http_get_json(url, params=params)
    if not isinstance(data, dict):
        return None
    results = data.get("results") or []
    if not results:
        return None
    r0 = results[0]
    try:
        lat = float(r0.get("latitude"))
        lon = float(r0.get("longitude"))
        name = str(r0.get("name") or query)
        return (lat, lon, name)
    except Exception:
        return None


def _nearest_hour_index(times: list, target_iso: str) -> Optional[int]:
    """
    times: list of ISO strings (Open-Meteo hourly time)
    target_iso: target time in ISO (UTC) like '2025-12-27T21:25:00Z' or similar.
    """
    try:
        tgt = _dt_from_iso(target_iso)
        if tgt is None:
            return None
        best_i = None
        best_abs = None
        for i, t in enumerate(times or []):
            try:
                dt = datetime.fromisoformat(str(t))
                # Open-Meteo returns local time in the requested timezone; we request UTC below
                dt = dt.replace(tzinfo=timezone.utc)
                d = abs((dt - tgt).total_seconds())
                if best_abs is None or d < best_abs:
                    best_abs = d
                    best_i = i
            except Exception:
                continue
        return best_i
    except Exception:
        return None


def fetch_game_weather(
    *,
    home_team: str,
    commence_time_iso: Optional[str],
) -> Optional[Dict[str, float]]:
    """
    Returns a dict like:
      {
        "temp_f": ...,
        "wind_mph": ...,
        "precip_mm": ...,
      }
    or None if unavailable.

    Notes:
    - We geocode using "<home_team> stadium" (no key required).
    - We request Open-Meteo hourly data in UTC and pick the hour closest to kickoff.
    - Cached in results/weather_cache.json.
    """
    if WEATHER_DISABLE:
        return None

    if not home_team or not commence_time_iso:
        return None

    cache = _load_cache()
    ck = _cache_key(home_team, commence_time_iso)
    if ck in cache:
        try:
            v = cache.get(ck)
            return v if isinstance(v, dict) else None
        except Exception:
            pass

    kickoff_utc = _dt_from_iso(commence_time_iso)
    if kickoff_utc is None:
        return None

    # 1) geocode
    geo = _geocode_stadium(f"{home_team} stadium")
    if geo is None:
        # fallback: just team name
        geo = _geocode_stadium(str(home_team))
    if geo is None:
        return None

    lat, lon, _place = geo

    # 2) weather (Open-Meteo)
    # We use archive endpoint for past dates, forecast for near future.
    # For simplicity: always use forecast API; Open-Meteo typically serves recent history too.
    wx_url = "https://api.open-meteo.com/v1/forecast"
    day = kickoff_utc.date().isoformat()

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,precipitation",
        "timezone": "UTC",
        "start_date": day,
        "end_date": day,
    }

    data = _http_get_json(wx_url, params=params)
    if not isinstance(data, dict):
        return None

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps_c = hourly.get("temperature_2m") or []
    winds_kmh = hourly.get("windspeed_10m") or []
    precip_mm = hourly.get("precipitation") or []

    idx = _nearest_hour_index(times, kickoff_utc.isoformat().replace("+00:00", "Z"))
    if idx is None:
        return None

    def _get(lst, i, default=float("nan")):
        try:
            return float(lst[i])
        except Exception:
            return default

    temp_c = _get(temps_c, idx)
    wind_kmh = _get(winds_kmh, idx)
    pr_mm = _get(precip_mm, idx, default=0.0)

    # convert units
    temp_f = (temp_c * 9.0 / 5.0) + 32.0 if temp_c == temp_c else float("nan")
    wind_mph = wind_kmh * 0.621371 if wind_kmh == wind_kmh else float("nan")

    out = {
        "temp_f": float(temp_f),
        "wind_mph": float(wind_mph),
        "precip_mm": float(pr_mm),
    }

    cache[ck] = out
    _save_cache(cache)
    return out
