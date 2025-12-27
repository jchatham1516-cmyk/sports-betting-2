# sports/common/weather_sources.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 15
CACHE_PATH = "results/weather_cache.json"

# Open-Meteo (free, no key)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Lat/Lon for NFL stadium cities (approx). Good enough for totals/wind adjustments.
# Keys should match your canon_team() outputs for NFL.
NFL_TEAM_LATLON: Dict[str, Tuple[float, float]] = {
    "Arizona Cardinals": (33.5275, -112.2626),   # Glendale, AZ
    "Atlanta Falcons": (33.7550, -84.4006),      # Atlanta, GA
    "Baltimore Ravens": (39.2779, -76.6227),     # Baltimore, MD
    "Buffalo Bills": (42.7738, -78.7869),        # Orchard Park, NY
    "Carolina Panthers": (35.2258, -80.8528),    # Charlotte, NC
    "Chicago Bears": (41.8623, -87.6167),        # Chicago, IL
    "Cincinnati Bengals": (39.0954, -84.5160),   # Cincinnati, OH
    "Cleveland Browns": (41.5061, -81.6995),     # Cleveland, OH
    "Dallas Cowboys": (32.7473, -97.0945),       # Arlington, TX
    "Denver Broncos": (39.7439, -105.0201),      # Denver, CO
    "Detroit Lions": (42.3400, -83.0456),        # Detroit, MI
    "Green Bay Packers": (44.5013, -88.0622),    # Green Bay, WI
    "Houston Texans": (29.6847, -95.4107),       # Houston, TX
    "Indianapolis Colts": (39.7601, -86.1639),   # Indianapolis, IN
    "Jacksonville Jaguars": (30.3239, -81.6373), # Jacksonville, FL
    "Kansas City Chiefs": (39.0489, -94.4839),   # Kansas City, MO
    "Las Vegas Raiders": (36.0908, -115.1830),   # Las Vegas, NV
    "Los Angeles Chargers": (33.9535, -118.3392),# Inglewood, CA
    "Los Angeles Rams": (33.9535, -118.3392),    # Inglewood, CA
    "Miami Dolphins": (25.9580, -80.2389),       # Miami Gardens, FL
    "Minnesota Vikings": (44.9738, -93.2576),    # Minneapolis, MN
    "New England Patriots": (42.0909, -71.2643), # Foxborough, MA
    "New Orleans Saints": (29.9511, -90.0812),   # New Orleans, LA
    "New York Giants": (40.8135, -74.0745),      # East Rutherford, NJ
    "New York Jets": (40.8135, -74.0745),        # East Rutherford, NJ
    "Philadelphia Eagles": (39.9008, -75.1675),  # Philadelphia, PA
    "Pittsburgh Steelers": (40.4468, -80.0158),  # Pittsburgh, PA
    "San Francisco 49ers": (37.4030, -121.9700), # Santa Clara, CA
    "Seattle Seahawks": (47.5952, -122.3316),    # Seattle, WA
    "Tampa Bay Buccaneers": (27.9759, -82.5033), # Tampa, FL
    "Tennessee Titans": (36.1665, -86.7713),     # Nashville, TN
    "Washington Commanders": (38.9078, -76.8645),# Landover, MD
}

# If your canon_team uses "Washington Football Team" or "Washington Redskins" historically,
# you can alias those too:
NFL_TEAM_LATLON.setdefault("Washington Football Team", NFL_TEAM_LATLON["Washington Commanders"])
NFL_TEAM_LATLON.setdefault("Washington Redskins", NFL_TEAM_LATLON["Washington Commanders"])


def _ensure_cache_dir() -> None:
    os.makedirs("results", exist_ok=True)


def _load_cache() -> Dict[str, Any]:
    _ensure_cache_dir()
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    _ensure_cache_dir()
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_dt(dt_like: Any) -> Optional[datetime]:
    if dt_like is None:
        return None
    if isinstance(dt_like, datetime):
        return _to_utc(dt_like)
    s = str(dt_like)
    if not s:
        return None
    try:
        # handles "2025-12-27T21:20:00Z" etc
        return _to_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except Exception:
        return None


def _nearest_hour_index(times_iso: list, target_utc: datetime) -> Optional[int]:
    if not times_iso:
        return None
    best_i = None
    best_abs = None
    for i, t in enumerate(times_iso):
        try:
            dt = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            dt = _to_utc(dt)
            diff = abs((dt - target_utc).total_seconds())
            if best_abs is None or diff < best_abs:
                best_abs = diff
                best_i = i
        except Exception:
            continue
    return best_i


def fetch_game_weather(
    *,
    home_team: str,
    # Accept BOTH names so your existing model call won't crash:
    game_dt_utc: Any = None,
    game_dt: Any = None,
    kickoff_dt: Any = None,
) -> Dict[str, Any]:
    """
    Returns dict with keys:
      - temp_f
      - wind_mph
      - precip_mm
      - source
      - ok (bool)
    Safe: if anything fails, returns ok=False with NaNs.

    NOTE: This function intentionally accepts game_dt_utc to match your NFL model call.
    """

    # Pick whichever datetime arg was provided
    dt = _parse_dt(game_dt_utc) or _parse_dt(game_dt) or _parse_dt(kickoff_dt)
    if dt is None:
        return {"ok": False, "temp_f": float("nan"), "wind_mph": float("nan"), "precip_mm": float("nan"), "source": "none", "reason": "missing_datetime"}

    latlon = NFL_TEAM_LATLON.get(str(home_team))
    if not latlon:
        return {"ok": False, "temp_f": float("nan"), "wind_mph": float("nan"), "precip_mm": float("nan"), "source": "none", "reason": f"unknown_team:{home_team}"}

    lat, lon = latlon
    dt = _to_utc(dt)

    # Cache by team + hour
    cache = _load_cache()
    cache_key = f"nfl|{home_team}|{dt.strftime('%Y-%m-%dT%H:00Z')}"
    if cache_key in cache:
        return cache[cache_key]

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,precipitation",
        "timezone": "UTC",
        # keep it small-ish; same day is enough
        "start_date": dt.strftime("%Y-%m-%d"),
        "end_date": dt.strftime("%Y-%m-%d"),
    }

    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        temps_c = hourly.get("temperature_2m") or []
        wind_kmh = hourly.get("windspeed_10m") or []
        precip_mm = hourly.get("precipitation") or []

        i = _nearest_hour_index(times, dt)
        if i is None:
            out = {"ok": False, "temp_f": float("nan"), "wind_mph": float("nan"), "precip_mm": float("nan"), "source": "open-meteo", "reason": "no_hour_match"}
        else:
            # Convert units
            t_c = float(temps_c[i]) if i < len(temps_c) else float("nan")
            w_kmh = float(wind_kmh[i]) if i < len(wind_kmh) else float("nan")
            p_mm = float(precip_mm[i]) if i < len(precip_mm) else float("nan")

            t_f = (t_c * 9.0 / 5.0) + 32.0 if t_c == t_c else float("nan")
            w_mph = w_kmh * 0.621371 if w_kmh == w_kmh else float("nan")

            out = {
                "ok": True,
                "temp_f": float(t_f),
                "wind_mph": float(w_mph),
                "precip_mm": float(p_mm),
                "source": "open-meteo",
            }

    except Exception as e:
        out = {"ok": False, "temp_f": float("nan"), "wind_mph": float("nan"), "precip_mm": float("nan"), "source": "open-meteo", "reason": f"error:{e}"}

    cache[cache_key] = out
    _save_cache(cache)
    return out
