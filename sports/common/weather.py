# sports/common/weather.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import time
import requests


# NOTE:
# - Uses Open-Meteo (no API key) for forecast.
# - Returns conservative, simple features to adjust totals.
# - If the call fails for any reason, we safely return None and your model continues.

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT = 12


@dataclass
class WeatherInfo:
    temp_f: float
    wind_mph: float
    precip_mm: float


# Approx stadium / city lat/lon for each team (good enough for totals adjustments)
# You can refine later if you want dome/roof logic too.
TEAM_TO_LATLON: Dict[str, Tuple[float, float]] = {
    "Arizona Cardinals": (33.5275, -112.2626),
    "Atlanta Falcons": (33.7553, -84.4006),
    "Baltimore Ravens": (39.2779, -76.6227),
    "Buffalo Bills": (42.7738, -78.7870),
    "Carolina Panthers": (35.2258, -80.8528),
    "Chicago Bears": (41.8623, -87.6167),
    "Cincinnati Bengals": (39.0954, -84.5160),
    "Cleveland Browns": (41.5061, -81.6995),
    "Dallas Cowboys": (32.7473, -97.0945),
    "Denver Broncos": (39.7439, -105.0201),
    "Detroit Lions": (42.3400, -83.0456),
    "Green Bay Packers": (44.5013, -88.0622),
    "Houston Texans": (29.6847, -95.4107),
    "Indianapolis Colts": (39.7601, -86.1639),
    "Jacksonville Jaguars": (30.3239, -81.6373),
    "Kansas City Chiefs": (39.0489, -94.4839),
    "Las Vegas Raiders": (36.0908, -115.1830),
    "Los Angeles Chargers": (33.9535, -118.3392),
    "Los Angeles Rams": (33.9535, -118.3392),
    "Miami Dolphins": (25.9580, -80.2389),
    "Minnesota Vikings": (44.9738, -93.2581),
    "New England Patriots": (42.0909, -71.2643),
    "New Orleans Saints": (29.9511, -90.0812),
    "New York Giants": (40.8135, -74.0745),
    "New York Jets": (40.8135, -74.0745),
    "Philadelphia Eagles": (39.9008, -75.1675),
    "Pittsburgh Steelers": (40.4468, -80.0158),
    "San Francisco 49ers": (37.4030, -121.9700),
    "Seattle Seahawks": (47.5952, -122.3316),
    "Tampa Bay Buccaneers": (27.9759, -82.5033),
    "Tennessee Titans": (36.1665, -86.7713),
    "Washington Commanders": (38.9078, -76.8645),
}


def _mph_from_ms(ms: float) -> float:
    return float(ms) * 2.23693629


def fetch_game_weather(
    *,
    home_team: str,
    kickoff_iso_utc: Optional[str] = None,
    sleep_s: float = 0.0,
) -> Optional[WeatherInfo]:
    """
    Very small, robust weather fetch.
    If kickoff_iso_utc is provided, we try to sample around that hour.
    If not, we just grab the next-available forecast hour.

    Returns WeatherInfo or None.
    """
    latlon = TEAM_TO_LATLON.get(str(home_team))
    if not latlon:
        return None

    lat, lon = latlon
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,windspeed_10m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "mm",
        "timezone": "UTC",
    }

    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        hourly = (data or {}).get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        precs = hourly.get("precipitation") or []
        winds = hourly.get("windspeed_10m") or []

        if not times or not temps or not winds:
            return None

        # pick hour
        idx = 0
        if kickoff_iso_utc:
            # find closest hour
            best = None
            for i, t in enumerate(times):
                if not isinstance(t, str):
                    continue
                # same date-hour string match is good enough here
                if t[:13] == str(kickoff_iso_utc)[:13]:
                    best = i
                    break
            if best is not None:
                idx = best

        temp_f = float(temps[idx]) if idx < len(temps) else float(temps[0])
        wind_mph = float(winds[idx]) if idx < len(winds) else float(winds[0])
        precip_mm = float(precs[idx]) if idx < len(precs) else 0.0

        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))

        return WeatherInfo(temp_f=temp_f, wind_mph=wind_mph, precip_mm=precip_mm)

    except Exception:
        return None


def totals_weather_adjustment_points(w: Optional[WeatherInfo]) -> float:
    """
    Conservative heuristic:
      - Wind is biggest.
      - Cold also matters.
      - Precip matters a bit.
    Negative means LOWER expected total.
    """
    if w is None:
        return 0.0

    adj = 0.0

    # Wind
    if w.wind_mph >= 20:
        adj -= 4.0
    elif w.wind_mph >= 15:
        adj -= 2.5
    elif w.wind_mph >= 12:
        adj -= 1.5

    # Cold
    if w.temp_f <= 25:
        adj -= 2.0
    elif w.temp_f <= 32:
        adj -= 1.0
    elif w.temp_f <= 40:
        adj -= 0.5

    # Precip (mm per hour)
    if w.precip_mm >= 2.0:
        adj -= 1.0
    elif w.precip_mm >= 0.5:
        adj -= 0.5

    return float(adj)
