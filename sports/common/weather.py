# sports/common/weather_sources.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict

import requests

# Minimal-but-usable: NFL stadium-ish coordinates (approx).
# You can refine later; this is good enough to capture "wind matters" signal.
TEAM_TO_COORDS: Dict[str, Tuple[float, float]] = {
    "Arizona Cardinals": (33.5277, -112.2626),
    "Atlanta Falcons": (33.7573, -84.4008),
    "Baltimore Ravens": (39.2779, -76.6227),
    "Buffalo Bills": (42.7738, -78.7868),
    "Carolina Panthers": (35.2258, -80.8528),
    "Chicago Bears": (41.8623, -87.6167),
    "Cincinnati Bengals": (39.0955, -84.5161),
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

@dataclass
class GameWeather:
    temp_f: Optional[float] = None
    wind_mph: Optional[float] = None
    precip_prob: Optional[float] = None


def fetch_game_weather(
    home_team: str,
    game_dt_utc: datetime,
    *,
    timeout: int = 15,
) -> GameWeather:
    """
    Pull hourly forecast around kickoff time using Open-Meteo.
    If anything fails, returns None fields.
    """
    coords = TEAM_TO_COORDS.get(str(home_team))
    if not coords:
        return GameWeather()

    lat, lon = coords

    # Open-Meteo wants ISO date range; we’ll just request that day
    day = game_dt_utc.date().isoformat()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,precipitation_probability",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "UTC",
        "start_date": day,
        "end_date": day,
    }

    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        hourly = (data or {}).get("hourly") or {}
        times = hourly.get("time") or []
        temps = hourly.get("temperature_2m") or []
        winds = hourly.get("windspeed_10m") or []
        pprob = hourly.get("precipitation_probability") or []

        # pick hour closest to kickoff
        kick = game_dt_utc.replace(minute=0, second=0, microsecond=0).isoformat(timespec="hours")
        best_i = None
        best_abs = 10**9
        for i, t in enumerate(times):
            try:
                # string compare is enough for same-day hours; but compute absolute “hour distance”
                dt = datetime.fromisoformat(t)
                diff = abs(int((dt - game_dt_utc).total_seconds()))
                if diff < best_abs:
                    best_abs = diff
                    best_i = i
            except Exception:
                continue

        if best_i is None:
            return GameWeather()

        def _get(lst, i):
            try:
                return float(lst[i])
            except Exception:
                return None

        return GameWeather(
            temp_f=_get(temps, best_i),
            wind_mph=_get(winds, best_i),
            precip_prob=_get(pprob, best_i),
        )

    except Exception:
        return GameWeather()
