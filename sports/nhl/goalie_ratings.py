from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from typing import Optional

import requests


STATS_CACHE_DIR = "results/cache"
NHL_STATS_BASE = "https://api.nhle.com/stats/rest/en/goalie"
NHL_STATS_LIMIT = 300
DEFAULT_LEAGUE_AVG_SV = 0.903


def _get_with_retry(url: str, *, params: Optional[dict] = None, timeout: int = 20, max_retries: int = 3) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code >= 500:
                raise RuntimeError(f"server error {resp.status_code}")
            if resp.status_code != 200:
                raise RuntimeError(f"unexpected status {resp.status_code}")
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2)
    raise RuntimeError(f"failed to fetch {url}") from last_exc


def _season_for_date(dt: date) -> str:
    # NHL season label uses starting year, e.g. 2024 season for 2024-2025.
    return str(dt.year - 1 if dt.month < 7 else dt.year)


def _normalize_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def _load_cached_stats(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_cached_stats(cache_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _fetch_goalie_stats(season: str) -> dict:
    cache_path = os.path.join(STATS_CACHE_DIR, f"nhl_goalie_stats_{season}.json")
    cached = _load_cached_stats(cache_path)
    if cached:
        return cached

    params = {
        "isAggregate": "false",
        "isGame": "false",
        "sort": "[{\"property\":\"savePct\",\"direction\":\"DESC\"}]",
        "start": 0,
        "limit": NHL_STATS_LIMIT,
        "factCayenneExp": "gamesPlayed>=5",
        "cayenneExp": f"seasonId={season}{int(season) + 1}",
    }
    try:
        payload = _get_with_retry(NHL_STATS_BASE, params=params)
    except Exception as exc:
        print(f"[nhl goalie ratings] WARNING: stats fetch failed: {exc}")
        _write_cached_stats(cache_path, {})
        return {}

    _write_cached_stats(cache_path, payload)
    return payload


def get_goalie_rating(goalie_name: str, season: str) -> float:
    if not goalie_name:
        return 0.0

    payload = _fetch_goalie_stats(season)
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not data:
        return 0.0

    name_norm = _normalize_name(goalie_name)
    best = None
    for row in data:
        row_name = _normalize_name(row.get("goalieFullName") or row.get("playerName") or "")
        if row_name == name_norm:
            best = row
            break

    if best is None:
        return 0.0

    sv_pct = best.get("savePct")
    games = best.get("gamesPlayed", 0)
    try:
        sv_pct = float(sv_pct)
        games = int(games)
    except Exception:
        return 0.0

    league_avg = DEFAULT_LEAGUE_AVG_SV
    rating = (sv_pct - league_avg) * 1000.0
    rating = max(-30.0, min(30.0, rating))
    if games < 5:
        rating *= 0.5
    return float(rating)


def current_season_label() -> str:
    return _season_for_date(datetime.utcnow().date())
