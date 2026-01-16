from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import date, datetime
from typing import Optional

import requests


STATS_CACHE_DIR = "results/cache"
NHL_STATS_BASE = "https://api.nhle.com/stats/rest/en/goalie"
NHL_STATS_LIMIT = 300
DEFAULT_LEAGUE_AVG_SV = 0.903
_GOALIE_LOOKUP_CACHE: dict[str, dict[str, dict]] = {}
_GOALIE_LEAGUE_STATS_CACHE: dict[str, tuple[float, float, float]] = {}


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


def normalize_goalie_name(name: str) -> str:
    if not name:
        return ""
    cleaned = unicodedata.normalize("NFKD", str(name))
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.replace(".", " ").replace("-", " ").replace("’", "'")
    cleaned = re.sub(r"\(.*?\)", " ", cleaned)
    cleaned = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", " , ")
    cleaned = re.sub(r"[^\w\s,']", " ", cleaned)
    cleaned = " ".join(cleaned.strip().split())
    if "," in cleaned:
        last, first = [p.strip() for p in cleaned.split(",", 1)]
        if first and last:
            cleaned = f"{first} {last}"
    tokens = [t for t in cleaned.lower().split() if len(t) > 1]
    return " ".join(tokens)


def _normalize_name(name: str) -> str:
    return normalize_goalie_name(name)


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


def _goalie_lookup_for_season(season: str) -> dict[str, dict]:
    cached = _GOALIE_LOOKUP_CACHE.get(season)
    if cached is not None:
        return cached
    payload = _fetch_goalie_stats(season)
    data = payload.get("data", []) if isinstance(payload, dict) else []
    lookup: dict[str, dict] = {}
    for row in data:
        row_name = row.get("goalieFullName") or row.get("playerName") or ""
        name_norm = normalize_goalie_name(row_name)
        if not name_norm:
            continue
        current = lookup.get(name_norm)
        try:
            games = int(row.get("gamesPlayed") or 0)
        except Exception:
            games = 0
        if current is None:
            lookup[name_norm] = row
            continue
        try:
            current_games = int(current.get("gamesPlayed") or 0)
        except Exception:
            current_games = 0
        if games > current_games:
            lookup[name_norm] = row
    _GOALIE_LOOKUP_CACHE[season] = lookup
    return lookup


def _league_goalie_stats(season: str) -> tuple[float, float, float]:
    cached = _GOALIE_LEAGUE_STATS_CACHE.get(season)
    if cached is not None:
        return cached
    payload = _fetch_goalie_stats(season)
    data = payload.get("data", []) if isinstance(payload, dict) else []
    sv_values: list[float] = []
    for row in data:
        try:
            sv_pct = float(row.get("savePct"))
        except Exception:
            continue
        if sv_pct <= 0:
            continue
        sv_values.append(sv_pct)
    league_avg_sv = float(sum(sv_values) / len(sv_values)) if sv_values else DEFAULT_LEAGUE_AVG_SV
    ratings: list[float] = []
    for sv_pct in sv_values:
        rating = (sv_pct - league_avg_sv) * 1000.0
        rating = max(-30.0, min(30.0, rating))
        ratings.append(rating)
    if ratings:
        league_avg_rating = float(sum(ratings) / len(ratings))
        if len(ratings) > 1:
            mean = league_avg_rating
            league_std_rating = float((sum((r - mean) ** 2 for r in ratings) / len(ratings)) ** 0.5)
        else:
            league_std_rating = 1.0
    else:
        league_avg_rating = 0.0
        league_std_rating = 1.0
    if league_std_rating <= 1e-6:
        league_std_rating = 1.0
    stats = (league_avg_sv, league_avg_rating, league_std_rating)
    _GOALIE_LEAGUE_STATS_CACHE[season] = stats
    return stats


def get_goalie_rating_with_meta(goalie_name: str, season: str) -> tuple[float, bool]:
    if not goalie_name:
        return (0.0, False)

    lookup = _goalie_lookup_for_season(season)
    if not lookup:
        return (0.0, False)

    name_norm = normalize_goalie_name(goalie_name)
    best = lookup.get(name_norm)
    if best is None:
        if os.getenv("NHL_GOALIES_DEBUG") == "1":
            print(f"[goalie_rating] missing rating for: {goalie_name} season={season}")
        return (0.0, False)

    sv_pct = best.get("savePct")
    games = best.get("gamesPlayed", 0)
    try:
        sv_pct = float(sv_pct)
        games = int(games)
    except Exception:
        return (0.0, False)

    league_avg_sv, league_avg_rating, league_std_rating = _league_goalie_stats(season)
    rating = (sv_pct - league_avg_sv) * 1000.0
    rating = max(-30.0, min(30.0, rating))
    if games < 5:
        rating *= 0.5
    strength = (rating - league_avg_rating) / league_std_rating
    return (float(strength), True)


def get_goalie_rating(goalie_name: str, season: str) -> float:
    rating, _ = get_goalie_rating_with_meta(goalie_name, season)
    return float(rating)


def current_season_label() -> str:
    return _season_for_date(datetime.utcnow().date())
