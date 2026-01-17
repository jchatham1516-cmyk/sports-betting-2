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
_GOALIE_ALIAS_CACHE: dict[str, dict[str, dict[str, list[str]]]] = {}
_GOALIE_LEAGUE_STATS_CACHE: dict[str, tuple[float, float, float]] = {}


def _get_with_retry(url: str, *, params: Optional[dict] = None, timeout: int = 30, max_retries: int = 4) -> dict:
    last_exc: Optional[Exception] = None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    backoff_schedule = [2, 4, 8]
    for attempt in range(1, max_retries + 1):
        retryable = False
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            if resp.status_code in {403, 429} or 500 <= resp.status_code <= 599:
                retryable = True
                raise RuntimeError(f"retryable status {resp.status_code}")
            if resp.status_code != 200:
                raise RuntimeError(f"unexpected status {resp.status_code}")
            try:
                payload = resp.json()
            except Exception as exc:
                retryable = True
                raise RuntimeError("non-json response") from exc
            return payload
        except requests.RequestException as exc:
            last_exc = exc
            retryable = True
        except Exception as exc:
            last_exc = exc
        if attempt < max_retries and retryable:
            backoff_index = min(attempt - 1, len(backoff_schedule) - 1)
            time.sleep(backoff_schedule[backoff_index])
        elif not retryable:
            break
    raise RuntimeError(f"failed to fetch {url}") from last_exc


def _season_for_date(dt: date) -> str:
    # NHL season label uses starting year, e.g. 2024 season for 2024-2025.
    return str(dt.year - 1 if dt.month < 7 else dt.year)


def normalize_goalie_name(name: str) -> str:
    if not name:
        return ""
    cleaned = unicodedata.normalize("NFKD", str(name))
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"\(.*?\)", " ", cleaned)
    cleaned = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.strip().split())
    if "," in cleaned:
        last, first = [p.strip() for p in cleaned.split(",", 1)]
        if first and last:
            cleaned = f"{first} {last}"
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = cleaned.replace("_", " ")
    cleaned = " ".join(cleaned.strip().split())
    return cleaned.lower()


def _season_id_from_label(season: str) -> str:
    if not season:
        return ""
    digits = re.findall(r"\d{4}", str(season))
    if len(digits) >= 2:
        return f"{digits[0]}{digits[1]}"
    if len(digits) == 1:
        start = int(digits[0])
        end = start + 1
        return f"{start}{end}"
    return str(season).strip()


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


def _payload_rows_count(payload: object) -> int:
    if isinstance(payload, dict):
        data = payload.get("data", [])
        if isinstance(data, list):
            return len(data)
    return 0


def _write_cached_stats(cache_path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _fetch_goalie_stats(season: str) -> dict:
    cache_path = os.path.join(STATS_CACHE_DIR, f"nhl_goalie_stats_{season}.json")
    cached = _load_cached_stats(cache_path)
    cached_rows = _payload_rows_count(cached)
    cached_valid = cached_rows > 0
    debug_enabled = os.getenv("NHL_DEBUG_GOALIE_RATINGS") == "1"
    if cached_valid:
        if debug_enabled:
            print(
                "[nhl goalie ratings] season="
                f"{season} status=cached rows={cached_rows} cache=hit"
            )
        return cached

    season_id = _season_id_from_label(season)
    params = {
        "isAggregate": "false",
        "isGame": "false",
        "sort": "[{\"property\":\"savePct\",\"direction\":\"DESC\"}]",
        "start": 0,
        "limit": NHL_STATS_LIMIT,
        "factCayenneExp": "gamesPlayed>=5",
        "cayenneExp": f"seasonId={season_id}",
    }
    try:
        payload = _get_with_retry(NHL_STATS_BASE, params=params)
    except Exception as exc:
        print(f"[nhl goalie ratings] WARNING: stats fetch failed: {exc}")
        if cached_valid:
            if debug_enabled:
                print(
                    "[nhl goalie ratings] season="
                    f"{season} status=failure rows={cached_rows} cache=hit"
                )
            return cached
        if debug_enabled:
            print(
                "[nhl goalie ratings] season="
                f"{season} status=failure rows=0 cache=miss"
            )
        return {}

    rows = _payload_rows_count(payload)
    if rows == 0:
        if debug_enabled:
            cache_status = "hit" if cached_valid else "miss"
            print(
                "[nhl goalie ratings] season="
                f"{season} status=failure rows=0 cache={cache_status}"
            )
        if cached_valid:
            return cached
        return {}

    if debug_enabled:
        cache_status = "hit" if cached_valid else "miss"
        print(
            "[nhl goalie ratings] season="
            f"{season} status=ok rows={rows} cache={cache_status}"
        )
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


def _goalie_alias_maps(season: str) -> dict[str, dict[str, list[str]]]:
    cached = _GOALIE_ALIAS_CACHE.get(season)
    if cached is not None:
        return cached
    lookup = _goalie_lookup_for_season(season)
    last_map: dict[str, list[str]] = {}
    last_initial_map: dict[str, list[str]] = {}
    for name_key in lookup.keys():
        parts = name_key.split()
        if not parts:
            continue
        last = parts[-1]
        first = parts[0]
        last_map.setdefault(last, []).append(name_key)
        if first:
            last_initial_map.setdefault(f"{last}|{first[0]}", []).append(name_key)
    cached = {"last": last_map, "last_initial": last_initial_map}
    _GOALIE_ALIAS_CACHE[season] = cached
    return cached


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
    if best is None and name_norm:
        alias_maps = _goalie_alias_maps(season)
        parts = name_norm.split()
        if parts:
            last = parts[-1]
            first = parts[0]
            if last and first:
                candidates = alias_maps.get("last_initial", {}).get(f"{last}|{first[0]}", [])
                if len(candidates) == 1:
                    best = lookup.get(candidates[0])
            if best is None and last:
                candidates = alias_maps.get("last", {}).get(last, [])
                if len(candidates) == 1:
                    best = lookup.get(candidates[0])
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
