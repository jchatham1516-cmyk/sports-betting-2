from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests


STATS_CACHE_DIR = "results/cache"
NHL_STATS_ENDPOINTS = [
    "https://api.nhle.com/stats/rest/en/goalie/summary",
    "https://api.nhle.com/stats/rest/en/goalie",
]
NHL_STATS_LIMIT = 300
MONEYPUCK_GOALIES_URL = "https://moneypuck.com/goalies.htm"
DEFAULT_LEAGUE_AVG_SV = 0.903
_GOALIE_LOOKUP_CACHE: dict[str, dict[str, dict]] = {}
_GOALIE_ALIAS_CACHE: dict[str, dict[str, dict[str, list[str]]]] = {}
_GOALIE_LEAGUE_STATS_CACHE: dict[str, tuple[float, float, float]] = {}
_GOALIE_LOOKUP_EMPTY_WARNED = False


def debug_goalie_stats_summary(season: str) -> None:
    payload = _fetch_goalie_stats(season)
    data = payload.get("data", []) if isinstance(payload, dict) else []
    rows = len(data)
    sv_values: list[float] = []
    goalie_names: list[str] = []
    high_sv_count = 0
    top_by_games: list[tuple[str, int, Optional[float]]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        sv_pct = _goalie_row_save_pct(row)
        name = row.get("goalieFullName") or row.get("playerName") or ""
        if name:
            goalie_names.append(str(name))
        games = _parse_int_value(row.get("gamesPlayed"), default=0)
        top_by_games.append((str(name), games, sv_pct))
        if sv_pct is None or sv_pct <= 0:
            continue
        if sv_pct >= 0.99:
            high_sv_count += 1
        sv_values.append(sv_pct)
    min_sv = min(sv_values) if sv_values else float("nan")
    mean_sv = float(sum(sv_values) / len(sv_values)) if sv_values else float("nan")
    max_sv = max(sv_values) if sv_values else float("nan")
    unique_goalies = len(set(goalie_names))
    print(
        "[nhl goalie ratings] summary "
        f"season={season} rows={rows} unique_goalies={unique_goalies} "
        f"savePct_min={min_sv:.4f} savePct_mean={mean_sv:.4f} savePct_max={max_sv:.4f} "
        f"savePct_ge_0.99={high_sv_count}"
    )
    top_by_games.sort(key=lambda item: item[1], reverse=True)
    print("[nhl goalie ratings] top_10_by_gamesPlayed:")
    for name, games, sv_pct in top_by_games[:10]:
        sv_display = f"{sv_pct:.4f}" if sv_pct is not None else "None"
        print(f"  {name} gamesPlayed={games} savePct={sv_display}")


def _nhl_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nhl.com/stats/goalies",
    }


def _get_with_retry(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = 30,
    max_retries: int = 4,
    headers: Optional[dict[str, str]] = None,
) -> dict:
    last_exc: Optional[Exception] = None
    headers = headers or _nhl_headers()
    backoff_schedule = [1, 2, 4]
    for attempt in range(1, max_retries + 1):
        retryable = False
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            text_snippet = resp.text[:200].strip().replace("\n", " ")
            if resp.status_code != 200 and os.getenv("NHL_GOALIES_DEBUG") == "1":
                print(
                    "[nhl goalie ratings] debug status="
                    f"{resp.status_code} body={text_snippet}"
                )
            if resp.status_code in {403, 429} or 500 <= resp.status_code <= 599:
                retryable = True
                raise RuntimeError(
                    "retryable status "
                    f"{resp.status_code} body={text_snippet}"
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    "unexpected status "
                    f"{resp.status_code} body={text_snippet}"
                )
            try:
                payload = resp.json()
            except Exception as exc:
                retryable = True
                raise RuntimeError(
                    f"failed to decode JSON status={resp.status_code} body={text_snippet}"
                ) from exc
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
    detail = f": {last_exc}" if last_exc else ""
    raise RuntimeError(f"failed to fetch {url}{detail}") from last_exc


def _season_for_date(dt: date) -> str:
    # NHL season label uses starting year, e.g. 2024 season for 2024-2025.
    return str(dt.year - 1 if dt.month < 7 else dt.year)


def normalize_goalie_name(name: str) -> str:
    if not name:
        return ""
    cleaned = unicodedata.normalize("NFKD", str(name))
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
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
    cleaned = cleaned.lower()
    cleaned = cleaned.replace("vasilevsky", "vasilevskiy")
    return cleaned


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


def _name_tokens(name: str) -> list[str]:
    normalized = normalize_goalie_name(name)
    if not normalized:
        return []
    return [token for token in normalized.split() if token]


def _best_goalie_by_games(lookup: dict[str, dict], candidates: list[str]) -> Optional[dict]:
    best = None
    best_games = -1
    for candidate in candidates:
        row = lookup.get(candidate)
        if not row:
            continue
        try:
            games = int(row.get("gamesPlayed") or 0)
        except Exception:
            games = 0
        if best is None or games > best_games:
            best = row
            best_games = games
    return best


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


def _money_puck_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3_1) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://moneypuck.com/",
    }


def _normalize_header(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", str(text or "").strip().lower())
    return cleaned


def _parse_sv_pct(raw_value: str) -> Optional[float]:
    if raw_value is None:
        return None
    cleaned = str(raw_value).strip()
    if cleaned.lower() in {"nan", "none", ""}:
        return None
    cleaned = cleaned.replace("%", "")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except Exception:
        return None
    if value <= 0:
        return None
    if value > 1.5:
        value /= 100.0
    return value


def _parse_save_pct_value(raw_value: object) -> Optional[float]:
    if raw_value is None:
        return None
    cleaned = str(raw_value).strip()
    if cleaned.lower() in {"nan", "none", ""}:
        return None
    try:
        value = float(cleaned)
    except Exception:
        return None
    if value <= 0:
        return None
    if value > 1.0:
        return value / 100.0
    return value


def _parse_int_value(raw_value: object, default: int = 0) -> int:
    try:
        return int(float(raw_value))
    except Exception:
        return default


def _parse_float_value(raw_value: object) -> Optional[float]:
    try:
        return float(raw_value)
    except Exception:
        return None


def _compute_save_pct_from_counts(saves: object, shots_against: object) -> Optional[float]:
    saves_value = _parse_float_value(saves)
    shots_value = _parse_float_value(shots_against)
    if saves_value is None or shots_value is None or shots_value <= 0:
        return None
    if saves_value < 0:
        return None
    return saves_value / shots_value


def _goalie_row_save_pct(row: dict) -> Optional[float]:
    sv_pct = _parse_save_pct_value(row.get("savePct"))
    if sv_pct is not None:
        return sv_pct
    return _compute_save_pct_from_counts(row.get("saves"), row.get("shotsAgainst"))


def _goalie_row_save_pct_meta(row: dict) -> tuple[Optional[float], str, dict[str, object]]:
    raw_save_pct = row.get("savePct")
    sv_pct = _parse_save_pct_value(raw_save_pct)
    if sv_pct is not None:
        return (sv_pct, "savePct", {"savePct": raw_save_pct})
    raw_saves = row.get("saves")
    raw_shots = row.get("shotsAgainst")
    sv_pct = _compute_save_pct_from_counts(raw_saves, raw_shots)
    if sv_pct is not None:
        return (sv_pct, "saves/shotsAgainst", {"saves": raw_saves, "shotsAgainst": raw_shots})
    return (None, "missing", {"savePct": raw_save_pct, "saves": raw_saves, "shotsAgainst": raw_shots})


def _save_pct_fields(row: dict) -> dict[str, object]:
    save_fields: dict[str, object] = {}
    for key in row.keys():
        normalized = _normalize_header(key)
        if "save" in normalized and ("pct" in normalized or "percent" in normalized or "sv" in normalized):
            save_fields[key] = row.get(key)
    return save_fields


def _matches_goalie_name_column(col_name: str) -> bool:
    normalized = _normalize_header(col_name)
    return normalized in {"goalie", "player", "name"} or any(
        key in normalized for key in ("goalie", "player", "name")
    )


def _matches_save_pct_column(col_name: str) -> bool:
    normalized = _normalize_header(col_name)
    if normalized in {"sv", "save"}:
        return True
    return any(
        key in normalized
        for key in ("svpct", "savepct", "savepercentage", "svpercent", "savepercent")
    )


def _matches_games_played_column(col_name: str) -> bool:
    normalized = _normalize_header(col_name)
    return normalized in {"gp", "games", "gamesplayed"} or "games" in normalized


def _fetch_goalie_stats_from_moneypuck(season: str) -> dict:
    resp = requests.get(MONEYPUCK_GOALIES_URL, headers=_money_puck_headers(), timeout=30)
    text_snippet = resp.text[:200].strip().replace("\n", " ")
    if resp.status_code != 200:
        raise RuntimeError(f"moneypuck status={resp.status_code} body={text_snippet}")

    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError("moneypuck goalie tables not found")

    data_rows: list[dict] = []
    for table in tables:
        if table.empty:
            continue
        columns = [str(col) for col in table.columns]
        name_cols = [col for col in columns if _matches_goalie_name_column(col)]
        sv_cols = [col for col in columns if _matches_save_pct_column(col)]
        if not name_cols or not sv_cols:
            continue
        name_col = name_cols[0]
        sv_col = sv_cols[0]
        gp_col = None
        gp_cols = [col for col in columns if _matches_games_played_column(col)]
        if gp_cols:
            gp_col = gp_cols[0]

        for _, row in table.iterrows():
            name = str(row.get(name_col, "")).strip()
            if not name or name.lower() in {"nan", "none"}:
                continue
            sv_pct = _parse_sv_pct(row.get(sv_col))
            if sv_pct is None:
                continue
            games_played = 0
            if gp_col:
                try:
                    games_played = int(float(row.get(gp_col)))
                except Exception:
                    games_played = 0
            data_rows.append(
                {
                    "goalieFullName": name,
                    "savePct": sv_pct,
                    "gamesPlayed": games_played,
                    "source": "moneypuck",
                }
            )
        if data_rows:
            break

    if not data_rows:
        raise RuntimeError("moneypuck goalie table empty")

    return {
        "data": data_rows,
        "season": season,
        "source": "moneypuck",
    }


def _fetch_goalie_stats_from_nhl(season: str) -> dict:
    season_id = _season_id_from_label(season)
    params_base = {
        "isAggregate": "false",
        "isGame": "true",
        "start": 0,
        "limit": 500,
        "sort": "[{\"property\":\"savePct\",\"direction\":\"DESC\"}]",
    }
    params_with_filters = {
        **params_base,
        "cayenneExp": f"seasonId={season_id} and gameTypeId=2",
        "factCayenneExp": "gamesPlayed>=5",
    }
    params_no_fact = {
        **params_base,
        "cayenneExp": f"seasonId={season_id} and gameTypeId=2",
    }
    last_exc: Optional[Exception] = None
    for endpoint in NHL_STATS_ENDPOINTS:
        for params in (params_with_filters, params_no_fact):
            try:
                payload = _get_with_retry(endpoint, params=params, headers=_nhl_headers())
            except Exception as exc:
                last_exc = exc
                continue
            rows = _payload_rows_count(payload)
            if rows > 0:
                if isinstance(payload, dict):
                    payload.setdefault("season", season)
                    payload.setdefault("source", "nhl_stats")
                return payload
    detail = f": {last_exc}" if last_exc else ""
    raise RuntimeError(f"NHL stats endpoints failed for season={season}{detail}")


def _fetch_goalie_stats(season: str) -> dict:
    cache_path = os.path.join(STATS_CACHE_DIR, f"nhl_goalie_stats_{season}.json")
    cache_exists = os.path.exists(cache_path)
    cached = _load_cached_stats(cache_path)
    cached_rows = _payload_rows_count(cached)
    cached_valid = cached_rows > 0
    debug_enabled = os.getenv("NHL_GOALIES_DEBUG") == "1"
    if cached_valid:
        if debug_enabled:
            print(
                "[nhl goalie ratings] debug season="
                f"{season} cache_used=True live_ok=False rows={cached_rows}"
            )
            print(
                "[nhl goalie ratings] season="
                f"{season} status=cached rows={cached_rows} cache=hit"
            )
        if debug_enabled:
            data = cached.get("data", []) if isinstance(cached, dict) else []
            sample_names = [
                row.get("goalieFullName") or row.get("playerName")
                for row in data[:3]
                if isinstance(row, dict)
            ]
            print(
                "[nhl goalie ratings] debug payload rows="
                f"{cached_rows} sample_goalies={sample_names}"
            )
        return cached

    try:
        payload = _fetch_goalie_stats_from_nhl(season)
    except Exception as exc:
        if debug_enabled:
            print(
                "[nhl goalie ratings] debug season="
                f"{season} cache_used={cache_exists} live_ok=False error={exc}"
            )
        print(f"[nhl goalie ratings] WARNING: stats fetch failed: {exc}")
        if cache_exists:
            if debug_enabled:
                print(
                    "[nhl goalie ratings] season="
                    f"{season} status=failure rows={cached_rows} cache=hit"
                )
            return cached
        try:
            payload = _fetch_goalie_stats_from_moneypuck(season)
            rows = _payload_rows_count(payload)
            if debug_enabled:
                print(
                    "[nhl goalie ratings] season="
                    f"{season} status=fallback rows={rows} cache=miss source=moneypuck"
                )
            if rows > 0:
                _write_cached_stats(cache_path, payload)
                return payload
        except Exception as fallback_exc:
            print(
                "[nhl goalie ratings] WARNING: fallback fetch failed: "
                f"{fallback_exc}"
            )
        if debug_enabled:
            print(
                "[nhl goalie ratings] season="
                f"{season} status=failure rows=0 cache=miss"
            )
        return {"data": [], "error": str(exc), "season": season}

    rows = _payload_rows_count(payload)
    if rows == 0:
        if debug_enabled:
            print(
                "[nhl goalie ratings] debug season="
                f"{season} cache_used={cache_exists} live_ok=False rows=0"
            )
        if debug_enabled:
            cache_status = "hit" if cached_valid else "miss"
            print(
                "[nhl goalie ratings] season="
                f"{season} status=failure rows=0 cache={cache_status}"
            )
        if cached_valid:
            return cached
        try:
            fallback_payload = _fetch_goalie_stats_from_moneypuck(season)
            fallback_rows = _payload_rows_count(fallback_payload)
            if fallback_rows > 0:
                if debug_enabled:
                    print(
                        "[nhl goalie ratings] season="
                        f"{season} status=fallback rows={fallback_rows} cache=miss source=moneypuck"
                    )
                _write_cached_stats(cache_path, fallback_payload)
                return fallback_payload
        except Exception as fallback_exc:
            print(
                "[nhl goalie ratings] WARNING: fallback fetch failed: "
                f"{fallback_exc}"
            )
        return payload

    if debug_enabled:
        print(
            "[nhl goalie ratings] debug season="
            f"{season} cache_used={cache_exists} live_ok=True rows={rows}"
        )
        cache_status = "hit" if cached_valid else "miss"
        print(
            "[nhl goalie ratings] season="
            f"{season} status=ok rows={rows} cache={cache_status}"
        )
        data = payload.get("data", []) if isinstance(payload, dict) else []
        sample_names = [
            row.get("goalieFullName") or row.get("playerName")
            for row in data[:3]
            if isinstance(row, dict)
        ]
        print(
            "[nhl goalie ratings] debug payload rows="
            f"{rows} sample_goalies={sample_names}"
        )
    if rows > 0:
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
        if not isinstance(row, dict):
            continue
        sv_pct = _goalie_row_save_pct(row)
        if sv_pct is None or sv_pct <= 0:
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


def _resolve_goalie_row(goalie_name: str, season: str) -> tuple[Optional[dict], str, bool, str]:
    lookup = _goalie_lookup_for_season(season)
    if not lookup:
        debug_enabled = os.getenv("NHL_GOALIES_DEBUG") == "1"
        global _GOALIE_LOOKUP_EMPTY_WARNED
        if debug_enabled and not _GOALIE_LOOKUP_EMPTY_WARNED:
            print(f"[nhl goalie ratings] WARNING: empty lookup for season={season}")
            _GOALIE_LOOKUP_EMPTY_WARNED = True
        return (None, "", False, "")

    name_norm = normalize_goalie_name(goalie_name)
    best = lookup.get(name_norm)
    matched_key = name_norm if best is not None else ""
    used_fallback = False
    if best is None and name_norm:
        alias_maps = _goalie_alias_maps(season)
        parts = _name_tokens(name_norm)
        if parts:
            last = parts[-1]
            first_initial = parts[0][0] if parts[0] else ""
            if last and first_initial:
                candidates = alias_maps.get("last_initial", {}).get(f"{last}|{first_initial}", [])
                if candidates:
                    used_fallback = True
                    best = _best_goalie_by_games(lookup, candidates)
                    if best is not None:
                        matched_key = candidates[0]
                        if len(candidates) > 1:
                            for candidate in candidates:
                                if lookup.get(candidate) is best:
                                    matched_key = candidate
                                    break
            if best is None and last:
                candidates = alias_maps.get("last", {}).get(last, [])
                if candidates:
                    used_fallback = True
                    if len(candidates) == 1:
                        matched_key = candidates[0]
                        best = lookup.get(candidates[0])
                    else:
                        best = _best_goalie_by_games(lookup, candidates)
                        if best is not None:
                            matched_key = candidates[0]
                            for candidate in candidates:
                                if lookup.get(candidate) is best:
                                    matched_key = candidate
                                    break
    return (best, matched_key, used_fallback, name_norm)


def _goalie_save_pct_with_meta(
    goalie_name: str,
    season: str,
) -> tuple[Optional[float], bool, int, str, dict[str, object]]:
    if not goalie_name:
        return (None, False, 0, "missing", {})
    best, matched_key, used_fallback, name_norm = _resolve_goalie_row(goalie_name, season)
    if best is None:
        return (None, False, 0, "missing", {})
    sv_pct, source, raw_values = _goalie_row_save_pct_meta(best)
    if sv_pct is None:
        return (None, False, _parse_int_value(best.get("gamesPlayed"), default=0), source, raw_values)
    games = _parse_int_value(best.get("gamesPlayed"), default=0)
    if games >= 5 and (sv_pct > 0.99 or sv_pct < 0.85):
        if os.getenv("NHL_GOALIES_DEBUG") == "1":
            print(
                "[nhl goalie ratings] WARNING: invalid savePct "
                f"for {goalie_name} gamesPlayed={games} savePct={sv_pct:.4f}"
            )
        return (None, False, games, source, raw_values)
    if used_fallback and os.getenv("NHL_GOALIES_DEBUG") == "1":
        print(
            "[nhl goalie ratings] fallback match "
            f"original={goalie_name} normalized={name_norm} matched_key={matched_key} "
            f"gamesPlayed={games} savePct={sv_pct:.4f}"
        )
    return (sv_pct, True, games, source, raw_values)


def get_goalie_save_pct(goalie_name: str, season: str) -> tuple[Optional[float], bool]:
    sv_pct, found, _, _, _ = _goalie_save_pct_with_meta(goalie_name, season)
    return (sv_pct, found)


def get_goalie_save_pct_meta(
    goalie_name: str,
    season: str,
) -> tuple[Optional[float], bool, int, str, dict[str, object]]:
    return _goalie_save_pct_with_meta(goalie_name, season)


def get_goalie_rating_with_meta(goalie_name: str, season: str) -> tuple[float, bool]:
    if not goalie_name:
        return (0.0, False)

    best, matched_key, used_fallback, name_norm = _resolve_goalie_row(goalie_name, season)
    if best is None:
        if os.getenv("NHL_GOALIES_DEBUG") == "1":
            print(f"[goalie_rating] missing rating for: {goalie_name} season={season}")
        return (0.0, False)

    debug_enabled = os.getenv("NHL_GOALIES_DEBUG") == "1"
    raw_games = best.get("gamesPlayed")
    raw_shots = best.get("shotsAgainst")
    raw_saves = best.get("saves")
    raw_save_pct = best.get("savePct")
    games = _parse_int_value(raw_games, default=0)
    if debug_enabled:
        save_fields = _save_pct_fields(best)
        print(
            "[nhl goalie ratings] debug goalie_row "
            f"name={goalie_name} gamesPlayed={raw_games} shotsAgainst={raw_shots} "
            f"saves={raw_saves} savePct={raw_save_pct} savePct_fields={save_fields}"
        )
        print(
            "[nhl goalie ratings] debug goalie_row_keys "
            f"name={goalie_name} keys={list(best.keys())}"
        )

    sv_pct, found, games, _, _ = _goalie_save_pct_with_meta(goalie_name, season)
    if used_fallback and os.getenv("NHL_GOALIES_DEBUG") == "1":
        sv_display = f"{sv_pct:.4f}" if sv_pct is not None else "None"
        print(
            "[nhl goalie ratings] fallback match "
            f"original={goalie_name} normalized={name_norm} matched_key={matched_key} "
            f"gamesPlayed={games} savePct={sv_display}"
        )

    league_avg_sv, _, _ = _league_goalie_stats(season)
    if sv_pct is None or not found:
        if debug_enabled:
            print(
                "[nhl goalie ratings] WARNING: missing savePct, "
                f"using league average for {goalie_name}"
            )
        return (0.0, False)
    rating_raw = (sv_pct - league_avg_sv) * 1000.0
    rating_raw = max(-30.0, min(30.0, rating_raw))
    if games < 5:
        rating_raw *= 0.5
    strength = rating_raw / 10.0
    return (float(strength), True)


def get_goalie_rating(goalie_name: str, season: str) -> float:
    rating, _ = get_goalie_rating_with_meta(goalie_name, season)
    return float(rating)


def current_season_label() -> str:
    return _season_for_date(datetime.utcnow().date())


def debug_goalie_rating(names: list[str]) -> None:
    season = current_season_label()
    for name in names:
        strength, found = get_goalie_rating_with_meta(name, season)
        print(f"[goalie debug] season={season} name={name} strength={strength:.4f} found={found}")


if __name__ == "__main__":
    season = current_season_label()
    debug_goalie_stats_summary(season)
    for name in ["Jake Oettinger", "Andrei Vasilevskiy"]:
        print(name, get_goalie_rating_with_meta(name, season))
