from __future__ import annotations

import gzip
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, date as date_type
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

from sports.common.teams import canon_team


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
GOALIE_CACHE_DIR = "results/cache"
DEFAULT_CACHE_TTL_MINUTES = 120

DAILY_FACEOFF_URL = "https://www.dailyfaceoff.com/starting-goalies"
PUCKPEDIA_URL = "https://depth-charts.puckpedia.com/starting-goalies"
PUCKPEDIA_PARAMS = {"dayCount": 1, "utm_medium": "embed", "utm_source": "puckpedia", "ads": "true"}

# Accept these for parsing user input
DATE_FORMATS = ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y")
# Only these are safe for building URL path segments
DATE_URL_FORMATS = ("%Y-%m-%d", "%m-%d-%Y")


@dataclass
class GoalieInfo:
    team: str
    goalie_name: Optional[str]
    status: str  # CONFIRMED/PROJECTED/UNKNOWN
    source: str  # dailyfaceoff/puckpedia/cache/...
    original_team: Optional[str] = None


def _get_with_retry(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: int = 20,
    max_retries: int = 3,
    headers: Optional[dict] = None,
) -> tuple[str, int]:
    last_exc: Optional[Exception] = None
    req_headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if headers:
        req_headers.update(headers)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=req_headers, stream=True)
            if resp.status_code >= 500:
                raise RuntimeError(f"server error {resp.status_code}")
            if resp.status_code != 200:
                print(f"[nhl goalies] WARNING: status={resp.status_code} html_len={len(resp.text or '')} url={url}")
                raise RuntimeError(f"unexpected status {resp.status_code}")

            raw = resp.content
            try:
                html = raw.decode("utf-8")
            except UnicodeDecodeError:
                html = raw.decode("latin-1", errors="ignore")

            decompressed: Optional[bytes] = None
            if raw.startswith(b"\x1f\x8b"):
                try:
                    decompressed = gzip.decompress(raw)
                except OSError:
                    decompressed = None
            else:
                # Optional brotli support (only if installed)
                try:
                    import brotli  # type: ignore

                    decompressed = brotli.decompress(raw)
                except ImportError:
                    decompressed = None
                except Exception:
                    decompressed = None

            if decompressed:
                try:
                    html = decompressed.decode("utf-8")
                except UnicodeDecodeError:
                    html = decompressed.decode("latin-1", errors="ignore")

            return html, resp.status_code

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2)

    raise RuntimeError(f"failed to fetch {url}") from last_exc


def _normalize_status(text: str) -> str:
    t = (text or "").strip().upper()
    if "CONF" in t:
        return "CONFIRMED"
    if "PROJ" in t:
        return "PROJECTED"
    return "UNKNOWN"


def _normalize_goalie_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = " ".join(str(name).strip().split())
    if not cleaned:
        return None
    if cleaned.upper() in {"TBD", "UNKNOWN", "TBA", "N/A"}:
        return None
    return cleaned


def _build_goalie_info(
    team: str,
    *,
    goalie_raw: Optional[str],
    status_raw: Optional[str],
    source: str,
    original_team: Optional[str] = None,
) -> GoalieInfo:
    goalie_name = _normalize_goalie_name(goalie_raw)
    status = _normalize_status(status_raw or "")
    if not goalie_name:
        status = "UNKNOWN"
    return GoalieInfo(team=team, goalie_name=goalie_name, status=status, source=source, original_team=original_team)


def _parse_goalie_date(date_str: str) -> Optional[datetime]:
    if isinstance(date_str, datetime):
        return date_str
    if isinstance(date_str, date_type):
        return datetime.combine(date_str, datetime.min.time())

    raw = str(date_str).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _date_keys_for_goalies(date_str: str) -> tuple[str, list[str]]:
    """
    Returns:
      cache_key: YYYY-MM-DD (always)
      date_keys: url-safe keys for providers (NO slashes)
    """
    parsed = _parse_goalie_date(date_str)
    if not parsed:
        raw = str(date_str).strip()
        # build a safe-ish cache key even if input is odd
        safe = raw.replace("/", "-")
        return safe, [safe] if "/" not in safe else [safe.replace("/", "-")]

    cache_key = parsed.strftime("%Y-%m-%d")

    url_keys: list[str] = []
    for fmt in DATE_URL_FORMATS:
        try:
            url_keys.append(parsed.strftime(fmt))
        except Exception:
            continue

    # extra safety: never allow slash dates into URL list
    url_keys = [k for k in url_keys if "/" not in k]

    return cache_key, url_keys


def _debug_goalies_summary(source: str, date_received: str, goalies: Dict[str, GoalieInfo]) -> None:
    sample_keys = list(goalies.keys())[:5]
    print(
        "[nhl goalies] "
        f"debug summary source={source} date_received={date_received} teams={len(goalies)} sample_keys={sample_keys}"
    )


def _cache_ttl_seconds() -> int:
    try:
        ttl_min = int(float(os.getenv("NHL_GOALIE_CACHE_TTL_MIN", str(DEFAULT_CACHE_TTL_MINUTES))))
    except Exception:
        ttl_min = DEFAULT_CACHE_TTL_MINUTES
    return max(0, ttl_min) * 60


def _parse_daily_faceoff(html: str) -> Dict[str, GoalieInfo]:
    soup = BeautifulSoup(html or "", "html.parser")
    results: Dict[str, GoalieInfo] = {}
    debug = os.getenv("NHL_DEBUG_GOALIES") == "1"

    next_data = soup.find("script", {"id": "__NEXT_DATA__"})
    script_text = (next_data.string or next_data.get_text()) if next_data else None
    parsed = None
    if script_text:
        try:
            parsed = json.loads(script_text)
        except Exception:
            parsed = None

    # NEWER schema: props.pageProps.data is a list of games with homeTeamName/homeGoalieName etc
    if isinstance(parsed, dict):
        data = (
            parsed.get("props", {})
            .get("pageProps", {})
            .get("data")
        )
        if isinstance(data, list):
            if debug:
                print(f"[nhl goalies] debug dailyfaceoff pageProps.data games={len(data)}")
            for game in data:
                if not isinstance(game, dict):
                    continue

                home_team_raw = game.get("homeTeamName")
                away_team_raw = game.get("awayTeamName")
                home_goalie_raw = game.get("homeGoalieName")
                away_goalie_raw = game.get("awayGoalieName")

                home_team = canon_team(home_team_raw or "") if home_team_raw else ""
                away_team = canon_team(away_team_raw or "") if away_team_raw else ""

                if home_team:
                    results[home_team] = _build_goalie_info(
                        home_team,
                        goalie_raw=home_goalie_raw,
                        status_raw="PROJECTED",
                        source="dailyfaceoff",
                        original_team=home_team_raw,
                    )
                if away_team:
                    results[away_team] = _build_goalie_info(
                        away_team,
                        goalie_raw=away_goalie_raw,
                        status_raw="PROJECTED",
                        source="dailyfaceoff",
                        original_team=away_team_raw,
                    )

            if debug:
                print(f"[nhl goalies] debug dailyfaceoff next_data_goalies={len(results)}")
            if results:
                return results

    # Fallback selectors (older HTML)
    for card in soup.select(".starting-goalies__goalie-card"):
        team_el = card.select_one(".starting-goalies__team-name")
        goalie_el = card.select_one(".starting-goalies__goalie-name")
        status_el = card.select_one(".starting-goalies__status")

        team_raw = team_el.get_text(" ", strip=True) if team_el else ""
        goalie_raw = goalie_el.get_text(" ", strip=True) if goalie_el else ""
        status_raw = status_el.get_text(" ", strip=True) if status_el else ""

        team = canon_team(team_raw)
        goalie_name = _normalize_goalie_name(goalie_raw)

        if team and goalie_name:
            results[team] = _build_goalie_info(
                team,
                goalie_raw=goalie_name,
                status_raw=status_raw,
                source="dailyfaceoff",
                original_team=team_raw or None,
            )

    return results


def _parse_puckpedia(html: str) -> Dict[str, GoalieInfo]:
    soup = BeautifulSoup(html or "", "html.parser")
    results: Dict[str, GoalieInfo] = {}

    tables = soup.find_all("table")
    if not tables:
        # Some layouts use cards
        for card in soup.select(".goalie-card"):
            team_raw = (
                card.get("data-team")
                or card.get("data-team-name")
                or (card.select_one(".team-name").get_text(" ", strip=True) if card.select_one(".team-name") else "")
            )
            goalie_el = card.select_one(".goalie-name")
            status_el = card.select_one(".status")
            goalie_raw = goalie_el.get_text(" ", strip=True) if goalie_el else ""
            status_raw = status_el.get_text(" ", strip=True) if status_el else ""

            team = canon_team(team_raw)
            goalie_name = _normalize_goalie_name(goalie_raw)
            if team and goalie_name:
                results[team] = _build_goalie_info(
                    team,
                    goalie_raw=goalie_name,
                    status_raw=status_raw,
                    source="puckpedia",
                    original_team=team_raw if team_raw and team_raw != team else None,
                )
        return results

    def _row_count(table) -> int:
        return len(table.find_all("tr"))

    largest_table = max(tables, key=_row_count)

    for row in largest_table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        team_raw = cells[0].get_text(" ", strip=True)
        goalie_raw = cells[1].get_text(" ", strip=True)
        status_raw = cells[2].get_text(" ", strip=True) if len(cells) > 2 else ""

        team = canon_team(team_raw)
        goalie_name = _normalize_goalie_name(goalie_raw)
        if team and goalie_name:
            results[team] = _build_goalie_info(
                team,
                goalie_raw=goalie_name,
                status_raw=status_raw,
                source="puckpedia",
                original_team=team_raw if team_raw and team_raw != team else None,
            )

    return results


def _fetch_goalies_puckpedia_with_meta(day_count: int = 1) -> tuple[Dict[str, GoalieInfo], dict]:
    debug = os.getenv("NHL_DEBUG_GOALIES") == "1"
    html, status = _get_with_retry(
        PUCKPEDIA_URL,
        params={**PUCKPEDIA_PARAMS, "dayCount": int(day_count)},
        headers={"Accept": "text/html"},
    )
    html_len = len(html or "")
    if debug:
        print(f"[nhl goalies] debug provider=puckpedia url={PUCKPEDIA_URL} status={status} html_len={html_len}")

    os.makedirs(GOALIE_CACHE_DIR, exist_ok=True)
    html_path = os.path.join(GOALIE_CACHE_DIR, "puckpedia_goalies_debug.html")
    meta_path = os.path.join(GOALIE_CACHE_DIR, "puckpedia_goalies_debug_meta.json")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html or "")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"status": status, "html_len": html_len}, f, indent=2, sort_keys=True)

    parsed = _parse_puckpedia(html)
    if debug:
        print(f"[nhl goalies] debug puckpedia parsed_goalies={len(parsed)} sample={list(parsed.items())[:5]}")

    return parsed, {"status": status, "html_len": html_len}


def _canonicalize_goalies_map(goalies: Dict[str, GoalieInfo]) -> Dict[str, GoalieInfo]:
    """
    Keep BOTH:
      - provider raw key (so we don't drop weird provider keys)
      - canonical key (so the model lookup works)
    """
    canon_results: Dict[str, GoalieInfo] = {}
    for team_key, info in (goalies or {}).items():
        raw_key = (str(team_key) if team_key is not None else "").strip()
        if not raw_key or info is None:
            continue

        team_canon = canon_team(raw_key)

        if info.original_team is None and info.team and info.team != (team_canon or raw_key):
            info.original_team = info.team

        info.team = team_canon or raw_key

        canon_results[raw_key] = info
        if team_canon:
            canon_results[team_canon] = info

    return canon_results


def _load_cached_goalies(cache_path: str) -> Dict[str, GoalieInfo]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}

        fetched_at = payload.get("fetched_at_iso")
        if fetched_at:
            try:
                fetched_dt = datetime.fromisoformat(str(fetched_at))
                if fetched_dt.tzinfo is None:
                    fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
                age_seconds = (datetime.now(timezone.utc) - fetched_dt).total_seconds()
                if age_seconds > _cache_ttl_seconds():
                    return {}
            except Exception:
                pass

        results: Dict[str, GoalieInfo] = {}
        for team, info in (payload.get("goalies") or {}).items():
            if not isinstance(info, dict):
                continue
            stored_team = info.get("team") or team
            team_canon = canon_team(stored_team)
            if not team_canon:
                continue
            goalie_name = _normalize_goalie_name(info.get("goalie_name"))
            results[team_canon] = GoalieInfo(
                team=team_canon,
                goalie_name=goalie_name,
                status=info.get("status") or "UNKNOWN",
                source=info.get("source") or "cache",
                original_team=info.get("team_raw") or (stored_team if stored_team != team_canon else None),
            )
        return results
    except Exception:
        return {}


def _load_most_recent_cached_goalies() -> Dict[str, GoalieInfo]:
    if not os.path.isdir(GOALIE_CACHE_DIR):
        return {}
    candidate_files = [
        os.path.join(GOALIE_CACHE_DIR, filename)
        for filename in os.listdir(GOALIE_CACHE_DIR)
        if filename.startswith("nhl_goalies_") and filename.endswith(".json")
    ]
    candidate_files.sort(key=lambda path: os.path.getmtime(path), reverse=True)

    for cache_path in candidate_files:
        cached = _load_cached_goalies(cache_path)
        if cached:
            for info in cached.values():
                info.source = "cache_fallback"
            return cached
    return {}


def _write_cached_goalies(
    cache_path: str,
    goalies: Dict[str, GoalieInfo],
    *,
    source: str,
    date_key: str,
    error: Optional[str] = None,
    http_status: Optional[int] = None,
    html_len: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "source": source,
        "date_key": date_key,
        "fetched_at_iso": datetime.now(timezone.utc).isoformat(),
        "goalies": {
            team: {
                "team": info.team,
                "goalie_name": info.goalie_name,
                "status": info.status,
                "source": info.source,
                **({"team_raw": info.original_team} if info.original_team else {}),
            }
            for team, info in (goalies or {}).items()
        },
    }
    if error:
        payload["error"] = error
    if http_status is not None:
        payload["http_status"] = http_status
    if html_len is not None:
        payload["html_len"] = html_len

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def get_starting_goalies(date: str) -> Dict[str, GoalieInfo]:
    date_received = str(date)
    cache_key, date_keys = _date_keys_for_goalies(date_received)
    cache_path = os.path.join(GOALIE_CACHE_DIR, f"nhl_goalies_{cache_key}.json")

    cached = _load_cached_goalies(cache_path)
    if cached:
        if os.getenv("NHL_DEBUG_GOALIES") == "1":
            _debug_goalies_summary("cache", date_received, cached)
        return cached

    debug = os.getenv("NHL_DEBUG_GOALIES") == "1"
    last_error: Optional[str] = None

    # 1) Try puckpedia first
    try:
        parsed, meta = _fetch_goalies_puckpedia_with_meta(day_count=1)
        parsed = _canonicalize_goalies_map(parsed)
        if len(parsed) >= 1:
            _write_cached_goalies(
                cache_path,
                parsed,
                source="puckpedia",
                date_key=cache_key,
                http_status=meta.get("status"),
                html_len=meta.get("html_len"),
            )
            if debug:
                _debug_goalies_summary("puckpedia", date_received, parsed)
            return parsed
        last_error = f"puckpedia returned {len(parsed)} goalies"
        if debug:
            print(f"[nhl goalies] WARNING: empty puckpedia parse status={meta.get('status')} html_len={meta.get('html_len')}")
    except Exception as exc:
        last_error = f"puckpedia fetch failed: {exc}"
        if debug:
            print(f"[nhl goalies] WARNING: failed to fetch puckpedia: {exc}")

    # 2) Try DailyFaceoff: base URL + safe dated URLs
    providers = [("dailyfaceoff", DAILY_FACEOFF_URL)]
    for dk in date_keys:
        # Extra safety: only allow yyyy-mm-dd or mm-dd-yyyy
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}", dk or ""):
            providers.insert(0, ("dailyfaceoff", f"{DAILY_FACEOFF_URL}/{dk}"))

    for name, url in providers:
        try:
            html, status = _get_with_retry(url, headers={"Accept": "text/html"})
            if debug:
                print(f"[nhl goalies] debug provider={name} url={url} status={status} html_len={len(html or '')}")

            # Write debug HTML so you can inspect what the provider returned
            os.makedirs(GOALIE_CACHE_DIR, exist_ok=True)
            debug_html_path = os.path.join(GOALIE_CACHE_DIR, "dailyfaceoff_goalies_debug.html")
            with open(debug_html_path, "w", encoding="utf-8") as f:
                f.write(html or "")

            parsed = _parse_daily_faceoff(html)
            parsed = _canonicalize_goalies_map(parsed)

            if len(parsed) >= 1:
                _write_cached_goalies(
                    cache_path,
                    parsed,
                    source="dailyfaceoff",
                    date_key=cache_key,
                    http_status=status,
                    html_len=len(html or ""),
                )
                if debug:
                    _debug_goalies_summary("dailyfaceoff", date_received, parsed)
                return parsed

            last_error = f"dailyfaceoff returned {len(parsed)} goalies"
            if debug:
                print(f"[nhl goalies] WARNING: empty parse status={status} html_len={len(html or '')} url={url}")

        except Exception as exc:
            last_error = f"dailyfaceoff fetch failed: {exc}"
            if debug:
                print(f"[nhl goalies] WARNING: failed to fetch {url}: {exc}")

    # 3) Cache fallback
    fallback = _load_most_recent_cached_goalies()
    if fallback:
        return fallback

    raise RuntimeError(last_error or "No goalie providers returned data")
