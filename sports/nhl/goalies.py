from __future__ import annotations

import gzip
import json
import os
import re
import time
from datetime import datetime, timezone, date as date_type
from dataclasses import dataclass
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
DATE_FORMATS = ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y")


@dataclass
class GoalieInfo:
    team: str
    goalie_name: Optional[str]
    status: str  # CONFIRMED/PROJECTED/UNKNOWN
    source: str  # dailyfaceoff/puckpedia/...
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
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=req_headers,
                stream=True,
            )
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
                try:
                    import brotli

                    decompressed = brotli.decompress(raw)
                except ImportError:
                    pass
                except Exception:
                    pass

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
    text = (text or "").strip().upper()
    if "CONF" in text:
        return "CONFIRMED"
    if "PROJ" in text:
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
    return GoalieInfo(
        team=team,
        goalie_name=goalie_name,
        status=status,
        source=source,
        original_team=original_team,
    )


def _parse_goalie_date(date_str: str) -> Optional[datetime]:
    if isinstance(date_str, datetime):
        return date_str
    if isinstance(date_str, date_type):
        return datetime.combine(date_str, datetime.min.time())
    raw = str(date_str)
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
    parsed = _parse_goalie_date(date_str)
    if not parsed:
        raw = str(date_str)
        return raw, [raw]
    return (
        parsed.strftime("%Y-%m-%d"),
        [
            parsed.strftime("%Y-%m-%d"),
            parsed.strftime("%m-%d-%Y"),
            parsed.strftime("%m/%d/%Y"),
        ],
    )


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
    soup = BeautifulSoup(html, "html.parser")
    results: Dict[str, GoalieInfo] = {}

    def _walk(obj: object):
        yield obj
        if isinstance(obj, dict):
            for value in obj.values():
                yield from _walk(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from _walk(value)

    def _coerce_string(value: object, *, fallback_keys: tuple[str, ...]) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in fallback_keys:
                candidate = value.get(key)
                if isinstance(candidate, str):
                    return candidate
        return None

    next_data = soup.find("script", {"id": "__NEXT_DATA__"})
    if not next_data:
        next_data = None
    script_text = (next_data.string or next_data.get_text()) if next_data else None
    if script_text:
        try:
            parsed = json.loads(script_text)
        except json.JSONDecodeError:
            parsed = None
    else:
        parsed = None

    team_keys = ("teamName", "team", "teamAbbrev", "abbrev", "shortName")
    goalie_keys = ("goalieName", "goalie", "starter", "startingGoalie", "goalieFullName")
    fallback_team_keys = ("teamName", "team", "teamAbbrev", "abbrev", "shortName", "name")
    fallback_goalie_keys = ("goalieName", "goalie", "starter", "startingGoalie", "goalieFullName", "name")

    if parsed:
        for node in _walk(parsed):
            if not isinstance(node, dict):
                continue
            team_key = next((key for key in team_keys if key in node), None)
            goalie_key = next((key for key in goalie_keys if key in node), None)
            if not team_key or not goalie_key:
                continue
            team_raw = _coerce_string(node.get(team_key), fallback_keys=fallback_team_keys)
            goalie_raw = _coerce_string(node.get(goalie_key), fallback_keys=fallback_goalie_keys)
            goalie_name = _normalize_goalie_name(goalie_raw)
            if goalie_name and len(goalie_name.split()) < 2:
                continue
            team = canon_team(team_raw or "") if team_raw else None
            if not team and team_raw:
                team = team_raw.strip()
            if not team:
                continue
            results[team] = _build_goalie_info(
                team,
                goalie_raw=goalie_raw,
                status_raw="UNKNOWN",
                source="dailyfaceoff",
                original_team=team_raw.strip() if team_raw else None,
            )

    if results:
        return results

    for card in soup.select(".starting-goalies__goalie-card"):
        team_raw = card.select_one(".starting-goalies__team-name")
        goalie_raw = card.select_one(".starting-goalies__goalie-name")
        status_raw = card.select_one(".starting-goalies__status")
        goalie_name = _normalize_goalie_name(goalie_raw.get_text(" ", strip=True) if goalie_raw else "")
        if goalie_name and len(goalie_name.split()) < 2:
            continue
        team = canon_team(team_raw.get_text(" ", strip=True) if team_raw else "")
        if not team:
            continue
        status_text = status_raw.get_text(" ", strip=True) if status_raw else ""
        results[team] = _build_goalie_info(
            team,
            goalie_raw=goalie_name,
            status_raw=status_text,
            source="dailyfaceoff",
            original_team=team_raw.get_text(" ", strip=True) if team_raw else None,
        )

    return results


def _parse_puckpedia(html: str) -> Dict[str, GoalieInfo]:
    soup = BeautifulSoup(html, "html.parser")
    results: Dict[str, GoalieInfo] = {}
    tables = soup.find_all("table")
    if not tables:
        for card in soup.select(".goalie-card"):
            team_raw = (
                card.get("data-team")
                or card.get("data-team-name")
                or (card.select_one(".team-name").get_text(" ", strip=True) if card.select_one(".team-name") else "")
            )
            goalie_raw = card.select_one(".goalie-name")
            status_raw = card.select_one(".status")
            goalie_name = _normalize_goalie_name(goalie_raw.get_text(" ", strip=True) if goalie_raw else "")
            if goalie_name and len(goalie_name.split()) < 2:
                continue
            team = canon_team(team_raw)
            if not team:
                continue
            original_team = team_raw if team_raw and team_raw != team else None
            results[team] = _build_goalie_info(
                team,
                goalie_raw=goalie_name,
                status_raw=status_raw.get_text(" ", strip=True) if status_raw else "",
                source="puckpedia",
                original_team=original_team,
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
        goalie_name = _normalize_goalie_name(goalie_raw)
        if goalie_name and len(goalie_name.split()) < 2:
            continue
        team = canon_team(team_raw)
        if not team:
            continue
        original_team = team_raw if team_raw and team_raw != team else None
        results[team] = _build_goalie_info(
            team,
            goalie_raw=goalie_name,
            status_raw=status_raw,
            source="puckpedia",
            original_team=original_team,
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
    text_path = os.path.join(GOALIE_CACHE_DIR, "puckpedia_goalies_debug.txt")
    meta_path = os.path.join(GOALIE_CACHE_DIR, "puckpedia_goalies_debug_meta.json")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html or "")
    soup = BeautifulSoup(html or "", "html.parser")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(soup.get_text())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"status": status, "html_len": html_len}, f, indent=2, sort_keys=True)
    team_links_count = len(soup.find_all("a", href=lambda h: h and "/team/" in h))
    player_links_count = len(
        soup.find_all("a", href=lambda h: h and ("/player/" in h or "puckpedia.com/player/" in h))
    )
    print("[PuckPedia] team_links=", team_links_count)
    print("[PuckPedia] player_links=", player_links_count)
    if debug:
        print(f"[nhl goalies] debug puckpedia team_links={team_links_count} player_links={player_links_count}")
    parsed = _parse_puckpedia(html)
    print("[PuckPedia] parsed_goalies=", len(parsed))
    print("[PuckPedia] sample=", list(parsed.items())[:5])
    if debug:
        _debug_parse_details("puckpedia", PUCKPEDIA_URL, status, html, parsed)
    return parsed, {"status": status, "html_len": html_len, "html": html}


def fetch_goalies_puckpedia(day_count: int = 1) -> Dict[str, GoalieInfo]:
    parsed, _meta = _fetch_goalies_puckpedia_with_meta(day_count=day_count)
    return parsed

def _canonicalize_goalies_map(goalies: Dict[str, GoalieInfo]) -> Dict[str, GoalieInfo]:
    canon_results: Dict[str, GoalieInfo] = {}

    for team_key, info in (goalies or {}).items():
        raw_key = (str(team_key) if team_key is not None else "").strip()
        if not raw_key:
            continue

        team_canon = canon_team(raw_key)

        # Preserve the original team label if we have it
        if info.original_team is None and info.team and info.team != (team_canon or raw_key):
            info.original_team = info.team

        # Set info.team to canonical if available, else raw
        info.team = team_canon or raw_key

        # ALWAYS keep raw key so we never drop provider keys
        canon_results[raw_key] = info

        # ALSO store canonical key when available (helps model lookups)
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
        results = {}
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


def _debug_parse_details(provider: str, url: str, status: int, html: str, parsed: Dict[str, GoalieInfo]) -> None:
    html_len = len(html or "")
    sample = [(t, g.goalie_name, g.status) for t, g in list(parsed.items())[:5]]
    print(
        f"[nhl goalies] debug provider={provider} url={url} status={status} html_len={html_len} "
        f"parsed_rows={len(parsed)} sample={sample}"
    )
    if not parsed:
        soup = BeautifulSoup(html, "html.parser")
        text_head = " ".join(soup.get_text(" ", strip=True).split())
        print(f"[nhl goalies] debug provider={provider} text_head={text_head[:800]}")


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
            for team, info in goalies.items()
        }
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
    last_http_status: Optional[int] = None
    last_html_len: Optional[int] = None

    try:
        parsed, meta = _fetch_goalies_puckpedia_with_meta(day_count=1)
        parsed = _canonicalize_goalies_map(parsed)
        parsed_count = len(parsed)
        if parsed_count >= 1:
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
        last_error = f"puckpedia returned {parsed_count} goalies"
        last_http_status = meta.get("status")
        last_html_len = meta.get("html_len")
        if debug:
            print(f"[nhl goalies] WARNING: empty puckpedia parse status={last_http_status} html_len={last_html_len}")
    except Exception as exc:
        last_error = f"puckpedia fetch failed: {exc}"
        if debug:
            print(f"[nhl goalies] WARNING: failed to fetch puckpedia: {exc}")

    providers = [("dailyfaceoff", DAILY_FACEOFF_URL, _parse_daily_faceoff)]
    for date_key in date_keys:
        providers.insert(0, ("dailyfaceoff", f"{DAILY_FACEOFF_URL}/{date_key}", _parse_daily_faceoff))
    for name, url, parser in providers:
        try:
            html, status = _get_with_retry(url, headers={"Accept": "text/html"})
            os.makedirs(GOALIE_CACHE_DIR, exist_ok=True)
            debug_html_path = os.path.join(GOALIE_CACHE_DIR, "dailyfaceoff_goalies_debug.html")
            with open(debug_html_path, "w", encoding="utf-8") as f:
                f.write(html or "")
            soup = BeautifulSoup(html or "", "html.parser")
            next_data = soup.find("script", {"id": "__NEXT_DATA__"})
            next_data_path = os.path.join(GOALIE_CACHE_DIR, "dailyfaceoff_next_data.json")
            if next_data and (next_data.string or next_data.get_text()):
                with open(next_data_path, "w", encoding="utf-8") as f:
                    f.write(next_data.string or next_data.get_text())
            else:
                missing_path = os.path.join(GOALIE_CACHE_DIR, "dailyfaceoff_next_data_missing.txt")
                with open(missing_path, "w", encoding="utf-8") as f:
                    f.write((html or "")[:500])
            if debug:
                print(f"[nhl goalies] debug provider={name} url={url} status={status} html_len={len(html or '')}")
            parsed = parser(html)
            parsed = _canonicalize_goalies_map(parsed)
            if debug:
                _debug_parse_details(name, url, status, html, parsed)
            if len(parsed) >= 1:
                _write_cached_goalies(
                    cache_path,
                    parsed,
                    source=name,
                    date_key=cache_key,
                    http_status=status,
                    html_len=len(html or ""),
                )
                if debug:
                    _debug_goalies_summary(name, date_received, parsed)
                return parsed
            last_error = f"{name} returned {len(parsed)} goalies"
            last_http_status = status
            last_html_len = len(html or "")
            if debug:
                print(f"[nhl goalies] WARNING: empty parse status={status} html_len={len(html or '')} url={url}")
        except Exception as exc:
            last_error = f"{name} fetch failed: {exc}"
            if debug:
                print(f"[nhl goalies] WARNING: failed to fetch {url}: {exc}")

    fallback = _load_most_recent_cached_goalies()
    if fallback:
        return fallback
    raise RuntimeError(last_error or "No goalie providers returned data")
