from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

from sports.common.teams import canon_team


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
GOALIE_CACHE_DIR = "results/cache"
DAILY_FACEOFF_URL = "https://www.dailyfaceoff.com/starting-goalies"
PUCKPEDIA_URL = "https://depth-charts.puckpedia.com/starting-goalies"
PUCKPEDIA_PARAMS = {"dayCount": 1, "utm_medium": "embed", "utm_source": "puckpedia", "ads": "true"}


@dataclass
class GoalieInfo:
    team: str
    goalie_name: Optional[str]
    status: str  # CONFIRMED/PROJECTED/UNKNOWN
    source: str  # dailyfaceoff/puckpedia/...


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
        return {}
    script_text = next_data.string or next_data.get_text()
    if not script_text:
        return {}
    try:
        parsed = json.loads(script_text)
    except json.JSONDecodeError:
        return {}

    team_keys = ("teamName", "team", "teamAbbrev", "abbrev", "shortName")
    goalie_keys = ("goalieName", "goalie", "starter", "startingGoalie", "goalieFullName")
    fallback_team_keys = ("teamName", "team", "teamAbbrev", "abbrev", "shortName", "name")
    fallback_goalie_keys = ("goalieName", "goalie", "starter", "startingGoalie", "goalieFullName", "name")

    for node in _walk(parsed):
        if not isinstance(node, dict):
            continue
        team_key = next((key for key in team_keys if key in node), None)
        goalie_key = next((key for key in goalie_keys if key in node), None)
        if not team_key or not goalie_key:
            continue
        team_raw = _coerce_string(node.get(team_key), fallback_keys=fallback_team_keys)
        goalie_raw = _coerce_string(node.get(goalie_key), fallback_keys=fallback_goalie_keys)
        if not goalie_raw:
            continue
        goalie_name = goalie_raw.strip()
        if len(goalie_name.split()) < 2:
            continue
        team = canon_team(team_raw or "") if team_raw else None
        if not team and team_raw:
            team = team_raw.strip()
        if not team:
            continue
        results[team] = GoalieInfo(team=team, goalie_name=goalie_name, status="UNKNOWN", source="dailyfaceoff")

    return results


def _parse_puckpedia(html: str) -> Dict[str, GoalieInfo]:
    soup = BeautifulSoup(html, "html.parser")
    results: Dict[str, GoalieInfo] = {}
    used_goalies: set[str] = set()
    duplicate_goalies: set[str] = set()

    for block in soup.select("div, li, tr, section, article"):
        team_links = [
            link
            for link in block.find_all("a", href=True)
            if "/team/" in str(link.get("href") or "")
        ]
        player_links = [
            link
            for link in block.find_all("a", href=True)
            if "/player/" in str(link.get("href") or "") or "puckpedia.com/player/" in str(link.get("href") or "")
        ]
        if len(team_links) != 1 or len(player_links) < 1:
            continue

        team_href = str(team_links[0].get("href") or "")
        slug = team_href.split("/team/")[-1].split("/")[0]
        guess = slug.replace("-", " ").title()
        team = canon_team(guess)
        if not team or team in results:
            continue

        goalie_name = player_links[0].get_text(" ", strip=True).strip()
        if len(goalie_name.split()) < 2:
            continue
        if goalie_name in used_goalies:
            duplicate_goalies.add(goalie_name)
            continue
        used_goalies.add(goalie_name)

        status = _normalize_status(block.get_text(" ", strip=True))
        results[team] = GoalieInfo(team=team, goalie_name=goalie_name, status=status, source="puckpedia")

    if os.getenv("NHL_GOALIES_DEBUG") == "1":
        sample = list(results.items())[:5]
        print(
            "[nhl goalies] debug puckpedia parsed_team_count="
            f"{len(results)} unique_goalies={len(used_goalies)} duplicates={sorted(duplicate_goalies)} sample={sample}"
        )

    return results


def _fetch_goalies_puckpedia_with_meta(day_count: int = 1) -> tuple[Dict[str, GoalieInfo], dict]:
    debug = os.getenv("NHL_GOALIES_DEBUG") == "1"
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
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html or "")
    soup = BeautifulSoup(html or "", "html.parser")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(soup.get_text())
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


def _load_cached_goalies(cache_path: str) -> Dict[str, GoalieInfo]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        results = {}
        for team, info in (payload.get("goalies") or {}).items():
            if not isinstance(info, dict):
                continue
            results[team] = GoalieInfo(
                team=team,
                goalie_name=info.get("goalie_name"),
                status=info.get("status") or "UNKNOWN",
                source=info.get("source") or "cache",
            )
        return results
    except Exception:
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
                "goalie_name": info.goalie_name,
                "status": info.status,
                "source": info.source,
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
    cache_path = os.path.join(GOALIE_CACHE_DIR, f"nhl_goalies_{date}.json")
    cached = _load_cached_goalies(cache_path)
    if cached:
        return cached

    debug = os.getenv("NHL_GOALIES_DEBUG") == "1"
    last_error: Optional[str] = None
    last_http_status: Optional[int] = None
    last_html_len: Optional[int] = None
    last_partial_goalies: Dict[str, GoalieInfo] = {}

    try:
        parsed, meta = _fetch_goalies_puckpedia_with_meta(day_count=1)
        parsed_count = len(parsed)
        if parsed_count >= 5:
            _write_cached_goalies(
                cache_path,
                parsed,
                source="puckpedia",
                date_key=date,
                http_status=meta.get("status"),
                html_len=meta.get("html_len"),
            )
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

    daily_faceoff_date_url = f"{DAILY_FACEOFF_URL}/{date}"
    providers = [
        ("dailyfaceoff", daily_faceoff_date_url, _parse_daily_faceoff),
        ("dailyfaceoff", DAILY_FACEOFF_URL, _parse_daily_faceoff),
    ]
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
            if debug:
                _debug_parse_details(name, url, status, html, parsed)
            if parsed:
                last_partial_goalies = parsed
            if len(parsed) >= 5:
                _write_cached_goalies(
                    cache_path,
                    parsed,
                    source=name,
                    date_key=date,
                    http_status=status,
                    html_len=len(html or ""),
                )
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

    _write_cached_goalies(
        cache_path,
        last_partial_goalies,
        source="none" if not last_partial_goalies else "partial",
        date_key=date,
        error=last_error or "no goalie providers succeeded",
        http_status=last_http_status,
        html_len=last_html_len,
    )
    return {}
