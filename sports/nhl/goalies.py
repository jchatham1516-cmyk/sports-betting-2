from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

from sports.common.teams import canon_team


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
GOALIE_CACHE_DIR = "results/cache"
DAILY_FACEOFF_URL = "https://www.dailyfaceoff.com/starting-goalies"
PUCKPEDIA_URL = "https://depth-charts.puckpedia.com/starting-goalies"


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
    req_headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    if headers:
        req_headers.update(headers)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=req_headers,
            )
            if resp.status_code >= 500:
                raise RuntimeError(f"server error {resp.status_code}")
            if resp.status_code != 200:
                print(f"[nhl goalies] WARNING: status={resp.status_code} html_len={len(resp.text or '')} url={url}")
                raise RuntimeError(f"unexpected status {resp.status_code}")
            return resp.text, resp.status_code
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

    def _add_row(team_raw: str, goalie_raw: str, status_raw: str) -> None:
        team = canon_team(team_raw)
        if not team:
            return
        goalie_name = goalie_raw.strip() if goalie_raw and goalie_raw.strip() else None
        if not goalie_name:
            return
        status = _normalize_status(status_raw)
        results[team] = GoalieInfo(team=team, goalie_name=goalie_name, status=status, source="dailyfaceoff")

    tables = soup.select(
        "table.starting-goalies__table, table.starting-goalies, table#starting-goalies, table[data-testid*='starting-goalies']"
    )
    for table in tables:
        for row in table.select("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            status_raw = cells[2] if len(cells) > 2 else ""
            _add_row(cells[0], cells[1], status_raw)

    if results:
        return results

    for card in soup.select(
        ".starting-goalies__goalie-card, .starting-goalies__goalie, .starting-goalies__row, .starting-goalies__item"
    ):
        team_node = card.select_one(
            ".starting-goalies__team-name, .starting-goalies__team, .team-name"
        )
        goalie_node = card.select_one(
            ".starting-goalies__goalie-name, .starting-goalies__goalie, .starting-goalies__player-name, .player-name"
        )
        status_node = card.select_one(
            ".starting-goalies__status, .starting-goalies__status-text, .status"
        )
        if not team_node or not goalie_node:
            continue
        _add_row(
            team_node.get_text(" ", strip=True),
            goalie_node.get_text(" ", strip=True),
            status_node.get_text(" ", strip=True) if status_node else "",
        )

    return results


def _parse_puckpedia(html: str) -> Dict[str, GoalieInfo]:
    soup = BeautifulSoup(html, "html.parser")
    results: Dict[str, GoalieInfo] = {}

    def _text_from(node) -> str:
        return node.get_text(" ", strip=True) if node else ""

    def _status_from_text(text: str) -> str:
        status = _normalize_status(text)
        if status != "UNKNOWN":
            return status
        return "UNKNOWN"

    def _guess_status(block) -> str:
        status_node = block.select_one(".status, .goalie-status, [class*='status']")
        status_text = _text_from(status_node)
        if status_text:
            return _status_from_text(status_text)
        return _status_from_text(block.get_text(" ", strip=True))

    def _guess_team(block) -> str:
        for attr in ("data-team", "data-team-name"):
            if block.has_attr(attr) and block.get(attr):
                return str(block.get(attr))
        selectors = [".team-name", ".team", "[class*='team-name']", "[class*='team']"]
        for sel in selectors:
            team_text = _text_from(block.select_one(sel))
            if team_text:
                return team_text
        return ""

    def _guess_goalie(block) -> Optional[str]:
        selectors = [".goalie-name", ".player-name", "[class*='goalie-name']", "[class*='player-name']"]
        for sel in selectors:
            name_text = _text_from(block.select_one(sel))
            if name_text:
                return name_text
        for sel in ("[class*='goalie']", "[class*='player']"):
            name_text = _text_from(block.select_one(sel))
            if name_text and _normalize_status(name_text) == "UNKNOWN":
                return name_text
        return None

    def _is_goalie_block(tag) -> bool:
        if not tag or not tag.has_attr("class"):
            return False
        classes = " ".join(tag.get("class", [])).lower()
        return "goalie" in classes and any(k in classes for k in ("card", "block", "row", "item"))

    blocks = list(soup.find_all(_is_goalie_block))
    if not blocks:
        blocks = list(soup.select("[data-team], [data-team-name]"))

    for block in blocks:
        team_raw = _guess_team(block)
        team = canon_team(team_raw)
        if not team:
            continue
        goalie_name = _guess_goalie(block)
        status = _guess_status(block)
        results[team] = GoalieInfo(team=team, goalie_name=goalie_name, status=status, source="puckpedia")

    if results:
        return results

    for row in soup.select("table tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        team_raw = cells[0]
        goalie_raw = cells[1]
        status_raw = cells[2] if len(cells) > 2 else ""
        team = canon_team(team_raw)
        if not team:
            continue
        goalie_name = goalie_raw.strip() if goalie_raw.strip() else None
        status = _normalize_status(status_raw)
        results[team] = GoalieInfo(team=team, goalie_name=goalie_name, status=status, source="puckpedia")

    return results


def _fetch_goalies_puckpedia_with_meta(day_count: int = 1) -> tuple[Dict[str, GoalieInfo], dict]:
    debug = os.getenv("NHL_GOALIES_DEBUG") == "1"
    html, status = _get_with_retry(
        PUCKPEDIA_URL,
        params={"dayCount": int(day_count)},
        headers={"Accept": "text/html"},
    )
    html_len = len(html or "")
    if debug:
        print(f"[nhl goalies] debug url={PUCKPEDIA_URL} status={status} html_len={html_len}")
    parsed = _parse_puckpedia(html)
    if debug:
        sample = [(t, g.goalie_name, g.status) for t, g in list(parsed.items())[:10]]
        print(f"[nhl goalies] debug parsed_rows={len(parsed)} sample={sample}")
        if not parsed:
            print(f"[nhl goalies] debug html_head={html[:500]}")
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

    try:
        parsed, meta = _fetch_goalies_puckpedia_with_meta(day_count=1)
        if parsed:
            _write_cached_goalies(cache_path, parsed, source="puckpedia", date_key=date)
            return parsed
        last_error = "puckpedia returned zero goalies"
        last_http_status = meta.get("status")
        last_html_len = meta.get("html_len")
        if debug:
            print(f"[nhl goalies] WARNING: empty puckpedia parse status={last_http_status} html_len={last_html_len}")
    except Exception as exc:
        last_error = f"puckpedia fetch failed: {exc}"
        if debug:
            print(f"[nhl goalies] WARNING: failed to fetch puckpedia: {exc}")

    providers = [("dailyfaceoff", DAILY_FACEOFF_URL, _parse_daily_faceoff)]
    for name, url, parser in providers:
        try:
            html, status = _get_with_retry(url, headers={"Accept": "text/html"})
            if debug:
                print(f"[nhl goalies] debug url={url} status={status} html_len={len(html or '')}")
            parsed = parser(html)
            if debug:
                sample = [(t, g.goalie_name, g.status) for t, g in list(parsed.items())[:10]]
                print(f"[nhl goalies] debug parsed_rows={len(parsed)} sample={sample}")
                if not parsed:
                    print(f"[nhl goalies] debug html_head={html[:500]}")
            if parsed:
                _write_cached_goalies(cache_path, parsed, source=name, date_key=date)
                return parsed
            last_error = f"{name} returned zero goalies"
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
        {},
        source="none",
        date_key=date,
        error=last_error or "no goalie providers succeeded",
        http_status=last_http_status,
        html_len=last_html_len,
    )
    return {}
