from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

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
    status_re = re.compile(r"\b(CONFIRMED|PROJECTED)\b", re.IGNORECASE)
    ignored_anchor_words = {"team", "lineup"}
    manual_slug_map = {
        "st-louis-blues": "St. Louis Blues",
        "new-york-rangers": "New York Rangers",
        "new-york-islanders": "New York Islanders",
        "new-jersey-devils": "New Jersey Devils",
        "utah-hockey-club": "Utah",
        "utah": "Utah",
    }

    def _team_from_slug(slug: str) -> str:
        if not slug:
            return ""
        if slug in manual_slug_map:
            return manual_slug_map[slug]
        name = slug.replace("-", " ").title()
        name = name.replace("St Louis", "St. Louis")
        name = name.replace("Ny Rangers", "NY Rangers")
        name = name.replace("Ny Islanders", "NY Islanders")
        name = name.replace("Ny Devils", "NY Devils")
        return name

    def _slug_from_href(href: str) -> str:
        try:
            path = urlparse(href).path or ""
        except Exception:
            path = href
        parts = [p for p in path.split("/") if p]
        if "team" in parts:
            idx = parts.index("team")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return parts[-1] if parts else ""

    def _text_from(node) -> str:
        return node.get_text(" ", strip=True) if node else ""

    def _status_from_block(block) -> str:
        if not block:
            return "UNKNOWN"
        text = " ".join(block.stripped_strings)
        match = status_re.search(text)
        if match:
            return _normalize_status(match.group(1))
        return "UNKNOWN"

    def _is_goalie_anchor(anchor) -> bool:
        href = anchor.get("href") or ""
        href_lower = href.lower()
        if "/team/" in href_lower:
            return False
        if "/player/" not in href_lower and "puckpedia.com/player/" not in href_lower:
            return False
        text_clean = _text_from(anchor).strip()
        if not text_clean or len(text_clean.split()) < 2:
            return False
        lower_text = text_clean.lower()
        if any(word in lower_text for word in ignored_anchor_words):
            return False
        return True

    def _status_near(anchor) -> str:
        for node in [anchor, anchor.parent, anchor.find_parent(["div", "td", "li", "p", "span"])]:
            status = _status_from_block(node)
            if status != "UNKNOWN":
                return status
        return "UNKNOWN"

    def _first_goalie_in_container(container) -> tuple[Optional[str], str]:
        if not container:
            return (None, "UNKNOWN")
        for anchor in container.select("a[href*='/player/'], a[href*='puckpedia.com/player/']"):
            if _is_goalie_anchor(anchor):
                goalie_name = _text_from(anchor)
                status = _status_near(anchor)
                return goalie_name, status
        return (None, "UNKNOWN")

    for team_anchor in soup.select("a[href*='/team/']"):
        href = team_anchor.get("href") or ""
        slug = _slug_from_href(href)
        team_guess = _team_from_slug(slug)
        team = canon_team(team_guess)
        if not team or team in results:
            continue
        container = None
        if team_anchor.parent and team_anchor.parent.parent:
            container = team_anchor.parent.parent
        if not container:
            container = team_anchor.find_parent(["section", "article", "div"])
        goalie_name, status = _first_goalie_in_container(container)
        if not goalie_name:
            continue
        results[team] = GoalieInfo(team=team, goalie_name=goalie_name, status=status, source="puckpedia")

    if len(results) < 10:
        for container in soup.select("section, article, div"):
            team_links = container.select("a[href*='/team/']")
            if len(team_links) != 2:
                continue
            player_links = [
                anchor
                for anchor in container.select("a[href*='/player/'], a[href*='puckpedia.com/player/']")
                if _is_goalie_anchor(anchor)
            ]
            if len(player_links) != 2:
                continue
            for team_anchor, player_anchor in zip(team_links, player_links):
                href = team_anchor.get("href") or ""
                slug = _slug_from_href(href)
                team_guess = _team_from_slug(slug)
                team = canon_team(team_guess)
                if not team or team in results:
                    continue
                goalie_name = _text_from(player_anchor)
                if not goalie_name:
                    continue
                status = _status_near(player_anchor)
                results[team] = GoalieInfo(team=team, goalie_name=goalie_name, status=status, source="puckpedia")

    if len(results) < 15:
        return {}
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
    team_links_count = len(soup.select("a[href*='/team/']"))
    player_links_count = len(soup.select("a[href*='/player/'], a[href*='puckpedia.com/player/']"))
    if debug:
        print(f"[nhl goalies] debug puckpedia team_links={team_links_count} player_links={player_links_count}")
    if team_links_count < 10 or player_links_count < 10:
        return {}, {"status": status, "html_len": html_len, "html": html}
    parsed = _parse_puckpedia(html)
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

    try:
        parsed, meta = _fetch_goalies_puckpedia_with_meta(day_count=1)
        parsed_count = len(parsed)
        if parsed_count >= 15:
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
        _write_cached_goalies(
            cache_path,
            {},
            source="puckpedia",
            date_key=date,
            error=last_error,
            http_status=last_http_status,
            html_len=last_html_len,
        )
        if debug:
            print(f"[nhl goalies] WARNING: empty puckpedia parse status={last_http_status} html_len={last_html_len}")
        return {}
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
            if debug:
                print(f"[nhl goalies] debug provider={name} url={url} status={status} html_len={len(html or '')}")
            parsed = parser(html)
            if debug:
                _debug_parse_details(name, url, status, html, parsed)
            if parsed:
                _write_cached_goalies(
                    cache_path,
                    parsed,
                    source=name,
                    date_key=date,
                    http_status=status,
                    html_len=len(html or ""),
                )
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
