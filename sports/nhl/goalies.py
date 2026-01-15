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
PUCKPEDIA_URL = "https://www.puckpedia.com/lineups"


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
) -> tuple[str, int]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
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


def _write_cached_goalies(cache_path: str, goalies: Dict[str, GoalieInfo]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "goalies": {
            team: {
                "goalie_name": info.goalie_name,
                "status": info.status,
                "source": info.source,
            }
            for team, info in goalies.items()
        }
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def get_starting_goalies(date: str) -> Dict[str, GoalieInfo]:
    cache_path = os.path.join(GOALIE_CACHE_DIR, f"nhl_goalies_{date}.json")
    cached = _load_cached_goalies(cache_path)
    if cached:
        return cached

    providers = [
        ("dailyfaceoff", DAILY_FACEOFF_URL, _parse_daily_faceoff),
        ("puckpedia", PUCKPEDIA_URL, _parse_puckpedia),
    ]

    debug = os.getenv("NHL_GOALIES_DEBUG") == "1"
    for _, url, parser in providers:
        try:
            html, status = _get_with_retry(url)
            if debug:
                print(f"[nhl goalies] debug url={url} status={status} html_len={len(html or '')}")
            parsed = parser(html)
            if debug:
                sample = [(t, g.goalie_name, g.status) for t, g in list(parsed.items())[:10]]
                print(f"[nhl goalies] debug parsed_rows={len(parsed)} sample={sample}")
            if parsed:
                _write_cached_goalies(cache_path, parsed)
                return parsed
            print(f"[nhl goalies] WARNING: empty parse status={status} html_len={len(html or '')} url={url}")
        except Exception as exc:
            print(f"[nhl goalies] WARNING: failed to fetch {url}: {exc}")

    _write_cached_goalies(cache_path, {})
    return {}
