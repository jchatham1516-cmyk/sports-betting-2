# sports/nba/injuries.py
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

from sports.common.teams import canon_team

# If you have util helpers, we use them; otherwise we do minimal normalization safely.
try:
    from sports.common.util import normalize_spaces
except Exception:
    def normalize_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)

# Fallback injuries source (very stable)
ESPN_NBA_INJ_URL = "https://www.espn.com/nba/injuries"

# Official NBA report page (often changes; we try it first)
NBA_OFFICIAL_INJ_URL = "https://official.nba.com/nba-injury-report-2025-26-season/"


# -----------------------------
# Status scaling
# -----------------------------
# Multiplier means "how much of the impact counts"
# OUT -> 1.0, QUESTIONABLE -> partial, PROBABLE -> small
STATUS_MULT = {
    "out": 1.0,
    "out for season": 1.0,
    "dnp": 1.0,
    "injured": 1.0,

    "doubtful": 0.75,
    "questionable": 0.45,
    "probable": 0.20,

    # If sites use these:
    "available": 0.0,
    "active": 0.0,
    "in": 0.15,
}


def _status_to_mult(status: str) -> float:
    s = (status or "").strip().lower()
    if not s:
        return 0.25
    # check longest keys first (e.g., "out for season")
    for key in sorted(STATUS_MULT.keys(), key=len, reverse=True):
        if key in s:
            return float(STATUS_MULT[key])
    # unknown -> small but nonzero
    return 0.25


# -----------------------------
# Position impact (NBA)
# -----------------------------
# NBA positions matter but not massively; we still weight bigs slightly.
POS_IMPACT = {
    "PG": 1.9,
    "SG": 1.8,
    "SF": 1.85,
    "PF": 1.95,
    "C": 2.05,
    "G": 1.85,
    "F": 1.90,
}


def _pos_to_impact(pos: str) -> float:
    p = (pos or "").strip().upper()
    if p in POS_IMPACT:
        return float(POS_IMPACT[p])
    # sometimes ESPN uses "G-F" etc
    if "-" in p:
        parts = [x.strip() for x in p.split("-") if x.strip()]
        vals = [POS_IMPACT.get(x, 1.85) for x in parts]
        return float(sum(vals) / max(len(vals), 1))
    return 1.85


# -----------------------------
# Fetchers
# -----------------------------
def _fetch_from_espn() -> Dict[str, List[dict]]:
    """
    ESPN injuries page parser.
    Returns:
      dict[canon_team(team)] -> [{"player":..,"pos":..,"status":..,"comment":..}, ...]
    """
    r = requests.get(ESPN_NBA_INJ_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    out: Dict[str, List[dict]] = {}

    titles = soup.select(".Table__Title")
    tables = soup.select("table.Table")

    # fallback selectors
    if not titles or not tables:
        titles = soup.find_all(["h2", "h3"])
        tables = soup.find_all("table")

    ti = 0
    for table in tables:
        team_name = None
        if ti < len(titles):
            team_name = titles[ti].get_text(" ", strip=True)
            ti += 1
        else:
            prev = table.find_previous(["h2", "h3"])
            if prev:
                team_name = prev.get_text(" ", strip=True)

        if not team_name:
            continue

        team_key = canon_team(team_name)
        rows = out.setdefault(team_key, [])

        tbody = table.find("tbody")
        if not tbody:
            continue

        for tr in tbody.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 4:
                continue

            # ESPN typically: NAME | POS | EST RETURN | STATUS | COMMENT
            name = normalize_spaces(tds[0].get_text(" ", strip=True))
            pos = normalize_spaces(tds[1].get_text(" ", strip=True))
            status = normalize_spaces(tds[3].get_text(" ", strip=True))
            comment = normalize_spaces(tds[4].get_text(" ", strip=True)) if len(tds) >= 5 else ""

            if not name or not status:
                continue

            name = re.sub(r"\s+", " ", name).strip()
            rows.append({"player": name, "pos": pos, "status": status, "comment": comment})

    return out


def _fetch_from_official_nba() -> Dict[str, List[dict]]:
    """
    Tries to scrape the official NBA injury report page.
    This page changes often, so this is best-effort.
    If parsing fails, caller should fall back to ESPN.
    """
    r = requests.get(NBA_OFFICIAL_INJ_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    html = r.text

    # Very common: page contains links to "Injury Report: <date>" PDFs.
    # If you already had a working PDF parser earlier, use it there.
    # Here we just attempt to parse any visible injury tables if present.
    soup = BeautifulSoup(html, "lxml")

    out: Dict[str, List[dict]] = {}

    # Heuristic: sometimes tables include team names in headers and rows with players.
    # We'll try generic table parsing.
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError("No tables found on official NBA injuries page")

    # If the official site is not giving parseable HTML tables, this will likely fail.
    # That’s OK — ESPN fallback will take over.
    for table in tables[:6]:  # don't over-scan
        # Find a heading near the table that looks like a team
        heading = table.find_previous(["h2", "h3", "h4"])
        team_name = heading.get_text(" ", strip=True) if heading else None
        if not team_name:
            continue

        team_key = canon_team(team_name)
        rows = out.setdefault(team_key, [])

        tbody = table.find("tbody")
        if not tbody:
            continue

        for tr in tbody.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 3:
                continue

            name = normalize_spaces(tds[0].get_text(" ", strip=True))
            pos = normalize_spaces(tds[1].get_text(" ", strip=True))
            status = normalize_spaces(tds[2].get_text(" ", strip=True))
            comment = normalize_spaces(tds[3].get_text(" ", strip=True)) if len(tds) >= 4 else ""

            if not name or not status:
                continue

            rows.append({"player": name, "pos": pos, "status": status, "comment": comment})

    # If we got basically nothing, treat it as failure so ESPN kicks in.
    total = sum(len(v) for v in out.values())
    if total < 5:
        raise RuntimeError("Official NBA injuries parse produced too few rows")

    return out


def fetch_official_nba_injuries() -> Dict[str, List[dict]]:
    """
    Primary entrypoint used by sports/nba/model.py.

    Returns:
      dict keyed by canonical team name -> list of dict rows
    """
    try:
        return _fetch_from_official_nba()
    except Exception as e:
        print(f"[nba injuries] WARNING: official NBA injury page failed ({e}); falling back to ESPN")
        try:
            return _fetch_from_espn()
        except Exception as e2:
            print(f"[nba injuries] WARNING: ESPN injuries failed too ({e2}); using empty injuries")
            return {}


# -----------------------------
# Convert map -> model tuples
# -----------------------------
def build_injury_list_for_team_nba(
    team_name: str,
    injuries_map: Dict[str, List[dict]],
) -> List[Tuple[str, str, float, float]]:
    """
    Converts injuries_map team rows into tuples:
      (player_name, role, status_multiplier, impact_points)

    role: "starter" or "rotation" (heuristic)
    """
    if not injuries_map:
        return []

    team_key = canon_team(team_name)
    rows = injuries_map.get(team_key)

    # fallback: loose match by last word (nickname)
    if rows is None:
        nick = team_key.split()[-1] if team_key else ""
        for k, v in injuries_map.items():
            if nick and nick in k:
                rows = v
                break

    if not rows:
        return []

    out: List[Tuple[str, str, float, float]] = []

    for rr in rows:
        name = (rr.get("player") or "").strip()
        pos = (rr.get("pos") or "").strip()
        status = (rr.get("status") or "").strip()
        comment = (rr.get("comment") or "").strip()

        if not name or not status:
            continue

        mult = _status_to_mult(status)
        impact = _pos_to_impact(pos)

        # Starter heuristic:
        # - If OUT/DOUBTFUL => more likely to matter
        # - If comment hints "starter" or "starting"
        # - Otherwise rotation
        role = "rotation"
        s = status.lower()
        c = comment.lower()
        if ("out" in s) or ("doubtful" in s) or ("starter" in c) or ("starting" in c):
            role = "starter"

        # Extra bump for clear season-ending notes
        if "out for the season" in c or "season-ending" in c:
            mult = max(mult, 1.0)
            role = "starter"

        out.append((name, role, float(mult), float(impact)))

    return out


def injury_adjustment_points(home_injuries=None, away_injuries=None) -> float:
    """
    Positive means advantage HOME (away more hurt),
    Negative means disadvantage HOME (home more hurt).

    Uses starter-weighted scaling.
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    # Starter weight matters more in NBA
    role_w = {"starter": 1.0, "rotation": 0.55}

    def total(lst):
        s = 0.0
        for item in lst:
            if not item or len(item) != 4:
                continue
            _, role, mult, impact = item
            s += role_w.get(role, 0.6) * float(mult) * float(impact)
        return s

    home_cost = total(home_injuries)
    away_cost = total(away_injuries)

    # Positive means away is more hurt -> advantage home
    return float(away_cost - home_cost)
