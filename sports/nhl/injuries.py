# sports/nhl/injuries.py
from __future__ import annotations

from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

from sports.common.util import normalize_spaces, normalize_team_name

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)
ESPN_NHL_INJ_URL = "https://www.espn.com/nhl/injuries"

INJURY_STATUS_MULTIPLIER = {
    "out": 1.0,
    "injured reserve": 1.0,
    "ir": 1.0,
    "doubtful": 0.75,
    "questionable": 0.45,
    "day-to-day": 0.35,
}

# Baseline leverage by position (tunable)
POS_IMPACT = {
    "G": 3.2,
    "D": 1.9, "LD": 1.9, "RD": 1.9,
    "C": 1.8,
    "LW": 1.7, "RW": 1.7, "F": 1.7,
}

# Starter-weighted goalie scaling
STARTER_G_MULT = 1.65  # likely starter / #1
OTHER_G_MULT = 1.25    # goalie in general


def _status_to_mult(status: str) -> float:
    s = (status or "").strip().lower()
    for k, v in INJURY_STATUS_MULTIPLIER.items():
        if k in s:
            return float(v)
    return 0.25


def _pos_to_impact(pos: str) -> float:
    p = (pos or "").strip().upper()
    if p in POS_IMPACT:
        return float(POS_IMPACT[p])
    return 1.6


def fetch_espn_nhl_injuries() -> Dict[str, List[dict]]:
    """
    Returns:
      { normalized_team_name: [ {"player","pos","status","comment"} , ... ] }
    """
    r = requests.get(ESPN_NHL_INJ_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    out: Dict[str, List[dict]] = {}

    titles = soup.select(".Table__Title")
    tables = soup.select("table.Table")

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

        team_norm = normalize_team_name(team_name)
        rows = out.setdefault(team_norm, [])

        tbody = table.find("tbody")
        if not tbody:
            continue

        for tr in tbody.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 4:
                continue

            # ESPN typical columns: NAME | POS | EST RETURN | STATUS | COMMENT (optional)
            name = normalize_spaces(tds[0].get_text(" ", strip=True))
            pos = normalize_spaces(tds[1].get_text(" ", strip=True))
            status = normalize_spaces(tds[3].get_text(" ", strip=True))
            comment = normalize_spaces(tds[4].get_text(" ", strip=True)) if len(tds) >= 5 else ""

            if not name or not status:
                continue

            rows.append({"player": name, "pos": pos, "status": status, "comment": comment})

    return out


def build_injury_list_for_team_nhl(
    team_name: str,
    injuries_map: Dict[str, List[dict]],
) -> List[Tuple[str, str, float, float]]:
    """
    Converts injury rows into tuples:
      (player, role, status_mult, impact_points)
    """
    if not injuries_map:
        return []

    team_norm = normalize_team_name(team_name)
    rows = injuries_map.get(team_norm)

    # loose fallback match by nickname
    if rows is None:
        nick = team_norm.split()[-1] if team_norm else ""
        for k, v in injuries_map.items():
            if nick and nick in k:
                rows = v
                break

    if not rows:
        return []

    out: List[Tuple[str, str, float, float]] = []

    for r in rows:
        name = (r.get("player") or "").strip()
        pos = (r.get("pos") or "").strip()
        status = (r.get("status") or "").strip()
        comment = (r.get("comment") or "").strip()

        if not name or not status:
            continue

        mult = _status_to_mult(status)
        impact = _pos_to_impact(pos)

        pos_u = pos.upper()
        c = comment.lower()

        # Role heuristic
        role = "starter" if pos_u in {"G", "D", "C"} else "rotation"

        # Starter-weighted goalie scaling
        if pos_u == "G":
            # comment hints at starter/#1
            if ("starter" in c) or ("starting" in c) or ("no. 1" in c) or ("number 1" in c) or ("no.1" in c):
                impact *= STARTER_G_MULT
            else:
                impact *= OTHER_G_MULT

        # bump if comment suggests season-ending
        if "out for the season" in c or "season-ending" in c:
            mult = max(mult, 1.0)

        out.append((name, role, float(mult), float(impact)))

    return out


def injury_adjustment_points(home_injuries=None, away_injuries=None) -> float:
    """
    Positive => advantage HOME (away more hurt)
    Negative => disadvantage HOME (home more hurt)
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    role_w = {"starter": 1.0, "rotation": 0.55}

    def total(lst) -> float:
        s = 0.0
        for item in lst:
            if len(item) != 4:
                continue
            _, role, mult, impact = item
            s += role_w.get(role, 0.6) * float(mult) * float(impact)
        return s

    return total(away_injuries) - total(home_injuries)
