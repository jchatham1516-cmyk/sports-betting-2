# sports/nfl/injuries.py
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

from sports.common.util import normalize_spaces, normalize_team_name

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)
ESPN_NFL_INJ_URL = "https://www.espn.com/nfl/injuries"

# Status → severity multiplier
INJURY_STATUS_MULTIPLIER = {
    "out": 1.0,
    "inactive": 1.0,
    "injured reserve": 1.0,
    "ir": 1.0,
    "doubtful": 0.75,
    "questionable": 0.45,
    "probable": 0.20,
    "suspended": 1.0,
    "physically unable to perform": 1.0,
    "pup": 1.0,
}

# Baseline leverage by position (tunable)
POS_IMPACT = {
    "QB": 4.0,
    "RB": 2.0, "FB": 1.4,
    "WR": 2.2, "TE": 1.8,
    "OT": 2.4, "T": 2.4, "OG": 2.0, "G": 2.0, "C": 1.9,
    "DE": 2.2, "DT": 2.0, "DL": 2.1,
    "LB": 2.0,
    "CB": 2.1, "S": 2.0, "DB": 2.0,
    "K": 1.0, "PK": 1.0, "P": 0.8,
}

# Starter-weighted scaling (NFL QBs matter a ton)
STARTER_QB_MULT = 1.75   # starting QB boost
BACKUP_QB_MULT = 1.20    # backup QB boost


def _status_to_mult(status: str) -> float:
    s = (status or "").strip().lower()
    for key, mult in INJURY_STATUS_MULTIPLIER.items():
        if key in s:
            return float(mult)
    return 0.25  # unknown status still has small effect


def _pos_to_impact(pos: str) -> float:
    p = (pos or "").strip().upper()
    if p in POS_IMPACT:
        return float(POS_IMPACT[p])
    p2 = p[:2]
    if p2 in POS_IMPACT:
        return float(POS_IMPACT[p2])
    return 1.6


def fetch_espn_nfl_injuries() -> Dict[str, List[dict]]:
    """
    Returns:
      { normalized_team_name: [ {"player","pos","status","comment"} , ... ] }
    """
    r = requests.get(ESPN_NFL_INJ_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    out: Dict[str, List[dict]] = {}

    titles = soup.select(".Table__Title")
    tables = soup.select("table.Table")

    # Fallback parsing if ESPN markup shifts
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

            name = re.sub(r"\s+", " ", name).strip()
            rows.append({"player": name, "pos": pos, "status": status, "comment": comment})

    return out


def build_injury_list_for_team_nfl(
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

        # role heuristic: treat key positions as starters more often
        role = "starter" if pos_u in {"QB", "OT", "T", "OG", "G", "C", "DE", "DT", "CB", "S", "LB"} else "rotation"

        # Starter-weighted QB scaling
        if pos_u == "QB":
            if role == "starter":
                impact *= STARTER_QB_MULT
            else:
                impact *= BACKUP_QB_MULT

        # bump if comment suggests season-ending
        c = comment.lower()
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

    home_cost = total(home_injuries)
    away_cost = total(away_injuries)
    return away_cost - home_cost
