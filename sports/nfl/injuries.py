import re
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup

from sports.common.util import normalize_spaces, normalize_team_name

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
ESPN_NFL_INJ_URL = "https://www.espn.com/nfl/injuries"

# Returned list items match your NBA style:
# (player_name, role, status_multiplier, impact_points)
#
# role is mostly used as "starter" vs "rotation"; for NFL we keep it simple.
INJURY_STATUS_MULTIPLIER = {
    "out": 1.0,
    "injured reserve": 1.0,
    "ir": 1.0,
    "doubtful": 0.75,
    "questionable": 0.45,
    "probable": 0.20,
    "suspended": 1.0,
    "physically unable to perform": 1.0,
    "pup": 1.0,
}

# Very simple “importance” by position group (tunable)
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

def _status_to_mult(status: str) -> float:
    s = (status or "").strip().lower()
    for key, mult in INJURY_STATUS_MULTIPLIER.items():
        if key in s:
            return float(mult)
    # if unknown status, assume small effect (not 0)
    return 0.25

def _pos_to_impact(pos: str) -> float:
    p = (pos or "").strip().upper()
    if p in POS_IMPACT:
        return float(POS_IMPACT[p])
    # try first 2 chars for things like "CB", "DT"
    p2 = p[:2]
    if p2 in POS_IMPACT:
        return float(POS_IMPACT[p2])
    return 1.6

def fetch_espn_nfl_injuries() -> Dict[str, List[dict]]:
    """
    Returns dict keyed by *normalized team name* -> list of rows:
      {"player":..., "pos":..., "status":..., "comment":...}
    """
    r = requests.get(ESPN_NFL_INJ_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # ESPN injuries pages usually repeat blocks:
    # Team title (h2/h3 with Table__Title) then a table.
    out: Dict[str, List[dict]] = {}

    titles = soup.select(".Table__Title")
    tables = soup.select("table.Table")

    # Fallback: if selectors fail, grab any table and attempt to parse headings nearby
    if not titles or not tables:
        # Try generic: any <h2> followed by a table
        titles = soup.find_all(["h2", "h3"])
        tables = soup.find_all("table")

    ti = 0
    for table in tables:
        # find nearest title before this table
        team_name = None
        if ti < len(titles):
            team_name = titles[ti].get_text(" ", strip=True)
            ti += 1
        else:
            # fallback: walk backwards for a heading
            prev = table.find_previous(["h2", "h3"])
            if prev:
                team_name = prev.get_text(" ", strip=True)

        if not team_name:
            continue

        team_norm = normalize_team_name(team_name)
        rows = out.setdefault(team_norm, [])

        # Parse table rows
        tbody = table.find("tbody")
        if not tbody:
            continue
        for tr in tbody.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 4:
                continue
            # ESPN table columns usually: NAME | POS | EST RETURN | STATUS | COMMENT (comment sometimes missing)
            name = normalize_spaces(tds[0].get_text(" ", strip=True))
            pos = normalize_spaces(tds[1].get_text(" ", strip=True))
            status = normalize_spaces(tds[3].get_text(" ", strip=True))
            comment = normalize_spaces(tds[4].get_text(" ", strip=True)) if len(tds) >= 5 else ""

            if not name or not status:
                continue
            # clean weird footnotes
            name = re.sub(r"\s+", " ", name).strip()

            rows.append({"player": name, "pos": pos, "status": status, "comment": comment})

    return out

def build_injury_list_for_team_nfl(team_name: str, injuries_map: Dict[str, List[dict]]) -> List[Tuple[str, str, float, float]]:
    """
    Converts ESPN injury rows into your model’s injury tuple format:
      (player, role, status_mult, impact_points)
    """
    if not injuries_map:
        return []

    team_norm = normalize_team_name(team_name)
    rows = injuries_map.get(team_norm)

    # fallback: loose match by last word (e.g. "Giants", "Jets")
    if rows is None:
        nick = team_norm.split()[-1] if team_norm else ""
        for k, v in injuries_map.items():
            if nick and nick in k:
                rows = v
                break

    if not rows:
        return []

    out = []
    for r in rows:
        name = (r.get("player") or "").strip()
        pos = (r.get("pos") or "").strip()
        status = (r.get("status") or "").strip()
        comment = (r.get("comment") or "").strip()

        if not name or not status:
            continue

        mult = _status_to_mult(status)
        impact = _pos_to_impact(pos)

        # role: treat QB/OL/DL/DB as "starter-ish" by default
        role = "starter" if pos.upper() in {"QB", "OT", "T", "OG", "G", "C", "DE", "DT", "CB", "S", "LB"} else "rotation"

        # tiny bump if comment suggests season-ending
        c = comment.lower()
        if "out for the season" in c or "season-ending" in c:
            mult = max(mult, 1.0)

        out.append((name, role, float(mult), float(impact)))

    return out

def injury_adjustment_points(home_injuries=None, away_injuries=None) -> float:
    """
    Positive means advantage HOME (away more hurt),
    Negative means disadvantage HOME (home more hurt).
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    role_w = {"starter": 1.0, "rotation": 0.55}

    def total(lst):
        s = 0.0
        for item in lst:
            if len(item) != 4:
                continue
            _, role, mult, impact = item
            s += role_w.get(role, 0.6) * float(mult) * float(impact)
        return s

    home_cost = total(home_injuries)
    away_cost = total(away_injuries)
    return (away_cost - home_cost)
