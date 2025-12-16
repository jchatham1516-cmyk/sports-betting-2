import re
from datetime import date
from io import BytesIO
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from sports.common.util import normalize_spaces, normalize_team_name


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

NBA_TEAM_NAMES = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards"
]

STATUS_WORDS = {"Out", "Doubtful", "Questionable", "Probable"}

INJURY_STATUS_MULTIPLIER = {
    "out": 1.0,
    "doubt": 0.75,
    "question": 0.45,
    "prob": 0.20
}

INJURY_WEIGHTS = {
    "starter": 1.0,
    "rotation": 0.55
}


def status_to_mult(status):
    if not isinstance(status, str):
        return 1.0
    s = status.lower()
    for key, mult in INJURY_STATUS_MULTIPLIER.items():
        if key in s:
            return mult
    return 1.0


def estimate_player_impact_from_reason(reason: str) -> float:
    r = (reason or "").lower()
    base = 2.0
    if "illness" in r or "sick" in r or "personal" in r:
        base = 1.0
    if "surgery" in r or "recovery" in r or "stress reaction" in r or "fracture" in r:
        base = max(base, 2.6)
    if "achilles" in r:
        base = max(base, 3.2)
    if "knee" in r:
        base = max(base, 2.8)
    if "hamstring" in r:
        base = max(base, 2.4)
    if "ankle" in r:
        base = max(base, 2.2)
    if "foot" in r:
        base = max(base, 2.3)
    if "shoulder" in r:
        base = max(base, 2.1)
    if "concussion" in r or "protocol" in r:
        base = max(base, 1.8)
    if "thumb" in r or "hand" in r or "finger" in r or "toe" in r:
        base = min(base, 1.6) if base <= 2.0 else base
    return float(base)


def nba_injury_report_page_url(season_label: str):
    return f"https://official.nba.com/nba-injury-report-{season_label}-season/"


def get_latest_nba_injury_pdf_url_for_date(game_date_obj: date, season_label: str):
    url = nba_injury_report_page_url(season_label)
    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    month_name = game_date_obj.strftime("%B")
    day_num = str(int(game_date_obj.strftime("%d")))
    needle = f"{month_name} {day_num}:"

    container = None
    for node in soup.find_all(["p", "div", "li"]):
        txt = node.get_text(" ", strip=True)
        if txt.startswith(needle):
            container = node
            break

    if container is None:
        raise RuntimeError(f"Could not find injury report links for '{needle}' on {url}")

    links = container.find_all("a", href=True)
    pdfs = [
        a["href"] for a in links
        if "ak-static.cms.nba.com" in a["href"].lower() and a["href"].lower().endswith(".pdf")
    ]
    if not pdfs:
        raise RuntimeError(f"Found date block for '{needle}' but no PDF links.")

    return pdfs[-1]


def download_pdf_bytes(url: str, timeout: int = 30) -> bytes:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://official.nba.com/",
    }
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    data = r.content
    if not data.startswith(b"%PDF"):
        snippet = data[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"Downloaded content is not a PDF. First bytes/snippet: {snippet}")
    return data


def parse_tokens_to_injuries(tokens: list[str]) -> pd.DataFrame:
    team_tokens_map: list[tuple[int, list[str], str]] = []
    for t in NBA_TEAM_NAMES:
        tt = t.split()
        team_tokens_map.append((len(tt), tt, t))
    team_tokens_map.sort(reverse=True)

    def match_team(i: int):
        for L, tt, full in team_tokens_map:
            if i + L <= len(tokens) and tokens[i:i+L] == tt:
                return full, L
        return None, 0

    def is_header(tok: str) -> bool:
        return tok in {
            "Injury", "Report:", "Page", "of", "Game", "Date", "Time", "Matchup",
            "Team", "Player", "Name", "Current", "Status", "Reason", "(ET)"
        }

    rows: list[dict] = []
    current_team: Optional[str] = None
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok in {"NOT", "YET", "SUBMITTED"}:
            i += 1
            continue
        if is_header(tok) or tok.endswith("(ET)"):
            i += 1
            continue

        if "@" in tok and len(tok) <= 10:
            i += 1
            team, L = match_team(i)
            if team:
                current_team = team
                i += L
            continue

        team, L = match_team(i)
        if team:
            current_team = team
            i += L
            continue

        if "," in tok:
            player_parts = [tok]
            j = i + 1
            while j < len(tokens) and tokens[j] not in STATUS_WORDS:
                if "@" in tokens[j] and len(tokens[j]) <= 10:
                    break
                t2, _ = match_team(j)
                if t2:
                    break
                if is_header(tokens[j]):
                    break
                player_parts.append(tokens[j])
                j += 1

            if j >= len(tokens) or tokens[j] not in STATUS_WORDS:
                i += 1
                continue

            status = tokens[j]
            player = normalize_spaces(" ".join(player_parts))

            reason_parts = []
            k = j + 1
            while k < len(tokens):
                if tokens[k] in STATUS_WORDS:
                    break
                if "," in tokens[k]:
                    break
                if "@" in tokens[k] and len(tokens[k]) <= 10:
                    break
                t3, _ = match_team(k)
                if t3:
                    break
                if is_header(tokens[k]):
                    break
                reason_parts.append(tokens[k])
                k += 1

            reason = normalize_spaces(" ".join(reason_parts))
            if current_team:
                rows.append({"Team": current_team, "Player": player, "Status": status, "Reason": reason})

            i = k
            continue

        i += 1

    return pd.DataFrame(rows, columns=["Team", "Player", "Status", "Reason"])


def parse_nba_injury_pdf_to_df(pdf_bytes: bytes) -> pd.DataFrame:
    reader = PdfReader(BytesIO(pdf_bytes))
    all_text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t:
            all_text.append(t)
    raw = "\n".join(all_text)
    tokens = raw.replace("\xa0", " ").split()
    return parse_tokens_to_injuries(tokens)


def fetch_injury_report_official_nba(game_date_obj: date) -> pd.DataFrame:
    season_label = "2025-26"
    pdf_url = get_latest_nba_injury_pdf_url_for_date(game_date_obj, season_label=season_label)
    pdf_bytes = download_pdf_bytes(pdf_url)
    return parse_nba_injury_pdf_to_df(pdf_bytes)


def build_injury_list_for_team_official(team_name: str, injury_df: pd.DataFrame):
    if injury_df is None or injury_df.empty or "Team" not in injury_df.columns:
        return []

    team_norm = normalize_team_name(team_name)
    df = injury_df.copy()
    df["Team_norm"] = df["Team"].astype(str).apply(normalize_team_name)

    df_team = df[df["Team_norm"] == team_norm].copy()
    if df_team.empty:
        nick = team_norm.split()[-1]
        df_team = df[df["Team_norm"].str.contains(nick, na=False)].copy()
    if df_team.empty:
        return []

    injuries = []
    for _, row in df_team.iterrows():
        name = str(row.get("Player", "") or "").strip()
        status = str(row.get("Status", "") or "").strip()
        reason = str(row.get("Reason", "") or row.get("Injury", "") or "").strip()
        reason_l = reason.lower()

        if ("g league" in reason_l) or ("two-way" in reason_l) or ("two way" in reason_l) or ("on assignment" in reason_l):
            continue
        if not name or not status:
            continue

        role = "rotation"
        mult = status_to_mult(status)
        impact_points = estimate_player_impact_from_reason(reason)
        injuries.append((name, role, mult, impact_points))

    return injuries


def injury_adjustment(home_injuries=None, away_injuries=None):
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    def parse_inj_list(lst, sign):
        adj = 0.0
        for item in lst:
            if len(item) == 4:
                _, role, mult, impact = item
            else:
                continue
            weight = INJURY_WEIGHTS.get(role, INJURY_WEIGHTS["starter"])
            adj += sign * weight * mult * (impact / 2.0)
        return adj

    return parse_inj_list(home_injuries, -1.0) + parse_inj_list(away_injuries, +1.0)

