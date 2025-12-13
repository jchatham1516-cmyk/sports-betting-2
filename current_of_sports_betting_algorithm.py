# current_of_sports_betting_algorithm.py
#
# Daily NBA betting model using BallDontLie + Official NBA injury report PDF + local odds CSV.
#
# Adds:
# - Play/Pass filter
# - Bankroll sizing (flat vs Kelly)
# - Units (1 unit = 4% of bankroll; default bankroll=250)
# - Backtest mode using BallDontLie final scores + your odds CSVs
# - ✅ NEW: reduced unit size for HIGH VALUE bets with MEDIUM/LOW confidence
#
# Requires:
# - recommendations.py in same folder (add_recommendations_to_df, Thresholds, etc.)

import os
import re
import math
import argparse
import time
from datetime import datetime, date, timedelta
from io import BytesIO
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from recommendations import add_recommendations_to_df, Thresholds


# -----------------------------
# Global tuning constants (model)
# -----------------------------

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

SPREAD_SCALE_FACTOR = 4.0
RECENT_FORM_WEIGHT = 0.35
SEASON_FORM_WEIGHT = 1.0 - RECENT_FORM_WEIGHT
RECENT_GAMES_WINDOW = 10


# -----------------------------
# Bankroll / unit settings
# -----------------------------
DEFAULT_BANKROLL = 250.0
UNIT_PCT = 0.04  # 4% of bankroll per unit (bankroll=250 -> 1 unit=$10)


# -----------------------------
# Helpers
# -----------------------------

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ").strip())

def normalize_team_name(s: str) -> str:
    s = normalize_spaces(s).lower()
    s = s.replace(".", "")
    return s

def season_start_year_for_date(d: date) -> int:
    return d.year - 1 if d.month < 8 else d.year

def american_to_implied_prob(odds):
    odds = float(odds)
    if odds < 0:
        p = (-odds) / ((-odds) + 100.0)
    else:
        p = 100.0 / (odds + 100.0)
    return max(min(p, 0.9999), 0.0001)

def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)

def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


# -----------------------------
# CSV odds loader
# -----------------------------

def fetch_odds_for_date_from_csv(game_date_str: str):
    """
    Expects: odds/odds_MM-DD-YYYY.csv
    columns: date, home, away, home_ml, away_ml, home_spread
    """
    fname = os.path.join("odds", f"odds_{game_date_str.replace('/', '-')}.csv")
    if not os.path.exists(fname):
        print(f"[odds_csv] No odds file found at {fname}.")
        return {}, {}

    print(f"[odds_csv] Loading odds from {fname}")
    df = pd.read_csv(fname)

    required_cols = {"home", "away", "home_ml", "away_ml"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Odds CSV {fname} must contain columns: {required_cols}. Found: {list(df.columns)}"
        )

    def _parse_number(val):
        if pd.isna(val):
            return None
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                return None
            try:
                return float(s)
            except ValueError:
                return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    odds_dict = {}
    spreads_dict = {}

    for _, row in df.iterrows():
        home = str(row["home"]).strip()
        away = str(row["away"]).strip()
        key = (home, away)

        home_ml = _parse_number(row.get("home_ml"))
        away_ml = _parse_number(row.get("away_ml"))
        home_spread = _parse_number(row.get("home_spread")) if "home_spread" in df.columns else None

        odds_dict[key] = {"home_ml": home_ml, "away_ml": away_ml, "home_spread": home_spread}
        if home_spread is not None:
            spreads_dict[key] = home_spread

    return odds_dict, spreads_dict


# -----------------------------
# BallDontLie low-level client
# -----------------------------

BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"

def get_bdl_api_key():
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError("BALLDONTLIE_API_KEY environment variable is not set.")
    return api_key

def bdl_get(path, params=None, api_key=None, max_retries=5):
    if api_key is None:
        api_key = get_bdl_api_key()

    url = BALLDONTLIE_BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Authorization": api_key}
    params = dict(params or {})

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after is not None else 15
                print(f"[bdl_get] Rate limited (429) on {path}, attempt {attempt}/{max_retries}. Sleeping {wait}s...")
                time.sleep(wait)
                last_exc = RuntimeError("Rate limited by BallDontLie")
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout as e:
            last_exc = e
            print(f"[bdl_get] Timeout calling {path} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"[bdl_get] HTTP error calling {path} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(5)

    raise RuntimeError(f"Failed to GET {path} from BallDontLie after {max_retries} attempts") from last_exc


# -----------------------------
# Team ratings built from games
# -----------------------------

def fetch_team_ratings_bdl(season_year: int, end_date_iso: str, api_key: str):
    teams_json = bdl_get("teams", params={}, api_key=api_key)
    teams_data = teams_json.get("data", [])

    agg = {}
    games_by_team = {}
    for t in teams_data:
        tid = t["id"]
        agg[tid] = {
            "TEAM_NAME": t["full_name"],
            "gp": 0,
            "pts_for": 0,
            "pts_against": 0,
            "wins": 0,
            "losses": 0,
        }
        games_by_team[tid] = []

    params = {"seasons[]": season_year, "end_date": end_date_iso, "per_page": 100}
    cursor = None

    while True:
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        games_json = bdl_get("games", params=params, api_key=api_key)
        games = games_json.get("data", [])
        meta = games_json.get("meta", {}) or {}
        cursor = meta.get("next_cursor")

        for g in games:
            home_team = g["home_team"]
            away_team = g["visitor_team"]

            home_id = home_team["id"]
            away_id = away_team["id"]
            home_score = g.get("home_team_score", 0) or 0
            away_score = g.get("visitor_team_score", 0) or 0

            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue

            g_date_str = g.get("date")
            try:
                g_date = datetime.fromisoformat(g_date_str.replace("Z", "+00:00")).date()
            except Exception:
                g_date = None

            if home_id in agg:
                agg[home_id]["gp"] += 1
                agg[home_id]["pts_for"] += home_score
                agg[home_id]["pts_against"] += away_score
                if g_date:
                    games_by_team[home_id].append({"date": g_date, "pts_for": home_score, "pts_against": away_score})

            if away_id in agg:
                agg[away_id]["gp"] += 1
                agg[away_id]["pts_for"] += away_score
                agg[away_id]["pts_against"] += home_score
                if g_date:
                    games_by_team[away_id].append({"date": g_date, "pts_for": away_score, "pts_against": home_score})

            if home_score > away_score:
                if home_id in agg:
                    agg[home_id]["wins"] += 1
                if away_id in agg:
                    agg[away_id]["losses"] += 1
            elif away_score > home_score:
                if away_id in agg:
                    agg[away_id]["wins"] += 1
                if home_id in agg:
                    agg[home_id]["losses"] += 1

        if not cursor:
            break

    rows = []
    for tid, rec in agg.items():
        gp = rec["gp"]
        wins = rec["wins"]
        losses = rec["losses"]
        w_pct = wins / gp if gp > 0 else 0.0
        or_p = rec["pts_for"] / gp if gp > 0 else 0.0
        dr_p = rec["pts_against"] / gp if gp > 0 else 0.0

        total_pts = rec["pts_for"] + rec["pts_against"]
        pace = total_pts / gp if gp > 0 else 0.0

        poss = max(total_pts, 1)
        off_eff = rec["pts_for"] / poss
        def_eff = rec["pts_against"] / poss

        recent_games = sorted(games_by_team[tid], key=lambda x: x["date"])[-RECENT_GAMES_WINDOW:]
        gp_recent = len(recent_games)

        if gp_recent > 0:
            pts_for_recent = sum(g["pts_for"] for g in recent_games)
            pts_against_recent = sum(g["pts_against"] for g in recent_games)
            total_pts_recent = pts_for_recent + pts_against_recent

            or_p_recent = pts_for_recent / gp_recent
            dr_p_recent = pts_against_recent / gp_recent
            pace_recent = total_pts_recent / gp_recent
            poss_recent = max(total_pts_recent, 1)
            off_eff_recent = pts_for_recent / poss_recent
            def_eff_recent = pts_against_recent / poss_recent
        else:
            or_p_recent, dr_p_recent, pace_recent = or_p, dr_p, pace
            off_eff_recent, def_eff_recent = off_eff, def_eff

        rows.append({
            "TEAM_ID": tid,
            "TEAM_NAME": rec["TEAM_NAME"],
            "GP": gp,
            "W": wins,
            "L": losses,
            "W_PCT": w_pct,
            "ORtg": or_p,
            "DRtg": dr_p,
            "PACE": pace,
            "OFF_EFF": off_eff,
            "DEF_EFF": def_eff,
            "ORtg_RECENT": or_p_recent,
            "DRtg_RECENT": dr_p_recent,
            "PACE_RECENT": pace_recent,
            "OFF_EFF_RECENT": off_eff_recent,
            "DEF_EFF_RECENT": def_eff_recent,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Team lookup + scoring model
# -----------------------------

def find_team_row(team_name_input, stats_df):
    name = team_name_input.strip().lower()
    full_match = stats_df[stats_df["TEAM_NAME"].str.lower() == name]
    if not full_match.empty:
        return full_match.iloc[0]

    contains_match = stats_df[stats_df["TEAM_NAME"].str.lower().str.contains(name)]
    if not contains_match.empty:
        return contains_match.iloc[0]

    raise ValueError(f"Could not find a team matching: {team_name_input}")

MATCHUP_WEIGHTS = np.array([0.2, 0.12, 0.12, 0.03, 4.0, 4.0])

def _blend_stat(row, base_col, recent_col):
    base_val = float(row[base_col])
    recent_val = float(row[recent_col]) if recent_col in row.index else base_val
    return SEASON_FORM_WEIGHT * base_val + RECENT_FORM_WEIGHT * recent_val

def build_matchup_features(home_row, away_row):
    h_ORtg = _blend_stat(home_row, "ORtg", "ORtg_RECENT")
    a_ORtg = _blend_stat(away_row, "ORtg", "ORtg_RECENT")
    h_DRtg = _blend_stat(home_row, "DRtg", "DRtg_RECENT")
    a_DRtg = _blend_stat(away_row, "DRtg", "DRtg_RECENT")
    h_PACE = _blend_stat(home_row, "PACE", "PACE_RECENT")
    a_PACE = _blend_stat(away_row, "PACE", "PACE_RECENT")
    h_OFF = _blend_stat(home_row, "OFF_EFF", "OFF_EFF_RECENT")
    a_OFF = _blend_stat(away_row, "OFF_EFF", "OFF_EFF_RECENT")
    h_DEF = _blend_stat(home_row, "DEF_EFF", "DEF_EFF_RECENT")
    a_DEF = _blend_stat(away_row, "DEF_EFF", "DEF_EFF_RECENT")

    d_ORtg = h_ORtg - a_ORtg
    d_DRtg = a_DRtg - h_DRtg
    d_pace = h_PACE - a_PACE
    d_off_eff = h_OFF - a_OFF
    d_def_eff = a_DEF - h_DEF

    home_edge = 1.0
    return np.array([home_edge, d_ORtg, d_DRtg, d_pace, d_off_eff, d_def_eff], dtype=float)

def season_matchup_base_score(home_row, away_row):
    return float(np.dot(MATCHUP_WEIGHTS, build_matchup_features(home_row, away_row)))

def score_to_prob(score, lam=0.25):
    return 1.0 / (1.0 + math.exp(-lam * score))

def score_to_spread(score, points_per_logit=SPREAD_SCALE_FACTOR):
    """Vegas-style HOME spread: negative=home favored, positive=home dog."""
    s = float(score)
    return -(s * points_per_logit + (s ** 2) * 1.5)


# -----------------------------
# Injuries (your logic retained)
# -----------------------------

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
    current_team: str | None = None
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


# -----------------------------
# Schedule / games (BallDontLie)
# -----------------------------

def fetch_games_for_date(game_date_str, api_key):
    dt = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    iso_date = dt.strftime("%Y-%m-%d")
    params = {"dates[]": iso_date, "per_page": 100}
    games_json = bdl_get("games", params=params, api_key=api_key)
    games = games_json.get("data", [])

    rows = []
    for g in games:
        home_team = g["home_team"]
        away_team = g["visitor_team"]
        rows.append({
            "GAME_ID": g["id"],
            "HOME_TEAM_NAME": home_team["full_name"],
            "AWAY_TEAM_NAME": away_team["full_name"],
            "HOME_TEAM_ID": home_team["id"],
            "AWAY_TEAM_ID": away_team["id"],
            "GAME_DATE": g.get("date"),
            "HOME_SCORE": g.get("home_team_score"),
            "AWAY_SCORE": g.get("visitor_team_score"),
        })
    return pd.DataFrame(rows)

def build_odds_csv_template_if_missing(game_date_str, api_key, odds_dir="odds"):
    os.makedirs(odds_dir, exist_ok=True)
    odds_path = os.path.join(odds_dir, f"odds_{game_date_str.replace('/', '-')}.csv")
    if os.path.exists(odds_path):
        return odds_path

    print(f"[template] No odds file found for {game_date_str}, creating template at {odds_path}...")
    games_df = fetch_games_for_date(game_date_str, api_key=api_key)
    if games_df.empty:
        print(f"[template] No games found on {game_date_str}; not creating odds template.")
        return odds_path

    out_rows = []
    for _, row in games_df.iterrows():
        out_rows.append({
            "date": game_date_str,
            "home": row["HOME_TEAM_NAME"],
            "away": row["AWAY_TEAM_NAME"],
            "home_ml": "",
            "away_ml": "",
            "home_spread": "",
        })

    pd.DataFrame(out_rows).to_csv(odds_path, index=False)
    print(f"[template] Template odds file created: {odds_path}")
    return odds_path

def compute_head_to_head_adjustment(home_team_id, away_team_id, season_year, api_key, max_seasons_back=3):
    return 0.0

def get_team_last_game_date(team_id, game_date_obj, season_year, api_key):
    iso_date = game_date_obj.strftime("%Y-%m-%d")
    params = {"seasons[]": season_year, "team_ids[]": team_id, "end_date": iso_date, "per_page": 100}
    cursor = None
    last_date = None

    while True:
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        games_json = bdl_get("games", params=params, api_key=api_key)
        games = games_json.get("data", [])
        meta = games_json.get("meta", {}) or {}
        cursor = meta.get("next_cursor")

        for g in games:
            g_date_str = g.get("date")
            if not g_date_str:
                continue
            g_date = datetime.fromisoformat(g_date_str.replace("Z", "+00:00")).date()
            if g_date >= game_date_obj:
                continue

            home_score = g.get("home_team_score", 0) or 0
            away_score = g.get("visitor_team_score", 0) or 0
            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue

            if (last_date is None) or (g_date > last_date):
                last_date = g_date

        if not cursor:
            break

    return last_date

def rest_days_to_fatigue_adjustment(days_rest):
    if days_rest is None:
        return 0.0
    if days_rest <= 1:
        return -2.0
    if days_rest == 2:
        return -1.0
    if days_rest >= 4:
        return +0.5
    return 0.0


# -----------------------------
# Play/Pass + bet sizing
# -----------------------------

def _tier_rank(tier: str) -> int:
    # must match what recommendations.py outputs
    t = (tier or "").upper().strip()
    if "HIGH" in t:
        return 2
    if "MEDIUM" in t:
        return 1
    return 0  # NO VALUE / unknown

def _conf_rank(conf: str) -> int:
    c = (conf or "").upper().strip()
    if c == "HIGH":
        return 2
    if c == "MEDIUM":
        return 1
    return 0

def play_pass_rule(
    row: pd.Series,
    *,
    require_pick: bool = True,
    min_value_tier: str = "HIGH VALUE",     # ✅ changed: MIN tier rather than exact match
    min_confidence: str = "MEDIUM",         # LOW / MEDIUM / HIGH
    max_abs_moneyline: Optional[int] = 400, # skip extreme prices in ML
) -> str:
    primary = str(row.get("primary_recommendation", ""))
    value_tier = str(row.get("value_tier", ""))
    conf = str(row.get("confidence", ""))

    if require_pick and ("PICK" not in primary):
        return "PASS"

    if min_value_tier:
        if _tier_rank(value_tier) < _tier_rank(min_value_tier):
            return "PASS"

    if _conf_rank(conf) < _conf_rank(min_confidence):
        return "PASS"

    if max_abs_moneyline is not None and "ML" in primary:
        hm = safe_float(row.get("home_ml"))
        am = safe_float(row.get("away_ml"))
        if "HOME ML" in primary and hm is not None and abs(hm) > max_abs_moneyline:
            return "PASS"
        if "AWAY ML" in primary and am is not None and abs(am) > max_abs_moneyline:
            return "PASS"

    return "PLAY"

def kelly_fraction(p: float, odds_american: float) -> float:
    d = american_to_decimal(odds_american)
    b = d - 1.0
    q = 1.0 - p
    frac = (b * p - q) / b
    return max(frac, 0.0)

def bet_size_flat(bankroll: float, flat_pct: float) -> float:
    return max(bankroll * float(flat_pct), 0.0)

def bet_size_kelly_ml(
    bankroll: float,
    p: float,
    odds_american: float,
    *,
    kelly_mult: float = 0.5,
    max_pct: float = 0.03,
) -> float:
    f = kelly_fraction(float(p), float(odds_american))
    f_adj = min(f * float(kelly_mult), float(max_pct))
    return max(bankroll * f_adj, 0.0)

def confidence_unit_multiplier(
    value_tier: str,
    confidence: str,
    *,
    high_value_reduce: bool = True,
    high_value_low_mult: float = 0.40,
    high_value_med_mult: float = 0.66,
    high_value_high_mult: float = 1.00,
) -> float:
    """
    ✅ NEW:
    If it's HIGH VALUE but confidence is MEDIUM/LOW, reduce unit size instead of skipping.
    """
    if not high_value_reduce:
        return 1.0

    vt = (value_tier or "").upper()
    conf = (confidence or "").upper().strip()

    if "HIGH" not in vt:
        return 1.0

    if conf == "HIGH":
        return float(high_value_high_mult)
    if conf == "MEDIUM":
        return float(high_value_med_mult)
    return float(high_value_low_mult)

def compute_bet_size(
    row: pd.Series,
    bankroll: float,
    *,
    sizing_mode: str = "flat",  # "flat" or "kelly"
    flat_pct: float = UNIT_PCT,
    kelly_mult: float = 0.5,
    kelly_max_pct: float = 0.03,
    # ✅ NEW controls
    reduce_units_for_high_value_lower_conf: bool = True,
    high_value_low_mult: float = 0.40,
    high_value_med_mult: float = 0.66,
    high_value_high_mult: float = 1.00,
) -> float:
    if str(row.get("play_pass")) != "PLAY":
        return 0.0

    primary = str(row.get("primary_recommendation", ""))

    # base stake
    if sizing_mode == "flat":
        stake = bet_size_flat(bankroll, flat_pct)
    else:
        # kelly: implemented for ML only
        if "HOME ML" in primary:
            ml = safe_float(row.get("home_ml"))
            if ml is None:
                return 0.0
            p = float(row.get("model_home_prob"))
            stake = bet_size_kelly_ml(bankroll, p, ml, kelly_mult=kelly_mult, max_pct=kelly_max_pct)
        elif "AWAY ML" in primary:
            ml = safe_float(row.get("away_ml"))
            if ml is None:
                return 0.0
            p = 1.0 - float(row.get("model_home_prob"))
            stake = bet_size_kelly_ml(bankroll, p, ml, kelly_mult=kelly_mult, max_pct=kelly_max_pct)
        else:
            # ATS in kelly mode falls back to flat
            stake = bet_size_flat(bankroll, flat_pct)

    # ✅ apply reduced sizing for HIGH VALUE if confidence lower
    mult = confidence_unit_multiplier(
        row.get("value_tier", ""),
        row.get("confidence", ""),
        high_value_reduce=reduce_units_for_high_value_lower_conf,
        high_value_low_mult=high_value_low_mult,
        high_value_med_mult=high_value_med_mult,
        high_value_high_mult=high_value_high_mult,
    )
    return float(stake) * float(mult)


# -----------------------------
# Settlement (backtest)
# -----------------------------

def settle_ml(side: str, home_score: int, away_score: int, home_ml: float, away_ml: float, stake: float) -> float:
    if stake <= 0:
        return 0.0
    side = side.upper()
    if side == "HOME":
        won = home_score > away_score
        odds = home_ml
    else:
        won = away_score > home_score
        odds = away_ml

    if odds is None:
        return 0.0

    if won:
        dec = american_to_decimal(odds)
        return stake * (dec - 1.0)
    return -stake

def settle_ats(
    side: str,
    home_score: int,
    away_score: int,
    home_spread: float,
    stake: float,
    price_american: float = -110.0,
) -> float:
    if stake <= 0:
        return 0.0
    if home_spread is None:
        return 0.0

    side = side.upper()
    if side == "HOME":
        adj_home = home_score + float(home_spread)
        adj_away = away_score
    else:
        adj_home = home_score
        adj_away = away_score - float(home_spread)

    if abs(adj_home - adj_away) < 1e-9:
        return 0.0  # push

    won = adj_home > adj_away if side == "HOME" else adj_away > adj_home

    if won:
        dec = american_to_decimal(price_american)
        return stake * (dec - 1.0)
    return -stake

def resolve_game_scores_for_date(game_date_str: str, api_key: str) -> Dict[Tuple[str, str], Tuple[int, int]]:
    gdf = fetch_games_for_date(game_date_str, api_key=api_key)
    out = {}
    for _, r in gdf.iterrows():
        hs = r.get("HOME_SCORE")
        aw = r.get("AWAY_SCORE")
        if hs is None or aw is None:
            continue
        if (hs == 0 and aw == 0):
            continue
        out[(r["HOME_TEAM_NAME"], r["AWAY_TEAM_NAME"])] = (int(hs), int(aw))
    return out


# -----------------------------
# Main daily engine
# -----------------------------

def run_daily_probs_for_date(
    game_date: str,
    odds_dict=None,
    spreads_dict=None,
    stats_df=None,
    api_key=None,
    lam=0.25,
):
    if api_key is None:
        api_key = get_bdl_api_key()
    if stats_df is None:
        raise ValueError("stats_df must be precomputed.")

    odds_dict = odds_dict or {}
    spreads_dict = spreads_dict or {}

    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)

    games_df = fetch_games_for_date(game_date, api_key=api_key)
    if games_df.empty:
        return pd.DataFrame()

    # Load injuries once
    try:
        injury_df = fetch_injury_report_official_nba(game_date_obj)
    except Exception as e:
        print(f"[inj-fetch] WARNING: injury report failed: {e}")
        injury_df = pd.DataFrame(columns=["Team", "Player", "Status", "Reason"])

    rows = []
    for _, g in games_df.iterrows():
        home_name = g["HOME_TEAM_NAME"]
        away_name = g["AWAY_TEAM_NAME"]

        home_row = find_team_row(home_name, stats_df)
        away_row = find_team_row(away_name, stats_df)
        home_id = int(home_row["TEAM_ID"])
        away_id = int(away_row["TEAM_ID"])

        base_score = season_matchup_base_score(home_row, away_row)

        # injuries
        home_inj = build_injury_list_for_team_official(home_name, injury_df)
        away_inj = build_injury_list_for_team_official(away_name, injury_df)
        inj_adj = injury_adjustment(home_inj, away_inj)

        # fatigue
        home_last = get_team_last_game_date(home_id, game_date_obj, season_year, api_key)
        away_last = get_team_last_game_date(away_id, game_date_obj, season_year, api_key)
        home_rest_days = (game_date_obj - home_last).days if home_last else None
        away_rest_days = (game_date_obj - away_last).days if away_last else None
        fatigue_adj = rest_days_to_fatigue_adjustment(home_rest_days) - rest_days_to_fatigue_adjustment(away_rest_days)

        h2h_adj = compute_head_to_head_adjustment(home_id, away_id, season_year, api_key)

        adj_score = base_score + inj_adj + fatigue_adj + h2h_adj
        model_home_prob = score_to_prob(adj_score, lam)
        model_spread_home = score_to_spread(adj_score)

        key = (home_name, away_name)
        odds_info = odds_dict.get(key, {}) or {}
        home_ml = odds_info.get("home_ml")
        away_ml = odds_info.get("away_ml")
        home_spread = spreads_dict.get(key, odds_info.get("home_spread"))
        home_spread = safe_float(home_spread)

        rows.append({
            "date": game_date,
            "home": home_name,
            "away": away_name,
            "model_home_prob": float(model_home_prob),
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "model_spread_home": float(model_spread_home),
        })

    df = pd.DataFrame(rows)

    df, debug_df = add_recommendations_to_df(
        df,
        thresholds=Thresholds(
            ml_edge_strong=0.06,
            ml_edge_lean=0.035,
            ats_edge_strong_pts=3.0,
            ats_edge_lean_pts=1.5,
            conf_high=0.18,
            conf_med=0.10,
        ),
        model_spread_home_col="model_spread_home",
        model_margin_home_col=None,
    )

    os.makedirs("results", exist_ok=True)
    debug_out = f"results/debug_why_ml_vs_ats_{game_date.replace('/', '-')}.csv"
    debug_df.to_csv(debug_out, index=False)

    return df


# -----------------------------
# Backtest driver
# -----------------------------

def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def backtest_range(
    start_date_str: str,
    end_date_str: str,
    *,
    initial_bankroll: float,
    sizing_mode: str,
    flat_pct: float,
    kelly_mult: float,
    kelly_max_pct: float,
    ats_price: float,
    play_require_pick: bool,
    play_min_value_tier: str,
    play_min_confidence: str,
    play_max_abs_moneyline: Optional[int],
    api_key: str,
    reduce_units_for_high_value_lower_conf: bool,
):
    start = datetime.strptime(start_date_str, "%m/%d/%Y").date()
    end = datetime.strptime(end_date_str, "%m/%d/%Y").date()

    bankroll = float(initial_bankroll)
    equity = []
    bets_log = []

    for d in daterange(start, end):
        game_date = d.strftime("%m/%d/%Y")
        print(f"[backtest] {game_date} bankroll={bankroll:.2f}")

        try:
            odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date)
        except Exception as e:
            print(f"[backtest] odds load failed {game_date}: {e}")
            continue
        if not odds_dict:
            continue

        season_year = season_start_year_for_date(d)
        end_date_iso = d.strftime("%Y-%m-%d")
        try:
            stats_df = fetch_team_ratings_bdl(season_year=season_year, end_date_iso=end_date_iso, api_key=api_key)
        except Exception as e:
            print(f"[backtest] stats fetch failed {game_date}: {e}")
            continue

        try:
            df = run_daily_probs_for_date(
                game_date=game_date,
                odds_dict=odds_dict,
                spreads_dict=spreads_dict,
                stats_df=stats_df,
                api_key=api_key,
            )
        except Exception as e:
            print(f"[backtest] model run failed {game_date}: {e}")
            continue
        if df.empty:
            continue

        df["play_pass"] = df.apply(
            lambda r: play_pass_rule(
                r,
                require_pick=play_require_pick,
                min_value_tier=play_min_value_tier,
                min_confidence=play_min_confidence,
                max_abs_moneyline=play_max_abs_moneyline,
            ),
            axis=1
        )

        df["bet_size"] = df.apply(
            lambda r: compute_bet_size(
                r,
                bankroll,
                sizing_mode=sizing_mode,
                flat_pct=flat_pct,
                kelly_mult=kelly_mult,
                kelly_max_pct=kelly_max_pct,
                reduce_units_for_high_value_lower_conf=reduce_units_for_high_value_lower_conf,
            ),
            axis=1
        )

        unit_dollars = bankroll * UNIT_PCT
        df["unit_dollars"] = unit_dollars
        df["units"] = df["bet_size"].apply(lambda x: 0.0 if not x else float(x) / unit_dollars)

        scores_map = resolve_game_scores_for_date(game_date, api_key=api_key)
        if not scores_map:
            continue

        day_profit = 0.0
        day_bets = 0

        for _, r in df.iterrows():
            if r["play_pass"] != "PLAY":
                continue
            stake = float(r["bet_size"] or 0.0)
            if stake <= 0:
                continue

            key = (r["home"], r["away"])
            if key not in scores_map:
                continue

            home_score, away_score = scores_map[key]
            primary = str(r["primary_recommendation"])

            profit = 0.0
            bet_type = None
            bet_side = None

            if "HOME ML" in primary:
                bet_type = "ML"
                bet_side = "HOME"
                profit = settle_ml("HOME", home_score, away_score, r["home_ml"], r["away_ml"], stake)
            elif "AWAY ML" in primary:
                bet_type = "ML"
                bet_side = "AWAY"
                profit = settle_ml("AWAY", home_score, away_score, r["home_ml"], r["away_ml"], stake)
            elif "ATS" in primary and "HOME" in primary:
                bet_type = "ATS"
                bet_side = "HOME"
                profit = settle_ats("HOME", home_score, away_score, r["home_spread"], stake, price_american=ats_price)
            elif "ATS" in primary and "AWAY" in primary:
                bet_type = "ATS"
                bet_side = "AWAY"
                profit = settle_ats("AWAY", home_score, away_score, r["home_spread"], stake, price_american=ats_price)
            else:
                continue

            bankroll += profit
            day_profit += profit
            day_bets += 1

            bets_log.append({
                "date": game_date,
                "home": r["home"],
                "away": r["away"],
                "primary_recommendation": primary,
                "bet_type": bet_type,
                "bet_side": bet_side,
                "stake": stake,
                "units": float(r.get("units", 0.0)),
                "unit_dollars": float(r.get("unit_dollars", 0.0)),
                "profit": profit,
                "bankroll_after": bankroll,
                "why_bet": r.get("why_bet", ""),
                "confidence": r.get("confidence", ""),
                "value_tier": r.get("value_tier", ""),
            })

        equity.append({
            "date": game_date,
            "bankroll": bankroll,
            "day_profit": day_profit,
            "num_bets": day_bets
        })

    equity_df = pd.DataFrame(equity)
    bets_df = pd.DataFrame(bets_log)

    os.makedirs("results", exist_ok=True)
    eq_out = f"results/backtest_equity_{start_date_str.replace('/','-')}_to_{end_date_str.replace('/','-')}.csv"
    bets_out = f"results/backtest_bets_{start_date_str.replace('/','-')}_to_{end_date_str.replace('/','-')}.csv"
    equity_df.to_csv(eq_out, index=False)
    bets_df.to_csv(bets_out, index=False)
    print(f"[backtest] saved equity: {eq_out}")
    print(f"[backtest] saved bets:   {bets_out}")

    return equity_df, bets_df


# -----------------------------
# CLI / entrypoint
# -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run daily NBA betting model (BallDontLie).")

    parser.add_argument("--date", type=str, default=None, help="Game date in MM/DD/YYYY (default: today UTC).")
    parser.add_argument("--backtest_start", type=str, default=None, help="Backtest start date MM/DD/YYYY")
    parser.add_argument("--backtest_end", type=str, default=None, help="Backtest end date MM/DD/YYYY")

    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, help="Starting bankroll (default 250)")
    parser.add_argument("--sizing", type=str, default="flat", choices=["flat", "kelly"], help="Sizing mode")
    parser.add_argument("--flat_pct", type=float, default=UNIT_PCT, help="Flat stake percent (default 0.04 = 4%)")
    parser.add_argument("--kelly_mult", type=float, default=0.5, help="Kelly multiplier (0.5 = half Kelly)")
    parser.add_argument("--kelly_max_pct", type=float, default=0.03, help="Max Kelly stake pct (cap)")

    parser.add_argument("--play_require_pick", action="store_true", help="Require 'PICK' in primary to PLAY")
    parser.add_argument("--play_min_value_tier", type=str, default="HIGH VALUE", help="Min value tier: NO VALUE / MEDIUM VALUE / HIGH VALUE")
    parser.add_argument("--play_min_conf", type=str, default="MEDIUM", choices=["LOW", "MEDIUM", "HIGH"], help="Min confidence")
    parser.add_argument("--play_max_abs_ml", type=int, default=400, help="Pass if selected ML |odds| > this (set 0 to disable)")

    parser.add_argument("--reduce_units_for_high_value_lower_conf", action="store_true",
                        help="If set: HIGH VALUE with MEDIUM/LOW confidence gets reduced bet size instead of full unit")

    parser.add_argument("--ats_price", type=float, default=-110.0, help="ATS price for backtest settlement (default -110)")

    args = parser.parse_args(argv)
    api_key = get_bdl_api_key()

    # Backtest mode
    if args.backtest_start and args.backtest_end:
        play_max_abs_ml = None if args.play_max_abs_ml == 0 else args.play_max_abs_ml
        backtest_range(
            args.backtest_start,
            args.backtest_end,
            initial_bankroll=args.bankroll,
            sizing_mode=args.sizing,
            flat_pct=args.flat_pct,
            kelly_mult=args.kelly_mult,
            kelly_max_pct=args.kelly_max_pct,
            ats_price=args.ats_price,
            play_require_pick=args.play_require_pick,
            play_min_value_tier=args.play_min_value_tier,
            play_min_confidence=args.play_min_conf,
            play_max_abs_moneyline=play_max_abs_ml,
            api_key=api_key,
            reduce_units_for_high_value_lower_conf=args.reduce_units_for_high_value_lower_conf,
        )
        return

    # Single-day mode
    if args.date is None:
        today = datetime.utcnow().date()
        game_date = today.strftime("%m/%d/%Y")
    else:
        game_date = args.date

    print(f"Running model for {game_date}...")

    build_odds_csv_template_if_missing(game_date, api_key=api_key)

    try:
        odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date)
    except Exception as e:
        print(f"Warning: failed to load odds: {e}")
        odds_dict, spreads_dict = {}, {}

    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)
    end_date_iso = game_date_obj.strftime("%Y-%m-%d")

    stats_df = fetch_team_ratings_bdl(season_year=season_year, end_date_iso=end_date_iso, api_key=api_key)

    results_df = run_daily_probs_for_date(
        game_date=game_date,
        odds_dict=odds_dict,
        spreads_dict=spreads_dict,
        stats_df=stats_df,
        api_key=api_key,
    )

    play_max_abs_ml = None if args.play_max_abs_ml == 0 else args.play_max_abs_ml

    results_df["play_pass"] = results_df.apply(
        lambda r: play_pass_rule(
            r,
            require_pick=args.play_require_pick,
            min_value_tier=args.play_min_value_tier,
            min_confidence=args.play_min_conf,
            max_abs_moneyline=play_max_abs_ml,
        ),
        axis=1
    )

    results_df["bet_size"] = results_df.apply(
        lambda r: compute_bet_size(
            r,
            args.bankroll,
            sizing_mode=args.sizing,
            flat_pct=args.flat_pct,
            kelly_mult=args.kelly_mult,
            kelly_max_pct=args.kelly_max_pct,
            reduce_units_for_high_value_lower_conf=args.reduce_units_for_high_value_lower_conf,
        ),
        axis=1
    )

    unit_dollars = float(args.bankroll) * UNIT_PCT
    results_df["unit_dollars"] = unit_dollars
    results_df["units"] = results_df["bet_size"].apply(lambda x: 0.0 if not x else float(x) / unit_dollars)

    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")
    print(f"Bankroll=${float(args.bankroll):.2f} | 1 unit={UNIT_PCT*100:.1f}% = ${unit_dollars:.2f}")


if __name__ == "__main__":
    main()
