# current_of_sports_betting_algorithm.py
#
# Daily NBA betting model using BallDontLie + Official NBA injury report PDF + local odds CSV.
#
# Features:
# - Team ratings from BallDontLie:
#     - ORtg/DRtg proxies
#     - Pace
#     - Off/Def efficiency
# - Recent form weighting (season + last N games)
# - Official NBA injury report (PDF) with a simple player impact model
# - Schedule fatigue (rest days → B2B / 3-in-4 approximation)
# - Local odds CSV for moneyline + spreads
# - Outputs: model_home_prob, edges, model_spread, ML + spread recommendations

import os
import re
import math
import argparse
import time
from datetime import datetime, date
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# -----------------------------
# Global tuning constants
# -----------------------------

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

SPREAD_SCALE_FACTOR = 1.35

RECENT_FORM_WEIGHT = 0.35
SEASON_FORM_WEIGHT = 1.0 - RECENT_FORM_WEIGHT
RECENT_GAMES_WINDOW = 10

EDGE_PROB_THRESHOLD = 0.08
STRONG_EDGE_THRESHOLD = 0.12
SPREAD_EDGE_THRESHOLD = 2.5
MIN_MODEL_CONFIDENCE = 0.05

# -----------------------------
# Helpers
# -----------------------------

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ").strip())

def normalize_team_name(s: str) -> str:
    s = normalize_spaces(s).lower()
    s = s.replace(".", "")
    return s

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
        print(f"[odds_csv] No odds file found at {fname}. Using 0.5 market defaults.")
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

        raw_home_ml = row.get("home_ml")
        raw_away_ml = row.get("away_ml")
        raw_home_spread = row.get("home_spread") if "home_spread" in df.columns else None

        print(
            "[DEBUG row]",
            key,
            "| raw home_ml=",
            raw_home_ml,
            "| raw away_ml=",
            raw_away_ml,
            "| raw home_spread=",
            raw_home_spread,
        )

        home_ml = _parse_number(raw_home_ml)
        away_ml = _parse_number(raw_away_ml)
        home_spread = _parse_number(raw_home_spread)

        odds_dict[key] = {"home_ml": home_ml, "away_ml": away_ml, "home_spread": home_spread}
        if home_spread is not None:
            spreads_dict[key] = home_spread

    print(f"[odds_csv] Built odds for {len(odds_dict)} games.")
    print("[odds_csv] Sample keys:", list(odds_dict.keys())[:5])
    return odds_dict, spreads_dict


# -----------------------------
# Season / date helpers
# -----------------------------

def season_start_year_for_date(d: date) -> int:
    # NBA season starts ~Oct; before Aug belongs to prior season year label for BDL
    return d.year - 1 if d.month < 8 else d.year

def american_to_implied_prob(odds):
    odds = float(odds)
    if odds < 0:
        p = (-odds) / ((-odds) + 100.0)
    else:
        p = 100.0 / (odds + 100.0)
    return max(min(p, 0.9999), 0.0001)


# -----------------------------
# BallDontLie low-level client
# -----------------------------

BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"

def get_bdl_api_key():
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "BALLDONTLIE_API_KEY environment variable is not set."
        )
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
                if home_id in agg: agg[home_id]["wins"] += 1
                if away_id in agg: agg[away_id]["losses"] += 1
            elif away_score > home_score:
                if away_id in agg: agg[away_id]["wins"] += 1
                if home_id in agg: agg[home_id]["losses"] += 1

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
    return -score * points_per_logit


# -----------------------------
# Injuries (Official NBA PDF)
# -----------------------------

INJURY_WEIGHTS = {"star": 3.0, "starter": 1.5, "rotation": 1.0, "bench": 0.5}

INJURY_STATUS_MULTIPLIER = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.5,
    "probable": 0.25,
}

NBA_TEAM_NAMES = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
    "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers",
    "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
    "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
]
NBA_TEAM_NAMES_NORM = {normalize_team_name(t): t for t in NBA_TEAM_NAMES}

def guess_role(player_name, pos):
    pos = (pos or "").upper()
    if pos in ["PG", "SG", "SF", "PF"]:
        return "starter"
    if pos in ["C", "F", "G"]:
        return "rotation"
    return "rotation"

def status_to_mult(status):
    if not isinstance(status, str):
        return 1.0
    s = status.lower()
    for key, mult in INJURY_STATUS_MULTIPLIER.items():
        if key in s:
            return mult
    return 1.0

def estimate_player_impact_simple(pos):
    pos = (pos or "").upper()
    if pos in ["PG", "SG", "SF", "PF", "C"]:
        return 2.0
    return 1.0

def nba_injury_report_page_url(season_label: str):
    return f"https://official.nba.com/nba-injury-report-{season_label}-season/"

def get_latest_nba_injury_pdf_url_for_date(game_date_obj: date, season_label: str):
    url = nba_injury_report_page_url(season_label)
    print(f"[inj-fetch] NBA injury report page: {url}")

    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    month_name = game_date_obj.strftime("%B")          # December
    day_num = str(int(game_date_obj.strftime("%d")))   # 12
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

def parse_nba_injury_pdf_to_df(pdf_url: str) -> pd.DataFrame:
    """
    Parse the NBA official injury report PDF into a DataFrame.

    The NBA PDF layout frequently behaves like a table where Team is a column
    (not a section header). The original parser only worked when Team appeared
    as a standalone header line, which can yield 0 rows.

    This version:
      1) downloads with stronger headers and validates the bytes are a PDF
      2) extracts text with pypdf
      3) tries two parsing strategies:
         A) team-header mode (legacy)
         B) team-in-line mode (table-like PDFs)
      4) if still empty, writes a small debug text sample to results/injury_debug.txt
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://official.nba.com/",
    }
    r = requests.get(pdf_url, timeout=30, headers=headers, allow_redirects=True)
    r.raise_for_status()

    pdf_bytes = r.content or b""
    # Validate we actually received a PDF
    if not pdf_bytes.startswith(b"%PDF"):
        snippet = pdf_bytes[:200].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"[inj-parse] Downloaded content is not a PDF. "
            f"Content-Type={r.headers.get('Content-Type')} Snippet={snippet}"
        )

    reader = PdfReader(BytesIO(pdf_bytes))

    text_chunks = []
    for page in reader.pages:
        try:
            txt = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            txt = page.extract_text() or ""
        except Exception:
            txt = page.extract_text() or ""
        text_chunks.append(txt)

    if not any((c or "").strip() for c in text_chunks):
        print("[inj-parse] WARNING: PDF text extraction returned empty text.")
        return pd.DataFrame(columns=["Team", "Player", "Pos", "Status", "Injury"])

    raw_lines = []
    for chunk in text_chunks:
        for ln in (chunk or "").splitlines():
            ln = ln.replace("\xa0", " ").strip()
            if ln:
                raw_lines.append(ln)

    # Common helpers
    player_line_re = re.compile(
        r"(?P<Player>[A-Za-z\.\'\- ]+?)\s+"
        r"(?P<Pos>PG|SG|SF|PF|C|G|F)\s+"
        r"(?P<Status>Out|Doubtful|Questionable|Probable)\s*"
        r"(?P<Injury>.*)$"
    )

    # -------- Strategy B: table-like (Team + Player + Pos + Status on same line)
    # Build a team alternation regex that matches any official team name.
    team_alt = "|".join(sorted([re.escape(t) for t in NBA_TEAM_NAMES], key=len, reverse=True))
    team_in_line_re = re.compile(
        rf"^(?P<Team>(?:{team_alt}))\s+"
        rf"(?P<Player>[A-Za-z\.\'\- ]+?)\s+"
        rf"(?P<Pos>PG|SG|SF|PF|C|G|F)\s+"
        rf"(?P<Status>Out|Doubtful|Questionable|Probable)\s*"
        rf"(?P<Injury>.*)$"
    )

    rows = []
    last_row_idx = None

    for ln in raw_lines:
        ln_clean = normalize_spaces(ln)

        # Skip obvious non-data lines
        ln_lower = ln_clean.lower()
        if any(k in ln_lower for k in ["injury report", "referee", "page", "report time", "game date", "report updated"]):
            continue

        m2 = team_in_line_re.search(ln_clean)
        if m2:
            rows.append({
                "Team": normalize_spaces(m2.group("Team")),
                "Player": normalize_spaces(m2.group("Player")),
                "Pos": m2.group("Pos").strip(),
                "Status": m2.group("Status").strip(),
                "Injury": normalize_spaces(m2.group("Injury")),
            })
            last_row_idx = len(rows) - 1
            continue

        # Continuation lines for injury text (table-like)
        if last_row_idx is not None:
            # If it doesn't look like a new player row, append
            if not team_in_line_re.search(ln_clean) and not player_line_re.search(ln_clean):
                extra = normalize_spaces(ln_clean)
                if extra:
                    prev = rows[last_row_idx]["Injury"]
                    rows[last_row_idx]["Injury"] = normalize_spaces((prev + " " + extra).strip())

    if rows:
        return pd.DataFrame(rows, columns=["Team", "Player", "Pos", "Status", "Injury"])

    # -------- Strategy A: legacy team-header mode (Team on its own line)
    rows = []
    current_team = None
    last_row_idx = None

    for ln in raw_lines:
        ln_clean = normalize_spaces(ln)
        ln_norm = normalize_team_name(ln_clean)

        # If line matches a team name exactly, switch current team
        if ln_norm in NBA_TEAM_NAMES_NORM:
            current_team = NBA_TEAM_NAMES_NORM[ln_norm]
            last_row_idx = None
            continue

        if current_team is None:
            continue

        ln_lower = ln_clean.lower()
        if any(k in ln_lower for k in ["injury report", "referee", "page", "report time", "game date", "report updated"]):
            continue

        m = player_line_re.search(ln_clean)
        if m:
            rows.append({
                "Team": current_team,
                "Player": normalize_spaces(m.group("Player")),
                "Pos": m.group("Pos").strip(),
                "Status": m.group("Status").strip(),
                "Injury": normalize_spaces(m.group("Injury")),
            })
            last_row_idx = len(rows) - 1
            continue

        # Continuation lines
        if last_row_idx is not None:
            if not player_line_re.search(ln_clean):
                extra = normalize_spaces(ln_clean)
                if extra:
                    prev = rows[last_row_idx]["Injury"]
                    rows[last_row_idx]["Injury"] = normalize_spaces((prev + " " + extra).strip())

    df = pd.DataFrame(rows, columns=["Team", "Player", "Pos", "Status", "Injury"])

    # If still empty, write a debug sample so you can tune the regex quickly.
    if df.empty:
        try:
            os.makedirs("results", exist_ok=True)
            sample_path = os.path.join("results", "injury_debug.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(f"PDF URL: {pdf_url}\n\n")
                f.write("---- First 200 lines extracted ----\n")
                for ln in raw_lines[:200]:
                    f.write(ln + "\n")
            print(f"[inj-parse] Wrote debug extract sample to {sample_path}")
        except Exception as e:
            print(f"[inj-parse] Could not write debug sample: {e}")

    return df


def fetch_injury_report_official_nba(game_date_obj: date) -> pd.DataFrame:
    # If you want to compute season_label automatically, you can. Hardcoding is fine.
    season_label = "2025-26"
    pdf_url = get_latest_nba_injury_pdf_url_for_date(game_date_obj, season_label=season_label)
    print(f"[inj-fetch] Using NBA official injury PDF: {pdf_url}")
    df = parse_nba_injury_pdf_to_df(pdf_url)
    print(f"[inj-fetch] Parsed injury rows: {len(df)}")
    if not df.empty:
        print("[inj-fetch] Sample:\n", df.head(10))
    return df

def build_injury_list_for_team_official(team_name: str, injury_df: pd.DataFrame):
    if injury_df is None or injury_df.empty or "Team" not in injury_df.columns:
        return []

    team_norm = normalize_team_name(team_name)
    df = injury_df.copy()
    df["Team_norm"] = df["Team"].astype(str).apply(normalize_team_name)

    df_team = df[df["Team_norm"] == team_norm].copy()

    if df_team.empty:
        # fallback: match by nickname (last word)
        nick = team_norm.split()[-1]
        df_team = df[df["Team_norm"].str.contains(nick, na=False)].copy()

    if df_team.empty:
        return []

    injuries = []
    for _, row in df_team.iterrows():
        name = row.get("Player", "")
        pos = row.get("Pos", "")
        status = row.get("Status", "")
        role = guess_role(name, pos)
        mult = status_to_mult(status)
        impact_points = estimate_player_impact_simple(pos)
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
    print(f"[run_daily] Found {len(games_df)} games for {game_date}.")
    if games_df.empty:
        return pd.DataFrame()

    # Load injuries ONCE
    try:
        injury_df = fetch_injury_report_official_nba(game_date_obj)
    except Exception as e:
        print(f"[inj-fetch] WARNING: Could not load official NBA injury report: {e}")
        injury_df = pd.DataFrame(columns=["Team", "Player", "Pos", "Status", "Injury"])

    print(f"[run_daily] injury_df rows = {len(injury_df)}")

    rows = []
    for _, g in games_df.iterrows():
        home_name = g["HOME_TEAM_NAME"]
        away_name = g["AWAY_TEAM_NAME"]

        home_row = find_team_row(home_name, stats_df)
        away_row = find_team_row(away_name, stats_df)
        home_id = int(home_row["TEAM_ID"])
        away_id = int(away_row["TEAM_ID"])

        base_score = season_matchup_base_score(home_row, away_row)

        # Injuries from OFFICIAL DF
        home_inj = build_injury_list_for_team_official(home_name, injury_df)
        away_inj = build_injury_list_for_team_official(away_name, injury_df)
        inj_adj = injury_adjustment(home_inj, away_inj)

        print(f"[inj] {home_name} injuries={len(home_inj)}, {away_name} injuries={len(away_inj)}, inj_adj={inj_adj:.3f}")

        # Fatigue
        home_last = get_team_last_game_date(home_id, game_date_obj, season_year, api_key)
        away_last = get_team_last_game_date(away_id, game_date_obj, season_year, api_key)
        home_rest_days = (game_date_obj - home_last).days if home_last else None
        away_rest_days = (game_date_obj - away_last).days if away_last else None
        fatigue_adj = rest_days_to_fatigue_adjustment(home_rest_days) - rest_days_to_fatigue_adjustment(away_rest_days)

        h2h_adj = compute_head_to_head_adjustment(home_id, away_id, season_year, api_key)

        adj_score = base_score + inj_adj + fatigue_adj + h2h_adj
        model_home_prob = score_to_prob(adj_score, lam)
        model_spread = score_to_spread(adj_score)

        key = (home_name, away_name)
        odds_info = odds_dict.get(key, {}) or {}
        home_ml = odds_info.get("home_ml")
        away_ml = odds_info.get("away_ml")

        if home_ml is not None and away_ml is not None:
            raw_home_prob = american_to_implied_prob(home_ml)
            raw_away_prob = american_to_implied_prob(away_ml)
            total = raw_home_prob + raw_away_prob
            home_imp = raw_home_prob / total if total > 0 else 0.5
            away_imp = raw_away_prob / total if total > 0 else 0.5
        elif home_ml is not None:
            home_imp = american_to_implied_prob(home_ml)
            away_imp = 1.0 - home_imp
        elif away_ml is not None:
            away_imp = american_to_implied_prob(away_ml)
            home_imp = 1.0 - away_imp
        else:
            home_imp = away_imp = 0.5

        edge_home_raw = model_home_prob - home_imp
        edge_away_raw = (1.0 - model_home_prob) - away_imp
        edge_shrink = 0.5
        edge_home = edge_home_raw * edge_shrink
        edge_away = edge_away_raw * edge_shrink

        home_spread = spreads_dict.get(key, odds_info.get("home_spread"))
        if home_spread is not None:
            home_spread = float(home_spread)
            spread_edge_home = home_spread - model_spread
        else:
            spread_edge_home = None

        value_edge = abs(edge_home)
        model_conf = abs(model_home_prob - 0.5)

        if (value_edge < EDGE_PROB_THRESHOLD) or (model_conf < MIN_MODEL_CONFIDENCE):
            ml_rec = "No ML bet (edge/conf too small)"
        else:
            if model_home_prob > 0.5:
                ml_rec = "Model PICK: HOME (strong)" if value_edge >= STRONG_EDGE_THRESHOLD else "Model lean: HOME"
            else:
                ml_rec = "Model PICK: AWAY (strong)" if value_edge >= STRONG_EDGE_THRESHOLD else "Model lean: AWAY"

        if (home_spread is None) or (spread_edge_home is None):
            spread_rec = "No spread bet (no line)"
        else:
            if abs(spread_edge_home) < SPREAD_EDGE_THRESHOLD:
                spread_rec = "Too close to call ATS (edge too small)"
            else:
                spread_rec = "Model lean ATS: AWAY" if spread_edge_home > 0 else "Model lean ATS: HOME"

        if ("No ML bet" in ml_rec) and ("Too close" in spread_rec or "No spread" in spread_rec):
            primary_rec = "NO BET – edges too small"
        elif "No spread bet" in spread_rec or "Too close" in spread_rec:
            primary_rec = ml_rec
        elif "No ML bet" in ml_rec:
            primary_rec = spread_rec
        else:
            primary_rec = spread_rec

        rows.append({
            "date": game_date,
            "home": home_name,
            "away": away_name,
            "model_home_prob": model_home_prob,
            "market_home_prob": home_imp,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "model_spread_home": model_spread,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_edge_home": spread_edge_home,
            "ml_recommendation": ml_rec,
            "spread_recommendation": spread_rec,
            "primary_recommendation": primary_rec,
        })

    df = pd.DataFrame(rows)
    df["abs_edge_home"] = (df["model_home_prob"] - 0.5).abs()

    def classify_conf(conf):
        if conf >= 0.20:
            return "HIGH CONFIDENCE"
        elif conf >= 0.10:
            return "MEDIUM CONFIDENCE"
        else:
            return "LOW CONFIDENCE"

    df["value_tier"] = df["abs_edge_home"].apply(classify_conf)
    df = df.sort_values("abs_edge_home", ascending=False).reset_index(drop=True)
    return df


# -----------------------------
# CLI / entrypoint
# -----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run daily NBA betting model (BallDontLie).")
    parser.add_argument("--date", type=str, default=None, help="Game date in MM/DD/YYYY (default: today UTC).")
    args = parser.parse_args(argv)

    if args.date is None:
        today = datetime.utcnow().date()
        game_date = today.strftime("%m/%d/%Y")
    else:
        game_date = args.date

    print(f"Running model for {game_date}...")

    api_key = get_bdl_api_key()

    # Ensure odds template exists
    build_odds_csv_template_if_missing(game_date, api_key=api_key)

    # Load odds
    try:
        odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date)
        print(f"Loaded odds for {len(odds_dict)} games from CSV.")
    except Exception as e:
        print(f"Warning: failed to load odds from CSV: {e}")
        odds_dict, spreads_dict = {}, {}
        print("Proceeding with market_home_prob = 0.5 defaults.")

    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)
    end_date_iso = game_date_obj.strftime("%Y-%m-%d")

    try:
        stats_df = fetch_team_ratings_bdl(season_year=season_year, end_date_iso=end_date_iso, api_key=api_key)
    except Exception as e:
        print(f"Error: Failed to fetch team ratings from BallDontLie: {e}")
        return

    try:
        results_df = run_daily_probs_for_date(
            game_date=game_date,
            odds_dict=odds_dict,
            spreads_dict=spreads_dict,
            stats_df=stats_df,
            api_key=api_key,
        )
    except Exception as e:
        print(f"Error: Failed to run daily model: {e}")
        return

    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")


if __name__ == "__main__":
    main()
