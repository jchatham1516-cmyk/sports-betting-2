# Daily NBA betting model using BallDontLie + ESPN injuries + local odds CSV.
#
# Features:
# - Team ratings from BallDontLie:
#     - ORtg/DRtg proxies
#     - Pace
#     - Off/Def efficiency
# - ESPN per-team injuries with a simple player impact model
# - Schedule fatigue (rest days, approximating B2B / 3-in-4 / 4-in-6)
# - Head-to-head historical matchup adjustment (currently stubbed)
# - Local odds CSV for moneyline + spreads
# - Outputs: model_home_prob, edges, model_spread, ML + spread recommendations

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

import os
import sys
import math
import argparse
import time
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import requests

pip install -r requirements.txt
python current_of_sports_betting_algorithm.py --date 12/12/2025
# -----------------------------
# Global tuning constants
# -----------------------------

# Scale factor to convert model score -> spread (bigger = more extreme spreads)
SPREAD_SCALE_FACTOR = 1.35  # ~1.3–1.4 works well

# Recent form vs full-season weighting (for ORtg/DRtg, etc.)
RECENT_FORM_WEIGHT = 0.35
SEASON_FORM_WEIGHT = 1.0 - RECENT_FORM_WEIGHT
RECENT_GAMES_WINDOW = 10  # last N games for "recent form"

# Betting thresholds (to avoid betting everything)
EDGE_PROB_THRESHOLD = 0.08       # ~8% edge vs market
STRONG_EDGE_THRESHOLD = 0.12     # >=12% is "strong"
SPREAD_EDGE_THRESHOLD = 2.5      # need at least 2.5 pts edge to bet spread
MIN_MODEL_CONFIDENCE = 0.05      # |model_home_prob - 0.5|

# -----------------------------
# CSV odds loader
# -----------------------------


def fetch_odds_for_date_from_csv(game_date_str):
    """
    Load odds for a given date from a local CSV file in the 'odds' folder.

    Expects a file named:
        odds/odds_MM-DD-YYYY.csv

    with columns:
        date, home, away, home_ml, away_ml, home_spread
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
            f"Odds CSV {fname} must contain columns: {required_cols}. "
            f"Found: {list(df.columns)}"
        )

    def _parse_number(val):
        """Robustly parse odds/spread values from CSV."""
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

        # Debug: show what raw values we read from CSV
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

        odds_dict[key] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
        }
        if home_spread is not None:
            spreads_dict[key] = home_spread

    print(f"[odds_csv] Built odds for {len(odds_dict)} games.")
    print("[odds_csv] Sample keys:", list(odds_dict.keys())[:5])
    return odds_dict, spreads_dict


# -----------------------------
# Season / date helpers
# -----------------------------


def season_start_year_for_date(d: date) -> int:
    """
    BallDontLie uses season years like 2024 for the 2024-25 season.
    NBA season usually starts ~Oct; before August => belongs to prior season.
    """
    if d.month < 8:
        return d.year - 1
    return d.year


def american_to_implied_prob(odds):
    """
    Convert American odds (e.g. -150, +130) into implied *raw* probability in (0,1),
    WITHOUT removing the bookmaker's vig.
    """
    odds = float(odds)
    if odds < 0:
        p = (-odds) / ((-odds) + 100.0)
    else:
        p = 100.0 / (odds + 100.0)
    # Clamp for numerical safety
    return max(min(p, 0.9999), 0.0001)


# -----------------------------
# BallDontLie low-level client
# -----------------------------

BALLDONTLIE_BASE_URL = "https://api.balldondlie.io/v1".replace("dond", "dont")  # avoid typo


def get_bdl_api_key():
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "BALLDONTLIE_API_KEY environment variable is not set. "
            "Set it locally or configure it as a GitHub Actions secret."
        )
    return api_key


def bdl_get(path, params=None, api_key=None, max_retries=5):
    """
    Generic GET helper for BallDontLie with basic retry + rate-limit handling.
    """
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
                if retry_after is not None:
                    wait = int(retry_after)
                else:
                    wait = 15
                print(
                    f"[bdl_get] Rate limited (429) on {path}, "
                    f"attempt {attempt}/{max_retries}. Sleeping {wait}s..."
                )
                time.sleep(wait)
                last_exc = RuntimeError("Rate limited by BallDontLie")
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout as e:
            last_exc = e
            print(
                f"[bdl_get] Timeout calling {path} (attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(
                f"[bdl_get] HTTP error calling {path} (attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                time.sleep(5)

    raise RuntimeError(
        f"Failed to GET {path} from BallDontLie after {max_retries} attempts"
    ) from last_exc


# -----------------------------
# Team ratings built from games
# -----------------------------


def fetch_team_ratings_bdl(
    season_year: int,
    end_date_iso: str,
    api_key: str,
):
    """
    Build a team rating DataFrame from BallDontLie games:

    - ORtg proxy  = average points scored per game
    - DRtg proxy  = average points allowed per game
    - Pace        = (pts_for + pts_against) / GP
    - Off_Eff/Def_Eff ~ points per "possession" style proxy
    - W, L, W_PCT

    NEW: also computes "recent form" (last RECENT_GAMES_WINDOW games)
         and blends it in via global RECENT_FORM_WEIGHT.
    """
    # 1) Get teams
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

    # 2) Get all games for this season up to end_date_iso
    params = {
        "seasons[]": season_year,
        "end_date": end_date_iso,
        "per_page": 100,
    }
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

            # skip not-started games
            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue

            # Parse game date for recent-form window
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
                    games_by_team[home_id].append(
                        {
                            "date": g_date,
                            "pts_for": home_score,
                            "pts_against": away_score,
                        }
                    )
            if away_id in agg:
                agg[away_id]["gp"] += 1
                agg[away_id]["pts_for"] += away_score
                agg[away_id]["pts_against"] += home_score
                if g_date:
                    games_by_team[away_id].append(
                        {
                            "date": g_date,
                            "pts_for": away_score,
                            "pts_against": home_score,
                        }
                    )

            # wins / losses
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

    # 3) Build DataFrame with pace + efficiency + recent-form versions
    rows = []
    for tid, rec in agg.items():
        gp = rec["gp"]
        wins = rec["wins"]
        losses = rec["losses"]
        w_pct = wins / gp if gp > 0 else 0.0
        or_p = rec["pts_for"] / gp if gp > 0 else 0.0
        dr_p = rec["pts_against"] / gp if gp > 0 else 0.0

        # Pace approx: total points / games
        total_pts = rec["pts_for"] + rec["pts_against"]
        pace = total_pts / gp if gp > 0 else 0.0

        # Simple efficiency proxies
        poss = max(total_pts, 1)
        off_eff = rec["pts_for"] / poss
        def_eff = rec["pts_against"] / poss

        # --- Recent form (last N games) ---
        recent_games = sorted(games_by_team[tid], key=lambda x: x["date"])[
            -RECENT_GAMES_WINDOW:
        ]
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
            # fall back to season numbers
            or_p_recent = or_p
            dr_p_recent = dr_p
            pace_recent = pace
            off_eff_recent = off_eff
            def_eff_recent = def_eff

        rows.append(
            {
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
                # recent-form variants
                "ORtg_RECENT": or_p_recent,
                "DRtg_RECENT": dr_p_recent,
                "PACE_RECENT": pace_recent,
                "OFF_EFF_RECENT": off_eff_recent,
                "DEF_EFF_RECENT": def_eff_recent,
            }
        )

    df = pd.DataFrame(rows)
    return df


# -----------------------------
# Team lookup + scoring model
# -----------------------------


def find_team_row(team_name_input, stats_df):
    """
    Fuzzy match a team name against TEAM_NAME in stats_df.
    """
    name = team_name_input.strip().lower()

    full_match = stats_df[stats_df["TEAM_NAME"].str.lower() == name]
    if not full_match.empty:
        return full_match.iloc[0]

    contains_match = stats_df[stats_df["TEAM_NAME"].str.lower().str.contains(name)]
    if not contains_match.empty:
        return contains_match.iloc[0]

    raise ValueError(f"Could not find a team matching: {team_name_input}")


# Global weights for the base matchup model.
MATCHUP_WEIGHTS = np.array(
    [
        0.2,   # home_edge baseline (smaller)
        0.12,  # d_ORtg
        0.12,  # d_DRtg
        0.03,  # d_pace
        4.0,   # d_off_eff
        4.0,   # d_def_eff
    ]
)


def _blend_stat(row, base_col, recent_col):
    """
    Blend season and recent values for a given stat, if recent is available.
    """
    base_val = float(row[base_col])
    if recent_col in row.index:
        recent_val = float(row[recent_col])
        return SEASON_FORM_WEIGHT * base_val + RECENT_FORM_WEIGHT * recent_val
    return base_val


def build_matchup_features(home_row, away_row):
    """
    Build a feature vector [home_edge, d_ORtg, d_DRtg, d_pace, d_off_eff, d_def_eff]
    for the home team.

    Uses blended season+recent stats for ORtg/DRtg/PACE/OFF_EFF/DEF_EFF.
    """
    h = home_row
    a = away_row

    # Blend season + recent form
    h_ORtg = _blend_stat(h, "ORtg", "ORtg_RECENT")
    a_ORtg = _blend_stat(a, "ORtg", "ORtg_RECENT")
    h_DRtg = _blend_stat(h, "DRtg", "DRtg_RECENT")
    a_DRtg = _blend_stat(a, "DRtg", "DRtg_RECENT")
    h_PACE = _blend_stat(h, "PACE", "PACE_RECENT")
    a_PACE = _blend_stat(a, "PACE", "PACE_RECENT")
    h_OFF = _blend_stat(h, "OFF_EFF", "OFF_EFF_RECENT")
    a_OFF = _blend_stat(a, "OFF_EFF", "OFF_EFF_RECENT")
    h_DEF = _blend_stat(h, "DEF_EFF", "DEF_EFF_RECENT")
    a_DEF = _blend_stat(a, "DEF_EFF", "DEF_EFF_RECENT")

    d_ORtg = h_ORtg - a_ORtg
    d_DRtg = a_DRtg - h_DRtg  # lower DRtg is better
    d_pace = h_PACE - a_PACE
    d_off_eff = h_OFF - a_OFF
    d_def_eff = a_DEF - h_DEF

    home_edge = 1.0  # constant/home-court bias term

    return np.array(
        [
            home_edge,
            d_ORtg,
            d_DRtg,
            d_pace,
            d_off_eff,
            d_def_eff,
        ],
        dtype=float,
    )


def season_matchup_base_score(home_row, away_row):
    """
    Base linear scoring model without injuries/fatigue/H2H.

    Positive score => home team stronger.
    """
    x = build_matchup_features(home_row, away_row)
    return float(np.dot(MATCHUP_WEIGHTS, x))


def score_to_prob(score, lam=0.25):
    """
    Convert matchup score into win probability via logistic function.
    """
    return 1.0 / (1.0 + math.exp(-lam * score))


def score_to_spread(score, points_per_logit=SPREAD_SCALE_FACTOR):
    """
    Convert model 'score' into a *Vegas-style* point spread for the HOME team.

    - Negative number  => home favorite (e.g., -5.5 means home -5.5)
    - Positive number  => home underdog (e.g., +3.5 means home +3.5)
    """
    return -score * points_per_logit


# -----------------------------
# Injuries (ESPN per-team)
# -----------------------------

INJURY_WEIGHTS = {
    "star": 3.0,
    "starter": 1.5,
    "rotation": 1.0,
    "bench": 0.5,
}

INJURY_STATUS_MULTIPLIER = {
    "out": 1.0,
    "doubtful": 0.75,
    "questionable": 0.5,
    "probable": 0.25,
}

# Map full team names (from BallDontLie) to ESPN injury-page abbreviations.
ESPN_TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GS",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NO",
    "New York Knicks": "NY",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTAH",
    "Washington Wizards": "WSH",
}


def guess_role(player_name, pos):
    """
    Very rough heuristic to map a player to a role category.
    """
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
    """
    Very simple position-based impact proxy.
    You can replace this later with per-player stats from BDL.
    """
    pos = (pos or "").upper()
    if pos in ["PG", "SG", "SF", "PF", "C"]:
        return 2.0
    return 1.0


def fetch_injury_report_espn_for_teams(team_names):
    """
    Fetch injuries from ESPN *per team* (team injury pages),
    and return a single DataFrame with columns:
        Player, Pos, Status, Injury, Team

    team_names: iterable of full team names from BallDontLie
                (e.g. "Golden State Warriors", "Chicago Bulls").
    """
    frames = []

    for full_name in sorted(set(team_names)):
        full_name_str = str(full_name).strip()
        abbr = ESPN_TEAM_ABBR.get(full_name_str)

        if not abbr:
            print(f"[inj-fetch] No ESPN_TEAM_ABBR mapping for '{full_name_str}', skipping.")
            continue

        slug = abbr.lower()  # "GS" -> "gs", "PHI" -> "phi"
        url = f"https://www.espn.com/nba/team/injuries/_/name/{slug}"
        print(f"[inj-fetch] Fetching injuries for {full_name_str} from {url}")

        try:
            tables = pd.read_html(url)
        except Exception as e:
            print(f"[inj-fetch] Failed to read HTML for {full_name_str} ({url}): {e}")
            continue

        if not tables:
            print(f"[inj-fetch] No tables found on {url} for {full_name_str}")
            continue

        df_team_raw = tables[0].copy()
        if df_team_raw.empty:
            print(f"[inj-fetch] Empty injury table for {full_name_str} at {url}")
            continue

        # Normalize column names
        rename_map = {}
        for c in df_team_raw.columns:
            lc = str(c).strip().lower()
            if "name" in lc or "player" in lc:
                rename_map[c] = "Player"
            elif "pos" in lc:
                rename_map[c] = "Pos"
            elif "status" in lc:
                rename_map[c] = "Status"
            elif "injury" in lc or "reason" in lc or "comment" in lc:
                rename_map[c] = "Injury"

        df_team = df_team_raw.rename(columns=rename_map)

        keep_cols = [c for c in ["Player", "Pos", "Status", "Injury"] if c in df_team.columns]
        df_team = df_team[keep_cols].copy()

        # Tag with the full team name so we can filter later
        df_team["Team"] = full_name_str

        print(f"[inj-fetch] {full_name_str}: {len(df_team)} injury rows scraped.")
        frames.append(df_team)

    if not frames:
        print("[inj-fetch] WARNING: No injuries found for any team. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Player", "Pos", "Status", "Injury", "Team"])

    injury_df = pd.concat(frames, ignore_index=True)
    print(f"[inj-fetch] Combined injury rows: {len(injury_df)}")
    print("[inj-fetch] Injury sample:\n", injury_df.head())
    return injury_df


def build_injury_list_for_team_espn(team_name_or_abbrev, injury_df):
    """
    Build a list of injuries for a given team from our per-team ESPN injuries DataFrame.
    Returns list of tuples: (player_name, role, multiplier, impact_points).
    """
    if injury_df is None or injury_df.empty:
        return []

    full_name = str(team_name_or_abbrev).strip()

    if "Team" not in injury_df.columns:
        print(f"[inj-match] No 'Team' column in injury_df when looking for {full_name}.")
        return []

    df_team = injury_df[injury_df["Team"].astype(str) == full_name].copy()

    if df_team.empty:
        print(f"[inj-match] No injuries found in DataFrame for '{full_name}'.")
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

    print(f"[inj-match] {full_name}: matched {len(injuries)} injuries.")
    return injuries


def injury_adjustment(home_injuries=None, away_injuries=None):
    """
    home_injuries / away_injuries: list of (name, role, mult, impact_points)
    Missing players on the HOME team LOWER the score.
    Missing players on the AWAY team RAISE the score (helps home).
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    def parse_inj_list(lst, sign):
        adj = 0.0
        for item in lst:
            # (name, role, mult, impact_points)
            if len(item) == 4:
                _, role, mult, impact = item
            elif len(item) == 3:
                _, role, mult = item
                impact = 2.0
            elif len(item) == 2:
                _, role = item
                mult = 1.0
                impact = 2.0
            else:
                role = "starter"
                mult = 1.0
                impact = 2.0

            weight = INJURY_WEIGHTS.get(role, INJURY_WEIGHTS["starter"])
            adj += sign * weight * mult * (impact / 2.0)
        return adj

    total_adj = 0.0
    total_adj += parse_inj_list(home_injuries, sign=-1.0)  # hurts home
    total_adj += parse_inj_list(away_injuries, sign=+1.0)  # hurts away
    return total_adj


# -----------------------------
# Schedule / games (BallDontLie)
# -----------------------------


def fetch_games_for_date(game_date_str, stats_df, api_key):
    """
    game_date_str format: 'MM/DD/YYYY'
    Returns a DataFrame with GAME_ID, HOME_TEAM_NAME, AWAY_TEAM_NAME using BallDontLie.
    """
    dt = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    iso_date = dt.strftime("%Y-%m-%d")

    params = {"dates[]": iso_date, "per_page": 100}
    games_json = bdl_get("games", params=params, api_key=api_key)
    games = games_json.get("data", [])

    rows = []
    for g in games:
        home_team = g["home_team"]
        away_team = g["visitor_team"]
        rows.append(
            {
                "GAME_ID": g["id"],
                "HOME_TEAM_NAME": home_team["full_name"],
                "AWAY_TEAM_NAME": away_team["full_name"],
                "HOME_TEAM_ID": home_team["id"],
                "AWAY_TEAM_ID": away_team["id"],
                "GAME_DATE": g.get("date"),
            }
        )

    return pd.DataFrame(rows)


def build_odds_csv_template_if_missing(game_date_str, api_key, odds_dir="odds"):
    """
    If odds/odds_MM-DD-YYYY.csv does not exist, create a template CSV with:
      date, home, away, home_ml, away_ml, home_spread
    """
    os.makedirs(odds_dir, exist_ok=True)
    odds_path = os.path.join(odds_dir, f"odds_{game_date_str.replace('/', '-')}.csv")

    if os.path.exists(odds_path):
        return odds_path

    print(f"[template] No odds file found for {game_date_str}, creating template at {odds_path}...")

    games_df = fetch_games_for_date(game_date_str, stats_df=None, api_key=api_key)

    if games_df.empty:
        print(f"[template] No games found on {game_date_str}; not creating odds template.")
        return odds_path

    rows = []
    for _, row in games_df.iterrows():
        rows.append(
            {
                "date": game_date_str,
                "home": row["HOME_TEAM_NAME"],
                "away": row["AWAY_TEAM_NAME"],
                "home_ml": "",
                "away_ml": "",
                "home_spread": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(odds_path, index=False)
    print(f"[template] Template odds file created: {odds_path}")
    print("[template] Fill in moneylines & spreads, then rerun the script for real edges.")
    return odds_path


# -----------------------------
# Schedule fatigue & H2H helpers
# -----------------------------


def compute_head_to_head_adjustment(home_team_id, away_team_id, season_year, api_key, max_seasons_back=3):
    """
    TEMP STUB: head-to-head adjustment is currently disabled.
    """
    return 0.0


def get_team_last_game_date(team_id, game_date_obj, season_year, api_key):
    """
    Find the last completed game for a team BEFORE game_date_obj.
    Returns a date or None.
    """
    iso_date = game_date_obj.strftime("%Y-%m-%d")
    params = {
        "seasons[]": season_year,
        "team_ids[]": team_id,
        "end_date": iso_date,
        "per_page": 100,
    }
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
            # parse game date
            g_date_str = g.get("date")
            if not g_date_str:
                continue
            g_date = datetime.fromisoformat(g_date_str.replace("Z", "+00:00")).date()
            if g_date >= game_date_obj:
                continue

            home_score = g.get("home_team_score", 0) or 0
            away_score = g.get("visitor_team_score", 0) or 0
            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue  # unfinished

            if (last_date is None) or (g_date > last_date):
                last_date = g_date

        if not cursor:
            break

    return last_date


def rest_days_to_fatigue_adjustment(days_rest):
    """
    Map rest days to a fatigue adjustment in points (for the team's score).
    Positive = more rested (helps team), negative = tired.
    """
    if days_rest is None:
        return 0.0
    if days_rest <= 1:
        # back-to-back or very short rest
        return -2.0
    if days_rest == 2:
        # likely 3-in-4 / 4-in-6 situations
        return -1.0
    if days_rest >= 4:
        # very rested
        return +0.5
    # 3 days rest → roughly neutral
    return 0.0


# -----------------------------
# Main daily engine
# -----------------------------


def run_daily_probs_for_date(
    game_date="12/04/2025",
    odds_dict=None,
    spreads_dict=None,
    stats_df=None,
    api_key=None,
    edge_threshold=0.03,  # kept for reference, no longer drives recs
    lam=0.25,
):
    """
    Run the full model for one NBA date.

    - Uses blended season + recent form for team strength.
    - Uses ESPN injuries + rest-day fatigue.
    - Scales spreads by SPREAD_SCALE_FACTOR.
    - Recommendations are filtered: no bet unless edge/confidence is large enough.
    """
    if api_key is None:
        api_key = get_bdl_api_key()

    if stats_df is None:
        raise ValueError("stats_df must be precomputed for BallDontLie version.")

    if odds_dict is None:
        odds_dict = {}

    if spreads_dict is None:
        spreads_dict = {}

    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)

    # Schedule from BallDontLie
    games_df = fetch_games_for_date(game_date, stats_df, api_key)
    print(f"[run_daily] Found {len(games_df)} games for {game_date}.")

    # Injuries: fetch ONLY for teams playing today
    team_names_today = set(games_df["HOME_TEAM_NAME"].tolist() + games_df["AWAY_TEAM_NAME"].tolist())
    print("[run_daily] Teams today for injuries:", team_names_today)

    try:
        injury_df = fetch_injury_report_espn_for_teams(team_names_today)
    except Exception as e:
        print(f"Warning: failed to fetch ESPN injuries: {e}")
        injury_df = pd.DataFrame(columns=["Player", "Pos", "Status", "Injury", "Team"])

    print(f"[run_daily] injury_df rows = {len(injury_df)}")

    rows = []

    for _, g in games_df.iterrows():
        home_name = g["HOME_TEAM_NAME"]
        away_name = g["AWAY_TEAM_NAME"]

        home_row = find_team_row(home_name, stats_df)
        away_row = find_team_row(away_name, stats_df)
        home_id = int(home_row["TEAM_ID"])
        away_id = int(away_row["TEAM_ID"])

        # Base matchup (season + recent form blended)
        base_score = season_matchup_base_score(home_row, away_row)

        # Injuries
        home_inj = build_injury_list_for_team_espn(home_name, injury_df)
        away_inj = build_injury_list_for_team_espn(away_name, injury_df)
        inj_adj = injury_adjustment(home_inj, away_inj)

        print(
            f"[inj] {home_name} injuries={len(home_inj)}, "
            f"{away_name} injuries={len(away_inj)}, inj_adj={inj_adj:.3f}"
        )

        # Schedule fatigue (rest days as B2B / 3-in-4 approximation)
        home_last = get_team_last_game_date(home_id, game_date_obj, season_year, api_key)
        away_last = get_team_last_game_date(away_id, game_date_obj, season_year, api_key)

        home_rest_days = (game_date_obj - home_last).days if home_last else None
        away_rest_days = (game_date_obj - away_last).days if away_last else None

        home_fatigue = rest_days_to_fatigue_adjustment(home_rest_days)
        away_fatigue = rest_days_to_fatigue_adjustment(away_rest_days)

        fatigue_adj = home_fatigue - away_fatigue  # positive helps home

        # Head-to-head historical adjustment (currently 0.0)
        h2h_adj = compute_head_to_head_adjustment(home_id, away_id, season_year, api_key)

        # Final score
        adj_score = base_score + inj_adj + fatigue_adj + h2h_adj

        # Model win prob & spread
        model_home_prob = score_to_prob(adj_score, lam)
        model_spread = score_to_spread(adj_score)  # VEGAS STYLE: negative = home favorite

        # -------------------------
        # Market odds (ML)
        # -------------------------
        key = (home_name, away_name)
        odds_info = odds_dict.get(key)
        if odds_info is None:
            print(f"[run_daily] No odds found for {home_name} vs {away_name}")
            odds_info = {}

        home_ml = odds_info.get("home_ml")
        away_ml = odds_info.get("away_ml")

        # Convert American odds -> fair win probabilities (vig removed)
        if home_ml is not None and away_ml is not None:
            raw_home_prob = american_to_implied_prob(home_ml)
            raw_away_prob = american_to_implied_prob(away_ml)
            total = raw_home_prob + raw_away_prob
            if total > 0:
                home_imp = raw_home_prob / total
                away_imp = raw_away_prob / total
            else:
                home_imp = away_imp = 0.5
        elif home_ml is not None:
            home_imp = american_to_implied_prob(home_ml)
            away_imp = 1.0 - home_imp
        elif away_ml is not None:
            away_imp = american_to_implied_prob(away_ml)
            home_imp = 1.0 - away_imp
        else:
            home_imp = away_imp = 0.5

        # Raw edges vs market
        edge_home_raw = model_home_prob - home_imp
        edge_away_raw = (1.0 - model_home_prob) - away_imp

        # Small shrink just to avoid gigantic edge numbers
        edge_shrink = 0.5
        edge_home = edge_home_raw * edge_shrink
        edge_away = edge_away_raw * edge_shrink

        # -------------------------
        # Spreads (Vegas style)
        # -------------------------
        home_spread = spreads_dict.get(key, odds_info.get("home_spread"))
        if home_spread is not None:
            home_spread = float(home_spread)
            spread_edge_home = home_spread - model_spread
        else:
            spread_edge_home = None

        # -------------------------
        # Recommendation logic with filters
        # -------------------------
        value_edge = abs(edge_home)                     # vs market
        model_confidence = abs(model_home_prob - 0.5)   # vs 0.5 coin flip

        # 1) Moneyline recommendation
        if (value_edge < EDGE_PROB_THRESHOLD) or (model_confidence < MIN_MODEL_CONFIDENCE):
            ml_rec = "No ML bet (edge/conf too small)"
        else:
            if model_home_prob > 0.5:
                ml_rec = (
                    "Model PICK: HOME (strong)"
                    if value_edge >= STRONG_EDGE_THRESHOLD
                    else "Model lean: HOME"
                )
            else:
                ml_rec = (
                    "Model PICK: AWAY (strong)"
                    if value_edge >= STRONG_EDGE_THRESHOLD
                    else "Model lean: AWAY"
                )

        # 2) Spread recommendation (requires a line + edge)
        if (home_spread is None) or (spread_edge_home is None):
            spread_rec = "No spread bet (no line)"
        else:
            if abs(spread_edge_home) < SPREAD_EDGE_THRESHOLD:
                spread_rec = "Too close to call ATS (edge too small)"
            else:
                if spread_edge_home > 0:
                    spread_rec = "Model lean ATS: AWAY"
                else:
                    spread_rec = "Model lean ATS: HOME"

        # 3) Primary recommendation
        if ("No ML bet" in ml_rec) and (
            "Too close" in spread_rec or "No spread" in spread_rec
        ):
            primary_rec = "NO BET – edges too small"
        elif "No spread bet" in spread_rec or "Too close" in spread_rec:
            primary_rec = ml_rec
        elif "No ML bet" in ml_rec:
            primary_rec = spread_rec
        else:
            primary_rec = spread_rec  # usually prefer spread

        rows.append(
            {
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
            }
        )

    df = pd.DataFrame(rows)

    # ------------------------------------
    # Confidence metric (model vs 0.5)
    # ------------------------------------
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
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Game date in MM/DD/YYYY (default: today in UTC).",
    )
    args = parser.parse_args(argv)

    if args.date is None:
        today = datetime.utcnow().date()
        game_date = today.strftime("%m/%d/%Y")
    else:
        game_date = args.date

    print(f"Running model for {game_date}...")

    api_key = get_bdl_api_key()

    # 1) Ensure odds template exists (optional)
    build_odds_csv_template_if_missing(game_date, api_key=api_key)

    # 2) Load odds from local CSV
    try:
        odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date)
        print(f"Loaded odds for {len(odds_dict)} games from CSV.")
    except Exception as e:
        print(f"Warning: failed to load odds from CSV: {e}")
        odds_dict = {}
        spreads_dict = {}
        print("Proceeding with market_home_prob = 0.5 defaults.")

    # 3) Determine season year
    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)
    end_date_iso = game_date_obj.strftime("%Y-%m-%d")

    # 4) Fetch team ratings
    try:
        stats_df = fetch_team_ratings_bdl(
            season_year=season_year,
            end_date_iso=end_date_iso,
            api_key=api_key,
        )
    except Exception as e:
        print(f"Error: Failed to fetch team ratings from BallDontLie: {e}")
        print("Exiting without predictions so the workflow can complete gracefully.")
        return

    # 5) Run daily model
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
        print("Exiting without predictions so the workflow can complete gracefully.")
        return

    # 6) Save + print
    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")


if __name__ == "__main__":
    main()
