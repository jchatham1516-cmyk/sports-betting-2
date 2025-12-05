# Daily NBA betting model using BallDontLie + ESPN injuries + local odds CSV.
#
# Features:
# - Team ratings from BallDontLie:
#     - ORtg/DRtg proxies
#     - Pace
#     - Off/Def efficiency
# - ESPN injuries with a simple player impact model
# - Schedule fatigue (rest days)
# - Head-to-head historical matchup adjustment
# - Local odds CSV for moneyline + spreads
# - Outputs: model_home_prob, edges, model_spread, ML + spread recommendations

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

import os
import sys
import math
import argparse
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests

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

    odds_dict = {}
    spreads_dict = {}

    for _, row in df.iterrows():
        home = str(row["home"]).strip()
        away = str(row["away"]).strip()
        key = (home, away)

        # Debug: show what raw values we read from CSV
        print(
            "[DEBUG row]", key,
            "| raw home_ml=", row.get("home_ml"),
            "| raw away_ml=", row.get("away_ml"),
            "| raw home_spread=", row.get("home_spread"),
        )

        home_ml = row["home_ml"]
        away_ml = row["away_ml"]
        home_spread = row["home_spread"] if "home_spread" in df.columns else None

        # Convert to float if not null
        home_ml = float(home_ml) if pd.notna(home_ml) else None
        away_ml = float(away_ml) if pd.notna(away_ml) else None
        if home_spread is not None and pd.notna(home_spread):
            home_spread = float(home_spread)
        else:
            home_spread = None

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
    Convert American odds (e.g. -150, +130) into implied probability in [0,1].
    """
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


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
    """
    # 1) Get teams
    teams_json = bdl_get("teams", params={}, api_key=api_key)
    teams_data = teams_json.get("data", [])

    agg = {}
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

            if home_id in agg:
                agg[home_id]["gp"] += 1
                agg[home_id]["pts_for"] += home_score
                agg[home_id]["pts_against"] += away_score
            if away_id in agg:
                agg[away_id]["gp"] += 1
                agg[away_id]["pts_for"] += away_score
                agg[away_id]["pts_against"] += home_score

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

    # 3) Build DataFrame with pace + efficiency
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
        # Using total points as possession proxy – this is rough but better than nothing.
        poss = max(total_pts, 1)
        off_eff = rec["pts_for"] / poss
        def_eff = rec["pts_against"] / poss

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


def season_matchup_base_score(home_row, away_row):
    """
    Base linear scoring model without injuries/fatigue/H2H.
    Positive score => home team stronger.
    Uses:
       - ORtg / DRtg
       - Pace
       - Off/Def efficiency
       - Home-court edge
    """

    h = home_row
    a = away_row

    # Differences
    d_ORtg = h["ORtg"] - a["ORtg"]
    d_DRtg = a["DRtg"] - h["DRtg"]   # lower DRtg is better
    d_pace = h["PACE"] - a["PACE"]
    d_off_eff = h["OFF_EFF"] - a["OFF_EFF"]
    d_def_eff = a["DEF_EFF"] - h["DEF_EFF"]

    # ✔ Calibrated weights (realistic)
    home_edge = 1.2   # reduced from 2.0

    score = (
        home_edge
        + 0.04 * d_ORtg
        + 0.04 * d_DRtg
        + 0.01 * d_pace
        + 2.0 * d_off_eff
        + 2.0 * d_def_eff
    )

    return score

def score_to_prob(score, lam=0.20):
    """
    Convert matchup score into win probability via logistic function.
    lam controls steepness.
    """
    return 1.0 / (1.0 + math.exp(-lam * score))


def score_to_spread(score, points_per_logit=1.3):
    """
    Convert model 'score' into predicted point spread for HOME team.
    Positive = home favorite by that many points.
    """
    return score * points_per_logit


# -----------------------------
# Injuries (ESPN scraping)
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


def fetch_injury_report_espn():
    """
    Scrape ESPN NBA injuries page into a DataFrame with columns:
    Player, Team, Pos, Status, Injury
    """
    url = "https://www.espn.com/nba/injuries"
    tables = pd.read_html(url)
    if not tables:
        raise ValueError("ESPN injuries page: no tables found.")

    injury_tables = []
    for t in tables:
        cols_norm = [str(c).strip().lower() for c in t.columns]
        if any("player" in c or "name" in c for c in cols_norm):
            injury_tables.append(t)

    if not injury_tables:
        raise ValueError("No matching injury tables found on ESPN injuries page.")

    df = pd.concat(injury_tables, ignore_index=True)

    rename_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if "player" in lc or "name" in lc:
            rename_map[c] = "Player"
        elif "team" in lc:
            rename_map[c] = "Team"
        elif "pos" in lc:
            rename_map[c] = "Pos"
        elif "status" in lc:
            rename_map[c] = "Status"
        elif "injury" in lc or "reason" in lc:
            rename_map[c] = "Injury"

    df = df.rename(columns=rename_map)
    keep = [c for c in ["Player", "Team", "Pos", "Status", "Injury"] if c in df.columns]
    df = df[keep].copy()

    return df


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
    # Starters / main positions get a bit more weight
    if pos in ["PG", "SG", "SF", "PF", "C"]:
        return 2.0
    return 1.0


def build_injury_list_for_team_espn(team_name_or_abbrev, injury_df):
    """
    Build a list of injuries for a given team from ESPN injuries DataFrame.
    Returns list of tuples: (player_name, role, multiplier, impact_points)
    """
    team = team_name_or_abbrev.lower()

    if "Team" in injury_df.columns:
        mask = injury_df["Team"].astype(str).str.lower().str.contains(team)
        df_team = injury_df[mask].copy()
    else:
        df_team = injury_df.iloc[0:0].copy()

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
        # back-to-back or 1 day rest
        return -2.0
    if days_rest == 2:
        # 3 in 4 type schedule
        return -1.0
    if days_rest >= 4:
        # very rested
        return +0.5
    # 3 days rest → neutral
    return 0.0


def compute_head_to_head_adjustment(home_team_id, away_team_id, season_year, api_key, max_games=5):
    """
    Look at up to `max_games` recent games between these two teams this season
    and create a small adjustment based on average margin.
    Positive => home historically does better.
    """
    params = {
        "seasons[]": season_year,
        "team_ids[]": home_team_id,
        "per_page": 100,
    }
    cursor = None
    margins = []

    while True and len(margins) < max_games:
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        games_json = bdl_get("games", params=params, api_key=api_key)
        games = games_json.get("data", [])
        meta = games_json.get("meta", {}) or {}
        cursor = meta.get("next_cursor")

        for g in games:
            if len(margins) >= max_games:
                break

            home = g["home_team"]["id"]
            away = g["visitor_team"]["id"]
            if not (
                (home == home_team_id and away == away_team_id)
                or (home == away_team_id and away == home_team_id)
            ):
                continue

            home_score = g.get("home_team_score", 0) or 0
            away_score = g.get("visitor_team_score", 0) or 0
            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue

            # margin from home-team perspective (for THIS game)
            margin = home_score - away_score
            # but we want margin from "our" home_team_id perspective:
            if home != home_team_id:
                margin = -margin
            margins.append(margin)

        if not cursor:
            break

    if not margins:
        return 0.0

    avg_margin = sum(margins) / len(margins)
    # convert to a small adjustment, capped
    adj = max(min(avg_margin / 5.0, 2.0), -2.0)
    return adj


# -----------------------------
# Main daily engine
# -----------------------------

def run_daily_probs_for_date(
    game_date="12/04/2025",
    odds_dict=None,
    spreads_dict=None,
    stats_df=None,
    api_key=None,
    edge_threshold=0.03,
    lam=0.20,
):
    """
    Run the full model for one NBA date.
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

    # Injuries
    try:
        injury_df = fetch_injury_report_espn()
    except Exception as e:
        print(f"Warning: failed to fetch ESPN injuries: {e}")
        injury_df = pd.DataFrame(columns=["Player", "Team", "Pos", "Status", "Injury"])

    rows = []

    for _, g in games_df.iterrows():
        home_name = g["HOME_TEAM_NAME"]
        away_name = g["AWAY_TEAM_NAME"]

        home_row = find_team_row(home_name, stats_df)
        away_row = find_team_row(away_name, stats_df)
        home_id = int(home_row["TEAM_ID"])
        away_id = int(away_row["TEAM_ID"])

        # Base matchup
        base_score = season_matchup_base_score(home_row, away_row)

        # Injuries
        home_inj = build_injury_list_for_team_espn(home_name, injury_df)
        away_inj = build_injury_list_for_team_espn(away_name, injury_df)
        inj_adj = injury_adjustment(home_inj, away_inj)

        # Schedule fatigue
        home_last = get_team_last_game_date(home_id, game_date_obj, season_year, api_key)
        away_last = get_team_last_game_date(away_id, game_date_obj, season_year, api_key)

        home_rest_days = (game_date_obj - home_last).days if home_last else None
        away_rest_days = (game_date_obj - away_last).days if away_last else None

        home_fatigue = rest_days_to_fatigue_adjustment(home_rest_days)
        away_fatigue = rest_days_to_fatigue_adjustment(away_rest_days)

        fatigue_adj = home_fatigue - away_fatigue  # positive helps home

        # Head-to-head historical adjustment
        h2h_adj = compute_head_to_head_adjustment(home_id, away_id, season_year, api_key)

        # Final score
        adj_score = base_score + inj_adj + fatigue_adj + h2h_adj

        model_spread = score_to_spread(adj_score)  # home spread
        model_home_prob = score_to_prob(adj_score, lam)

        # Market odds
        key = (home_name, away_name)
        odds_info = odds_dict.get(key)
        if odds_info is None:
            print(f"[run_daily] No odds found for {home_name} vs {away_name}")
            odds_info = {}

        home_ml = odds_info.get("home_ml")
        away_ml = odds_info.get("away_ml")

        if home_ml is not None and away_ml is not None:
            home_imp = american_to_implied_prob(home_ml)
            away_imp = american_to_implied_prob(away_ml)
        else:
            home_imp = 0.5
            away_imp = 0.5

        edge_home = model_home_prob - home_imp
        edge_away = (1.0 - model_home_prob) - away_imp

                # Spreads
        home_spread = spreads_dict.get(key, odds_info.get("home_spread"))
        if home_spread is not None:
            home_spread = float(home_spread)
            # model_spread: predicted home margin
            # home_spread: market line (home -pts if negative, +pts if positive)
            # Positive spread_edge_home = model thinks home should be more of a favorite
            spread_edge_home = model_spread - home_spread
        else:
            spread_edge_home = None

        # -------------------------
        # Recommendation logic
        # -------------------------

        # 1) Moneyline recommendation
        ml_rec = "No strong ML edge"
        if home_ml is not None and away_ml is not None:
            if edge_home > edge_threshold:
                ml_rec = f"Bet HOME ML ({home_ml:+})"
            elif edge_away > edge_threshold:
                ml_rec = f"Bet AWAY ML ({away_ml:+})"

        # 2) Spread recommendation (separate threshold in points)
        spread_rec = "No strong spread edge"
        spread_threshold_pts = 3.0  # tweak if you want tighter/looser filter

        if home_spread is not None and spread_edge_home is not None:
            if spread_edge_home > spread_threshold_pts:
                # model likes home side vs line
                if home_spread > 0:
                    line_str = f"home +{abs(home_spread)}"
                elif home_spread < 0:
                    line_str = f"home {home_spread}"
                else:
                    line_str = "home pk"
                spread_rec = f"Bet HOME spread ({line_str})"

            elif spread_edge_home < -spread_threshold_pts:
                # model likes away side vs line
                if home_spread > 0:
                    line_str = f"away -{abs(home_spread)}"
                elif home_spread < 0:
                    line_str = f"away +{abs(home_spread)}"
                else:
                    line_str = "away pk"
                spread_rec = f"Bet AWAY spread ({line_str})"

               # 3) Primary recommendation: compare ML vs spread on a similar scale
        # Treat spread edge in "probability" units ~ 0.04 win-prob per point of edge
        primary_rec = "No clear edge"

        # Only count ML edge if we actually like a side
        if home_ml is not None and away_ml is not None and ml_rec != "No strong ML edge":
            ml_edge_abs = abs(edge_home)
        else:
            ml_edge_abs = 0.0

        # Only count spread edge if we actually like a side
        if spread_edge_home is not None and spread_rec != "No strong spread edge":
            spread_edge_prob = min(abs(spread_edge_home) * 0.04, 0.5)  # cap at 50% edge
        else:
            spread_edge_prob = 0.0

        if ml_edge_abs == 0.0 and spread_edge_prob == 0.0:
            primary_rec = "No clear edge"
        elif ml_edge_abs >= spread_edge_prob:
            primary_rec = f"ML: {ml_rec}"
        else:
            primary_rec = f"Spread: {spread_rec}"

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
# Add absolute edge (already existed)
# ------------------------------------
df["abs_edge_home"] = df["edge_home"].abs()

# ------------------------------------
# Add Value Tier Classification
# ------------------------------------
def classify_value(edge):
    if edge >= 0.20:
        return "HIGH VALUE"
    elif edge >= 0.10:
        return "MEDIUM VALUE"
    else:
        return "LOW VALUE"

df["value_tier"] = df["abs_edge_home"].apply(classify_value)

# ------------------------------------
# Sort by strongest edges
# ------------------------------------
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

    # 1) Ensure odds template exists (optional, you can comment this out if you prefer)
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
