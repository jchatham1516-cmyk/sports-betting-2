# Daily NBA betting model using BallDontLie + ESPN injuries + The Odds API for odds.
#
# What it does:
# - Uses BallDontLie to get:
#     - All teams
#     - All regular-season games up to the model date
#     - Daily schedule for a specific date
# - Builds simple team ratings from:
#     - Average points scored (ORtg proxy)
#     - Average points allowed (DRtg proxy)
# - Uses ESPN injuries page to adjust matchup score
# - Uses The Odds API to pull daily moneyline & spread odds
# - Computes a matchup score → win probability → model spread
# - Outputs ALL games for the date with:
#     - model_home_prob, edges, model_spread, and a recommendation
# - Saves CSV under results/predictions_MM-DD-YYYY.csv

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
# Odds API helpers
# -----------------------------

def get_odds_api_key():
    """
    Read Odds API key from environment.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if api_key is None:
        api_key = ""
    api_key = api_key.strip()
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY environment variable is not set or is empty. "
            "Set it as a GitHub Actions secret."
        )
    return api_key


def odds_get(path, params=None, api_key=None, timeout=30):
    """
    Generic GET for The Odds API v4.
    """
    if api_key is None:
        api_key = get_odds_api_key()

    url = ODDS_API_BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    params = dict(params or {})
    params["apiKey"] = api_key

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_odds_for_date(
    game_date_str,
    sport_key="basketball_nba",
    region="us",
    bookmaker_preference=None,
    api_key=None,
):
    # Fetch moneyline + spread odds for NBA games from The Odds API v4.
    # We don't filter by date here. We grab all upcoming NBA odds and the
    # model will match by (home_team, away_team) when looping today's games.
    if api_key is None:
        api_key = get_odds_api_key()

    if bookmaker_preference is None:
        bookmaker_preference = ["draftkings", "fanduel", "betmgm"]

    params = {
        "regions": region,
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }

    events = odds_get(f"sports/{sport_key}/odds", params=params, api_key=api_key)
    print(f"[fetch_odds_for_date] Got {len(events)} events from The Odds API")

    odds_dict = {}

    for ev in events:
        home_team = ev["home_team"]
        away_team = ev["away_team"]
        bookmakers = ev.get("bookmakers", []) or []

        if not bookmakers:
            continue

        # Preferred bookmaker or fallback
        chosen = None
        by_key = {b["key"]: b for b in bookmakers if "key" in b}
        for bk in bookmaker_preference:
            if bk in by_key:
                chosen = by_key[bk]
                break
        if chosen is None:
            chosen = bookmakers[0]

        home_ml = None
        away_ml = None
        home_spread = None

        for m in chosen.get("markets", []):
            mkey = m.get("key")
            outcomes = m.get("outcomes", []) or []

            if mkey == "h2h":
                # moneyline
                for o in outcomes:
                    name = o.get("name")
                    price = o.get("price")
                    if name == home_team:
                        home_ml = price
                    elif name == away_team:
                        away_ml = price

            elif mkey == "spreads":
                # spread
                for o in outcomes:
                    name = o.get("name")
                    point = o.get("point")
                    if name == home_team:
                        home_spread = point

        odds_dict[(home_team, away_team)] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
        }

    print(f"[fetch_odds_for_date] Built odds entries for {len(odds_dict)} games.")
    print("[fetch_odds_for_date] Sample keys:", list(odds_dict.keys())[:5])
    return odds_dict
    
def fetch_odds_for_date(
    game_date_str,
    sport_key="basketball_nba",
    region="us",
    bookmaker_preference=None,
    api_key=None,
):
    # Fetch moneyline + spread odds for NBA games from The Odds API v4.
    # We don't filter by date here. We grab all upcoming NBA odds and the
    # model will match by (home_team, away_team) when looping today's games.

    if api_key is None:
        api_key = get_odds_api_key()

    if bookmaker_preference is None:
        bookmaker_preference = ["draftkings", "fanduel", "betmgm"]

    params = {
        "regions": region,
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }

    events = odds_get(f"sports/{sport_key}/odds", params=params, api_key=api_key)
    print(f"[fetch_odds_for_date] Got {len(events)} events from The Odds API")

    odds_dict = {}

    for ev in events:
        home_team = ev["home_team"]
        away_team = ev["away_team"]
        bookmakers = ev.get("bookmakers", []) or []

        if not bookmakers:
            continue

        # Preferred bookmaker or fallback
        chosen = None
        by_key = {b["key"]: b for b in bookmakers if "key" in b}
        for bk in bookmaker_preference:
            if bk in by_key:
                chosen = by_key[bk]
                break
        if chosen is None:
            chosen = bookmakers[0]

        home_ml = None
        away_ml = None
        home_spread = None

        for m in chosen.get("markets", []):
            mkey = m.get("key")
            outcomes = m.get("outcomes", []) or []

            if mkey == "h2h":
                # moneyline
                for o in outcomes:
                    name = o.get("name")
                    price = o.get("price")
                    if name == home_team:
                        home_ml = price
                    elif name == away_team:
                        away_ml = price

            elif mkey == "spreads":
                # spread
                for o in outcomes:
                    name = o.get("name")
                    point = o.get("point")
                    if name == home_team:
                        home_spread = point

        odds_dict[(home_team, away_team)] = {
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
        }

    print(f"[fetch_odds_for_date] Built odds entries for {len(odds_dict)} games.")
    print("[fetch_odds_for_date] Sample keys:", list(odds_dict.keys())[:5])
    return odds_dict
# -----------------------------
# BallDontLie low-level client
# -----------------------------

BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"


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
            # Handle rate limiting explicitly
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
    Build a simple team rating DataFrame from BallDontLie games:

    - ORtg proxy  = average points scored per game
    - DRtg proxy  = average points allowed per game
    - W, L, W_PCT from game results

    Returns DataFrame with columns:
      TEAM_ID, TEAM_NAME, GP, W, L, W_PCT, ORtg, DRtg, eFG, TOV, AST, ORB, DRB, FTAr
    (Advanced stats are zeroed but kept to match the rest of the model.)
    """
    # 1) Get all teams
    teams_json = bdl_get("teams", params={}, api_key=api_key)
    teams_data = teams_json.get("data", [])

    # Initialize aggregates
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
    #    docs: GET /v1/games with seasons[]=YYYY and end_date=YYYY-MM-DD, per_page, cursor :contentReference[oaicite:4]{index=4}
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
            away_team = g["visitor_team"]  # naming from docs :contentReference[oaicite:5]{index=5}

            home_id = home_team["id"]
            away_id = away_team["id"]
            home_score = g.get("home_team_score", 0) or 0
            away_score = g.get("visitor_team_score", 0) or 0

            # Skip games that have no score yet (0-0 and period==0 & status is start_time)
            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue

            # Home aggregates
            if home_id in agg:
                agg[home_id]["gp"] += 1
                agg[home_id]["pts_for"] += home_score
                agg[home_id]["pts_against"] += away_score
            # Away aggregates
            if away_id in agg:
                agg[away_id]["gp"] += 1
                agg[away_id]["pts_for"] += away_score
                agg[away_id]["pts_against"] += home_score

            # Wins / losses
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

    # 3) Build DataFrame
    rows = []
    for tid, rec in agg.items():
        gp = rec["gp"]
        wins = rec["wins"]
        losses = rec["losses"]
        w_pct = wins / gp if gp > 0 else 0.0
        or_p = rec["pts_for"] / gp if gp > 0 else 0.0
        dr_p = rec["pts_against"] / gp if gp > 0 else 0.0

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
                # Keep advanced columns as zeros for now
                "eFG": 0.0,
                "TOV": 0.0,
                "AST": 0.0,
                "ORB": 0.0,
                "DRB": 0.0,
                "FTAr": 0.0,
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

    # Exact
    full_match = stats_df[stats_df["TEAM_NAME"].str.lower() == name]
    if not full_match.empty:
        return full_match.iloc[0]

    # Contains
    contains_match = stats_df[stats_df["TEAM_NAME"].str.lower().str.contains(name)]
    if not contains_match.empty:
        return contains_match.iloc[0]

    raise ValueError(f"Could not find a team matching: {team_name_input}")


def season_matchup_score(home_row, away_row):
    """
    Linear scoring model:
    Positive score => home team stronger.

    Here ORtg/DRtg are simple PTS_for/PTS_against per game built from BallDontLie.
    Advanced stats are zeros, so only ORtg/DRtg and home edge drive results.
    """
    h = home_row
    a = away_row

    d_ORtg = h["ORtg"] - a["ORtg"]
    d_DRtg = a["DRtg"] - h["DRtg"]   # lower DRtg is better → flip

    home_edge = 2.0  # base home-court advantage (tune if you like)

    score = (
        home_edge
        + 0.08 * d_ORtg
        + 0.08 * d_DRtg
        # advanced metrics currently omitted (all zero)
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
    Scrape ESPN NBA injuries page into a DataFrame with columns like:
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

    # Standardize columns if possible
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
    In a real model you’d plug in RAPTOR, BPM, on/off, etc.
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


def build_injury_list_for_team_espn(team_name_or_abbrev, injury_df):
    """
    Build a list of injuries for a given team from ESPN injuries DataFrame.
    Returns list of tuples: (player_name, role, multiplier)
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

        injuries.append((name, role, mult))

    return injuries


def injury_adjustment(home_injuries=None, away_injuries=None):
    """
    home_injuries / away_injuries:
      - list of tuples: [("Player Name", "role", mult), ...]
        where role in {"star", "starter", "rotation", "bench"}
      - or list of dicts: [{"name": "...", "role": "starter", "mult": 1.0}, ...]

    Missing players on the HOME team LOWER the score.
    Missing players on the AWAY team RAISE the score (helps home).
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    def parse_inj_list(lst, sign):
        adj = 0.0
        for item in lst:
            if isinstance(item, dict):
                role = item.get("role", "starter")
                mult = float(item.get("mult", 1.0))
            else:
                # tuple: (name, role, mult?) or (name, role)
                if len(item) == 3:
                    _, role, mult = item
                    mult = float(mult)
                elif len(item) == 2:
                    _, role = item
                    mult = 1.0
                else:
                    role = "starter"
                    mult = 1.0
            weight = INJURY_WEIGHTS.get(role, INJURY_WEIGHTS["starter"])
            adj += sign * weight * mult
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

    Uses /v1/games?dates[]=YYYY-MM-DD :contentReference[oaicite:6]{index=6}
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
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Main daily engine
# -----------------------------

def run_daily_probs_for_date(
    game_date="12/04/2025",
    odds_dict=None,      # {(home, away): {"home_ml":..., "away_ml":..., "home_spread":...}}
    spreads_dict=None,   # OPTIONAL: {(home, away): home_spread}
    stats_df=None,
    api_key=None,
    edge_threshold=0.03,
    lam=0.20,
):
    """
    Run the full model for one NBA date.

    Returns a DataFrame with:
      home, away, model_home_prob, market_home_prob (if odds),
      edge_home, edge_away, model_spread, spread_edge_home (if spreads),
      recommended_bet (string).
    """
    if api_key is None:
        api_key = get_bdl_api_key()

    if stats_df is None:
        raise ValueError("stats_df must be precomputed for BallDontLie version.")

    if odds_dict is None:
        odds_dict = {}

    if spreads_dict is None:
        spreads_dict = {}

    # Fetch schedule from BallDontLie
    games_df = fetch_games_for_date(game_date, stats_df, api_key)

    # Fetch injuries (ESPN)
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

        # Build injuries
        home_inj = build_injury_list_for_team_espn(home_name, injury_df)
        away_inj = build_injury_list_for_team_espn(away_name, injury_df)

        base_score = season_matchup_score(home_row, away_row)
        inj_adj = injury_adjustment(home_inj, away_inj)
        adj_score = base_score + inj_adj

        model_spread = score_to_spread(adj_score)  # positive = home favorite
        model_home_prob = score_to_prob(adj_score, lam)

        # Market data (if provided)
        key = (home_name, away_name)
        odds_info = odds_dict.get(key)
        if odds_info is None:
            # debug: show mismatch if any
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

        # Spreads (optional)
        home_spread = spreads_dict.get(key, odds_info.get("home_spread"))
        if home_spread is not None:
            spread_edge_home = model_spread - float(home_spread)
        else:
            spread_edge_home = None

        # Recommendation logic:
        rec = "No clear edge"

        if home_ml is not None and away_ml is not None:
            if edge_home > edge_threshold:
                rec = f"Bet HOME ML ({home_ml:+})"
            elif edge_away > edge_threshold:
                rec = f"Bet AWAY ML ({away_ml:+})"
            else:
                rec = "No strong ML edge"
        elif home_spread is not None and spread_edge_home is not None:
            if spread_edge_home > 1.5:
                if home_spread > 0:
                    line_str = f"home +{abs(home_spread)}"
                elif home_spread < 0:
                    line_str = f"home {home_spread}"
                else:
                    line_str = "home pk"
                rec = f"Bet HOME spread ({line_str})"
            elif spread_edge_home < -1.5:
                if home_spread > 0:
                    line_str = f"away -{abs(home_spread)}"
                elif home_spread < 0:
                    line_str = f"away +{abs(home_spread)}"
                else:
                    line_str = "away pk"
                rec = f"Bet AWAY spread ({line_str})"
            else:
                rec = "No strong spread edge"
        else:
            # No odds at all: just lean ML with a confidence threshold
            if model_home_prob > 0.55:
                rec = "Lean HOME ML (no market odds provided)"
            elif model_home_prob < 0.45:
                rec = "Lean AWAY ML (no market odds provided)"
            else:
                rec = "No clear edge (no market odds provided)"

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
                "recommendation": rec,
            }
        )

    df = pd.DataFrame(rows)
    df["abs_edge_home"] = df["edge_home"].abs()
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

    # You can optionally hardcode your odds here:
    # odds_dict = {
    #   ("Cleveland Cavaliers", "Boston Celtics"): {
    #       "home_ml": -120,
    #       "away_ml": +105,
    #       "home_spread": -2.5,
    #   },
    # }
    odds_dict = {}
    spreads_dict = {}

    # Determine season year for BallDontLie based on the game date
    game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
    season_year = season_start_year_for_date(game_date_obj)
    end_date_iso = game_date_obj.strftime("%Y-%m-%d")

    # Fetch team ratings from BallDontLie
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

    # Run daily model; also fail gracefully if something blows up
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

    # Ensure output directory
    os.makedirs("results", exist_ok=True)
    out_name = f"results/predictions_{game_date.replace('/', '-')}.csv"
    results_df.to_csv(out_name, index=False)

    # Pretty print to console
    with pd.option_context("display.max_columns", None):
        print(results_df)

    print(f"\nSaved predictions to {out_name}")


if __name__ == "__main__":
    main()
