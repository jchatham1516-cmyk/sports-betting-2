
"""
Clean daily NBA betting model script.

What it does:
- Fetches team advanced stats from nba_api
- Fetches the NBA schedule for a given date
- Pulls injuries from ESPN
- Computes a matchup score, converts to win prob & model spread
- Optionally compares vs betting lines and suggests a bet
- Writes a CSV in results/ and prints a sorted table

To run locally:
    python current_of_sports_betting_algorithm.py --date 12/04/2025
"""

import os
import sys
import math
import argparse
from datetime import datetime, date
import time  # <-- ADD THIS

import numpy as np
import pandas as pd
import requests  # <-- ADD THIS

from nba_api.stats.endpoints import leaguedashteamstats, ScoreboardV2

# -----------------------------
# Helpers: seasons & odds
# -----------------------------

def current_season_str(today=None):
    """
    Return NBA season string like '2025-26' based on today's date.
    """
    if today is None:
        today = date.today()
    year = today.year
    # NBA season usually starts in October; if before August, we are in prior season year
    if today.month < 8:
        start_year = year - 1
    else:
        start_year = year
    end_year_short = (start_year + 1) % 100
    return f"{start_year}-{end_year_short:02d}"


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
# Team stats
# -----------------------------

def fetch_team_advanced_stats(season=None, max_retries=3, pause_seconds=5):
    """
    Fetch league-wide team advanced stats for a given season and return a DataFrame
    with columns standardized to:
      TEAM_ID, TEAM_NAME, ORtg, DRtg, eFG, TOV, AST, ORB, DRB, FTAr

    Retries a few times if stats.nba.com times out.
    """
    if season is None:
        season = current_season_str()

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            ).get_data_frames()[0]

            # If we got here, request succeeded, break out of retry loop
            last_exc = None
            break

        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            print(
                f"[fetch_team_advanced_stats] ReadTimeout talking to stats.nba.com "
                f"(attempt {attempt}/{max_retries})."
            )
            if attempt < max_retries:
                print(f"Sleeping {pause_seconds} seconds before retry...")
                time.sleep(pause_seconds)
        except Exception as e:
            last_exc = e
            print(
                f"[fetch_team_advanced_stats] Error talking to stats.nba.com "
                f"(attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                print(f"Sleeping {pause_seconds} seconds before retry...")
                time.sleep(pause_seconds)

    if last_exc is not None:
        raise RuntimeError(
            f"Failed to fetch team stats for season {season} after {max_retries} attempts"
        ) from last_exc

    # Keep a manageable subset of columns
    cols = [
        "TEAM_ID",
        "TEAM_NAME",
        "GP",
        "W",
        "L",
        "W_PCT",
        "OFF_RATING",
        "DEF_RATING",
        "EFG_PCT",
        "TM_TOV_PCT",
        "AST_PCT",
        "OREB_PCT",
        "DREB_PCT",
    ]
    stats = stats[cols].copy()

    stats.rename(
        columns={
            "OFF_RATING": "ORtg",
            "DEF_RATING": "DRtg",
            "EFG_PCT": "eFG",
            "TM_TOV_PCT": "TOV",
            "AST_PCT": "AST",
            "OREB_PCT": "ORB",
            "DREB_PCT": "DRB",
        },
        inplace=True,
    )

    # We don't have FTAr here, keep as 0.0 so Δ_FTAr is 0
    stats["FTAr"] = 0.0

    return stats



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


# -----------------------------
# Matchup scoring & probabilities
# -----------------------------

def season_matchup_score(home_row, away_row):
    """
    Linear scoring model:
    Positive score => home team stronger.
    """
    h = home_row
    a = away_row

    d_ORtg = h["ORtg"] - a["ORtg"]
    d_DRtg = a["DRtg"] - h["DRtg"]   # lower DRtg is better → flip
    d_eFG  = h["eFG"]  - a["eFG"]
    d_TOV  = a["TOV"]  - h["TOV"]    # fewer turnovers better
    d_AST  = h["AST"]  - a["AST"]
    d_ORB  = h["ORB"]  - a["ORB"]
    d_DRB  = h["DRB"]  - a["DRB"]
    d_FTAr = h["FTAr"] - a["FTAr"]

    home_edge = 2.0  # base home-court advantage

    score = (
        home_edge
        + 0.08 * d_ORtg
        + 0.08 * d_DRtg
        + 40.0 * d_eFG
        + 30.0 * d_TOV
        + 20.0 * d_AST
        + 25.0 * d_ORB
        + 25.0 * d_DRB
        + 10.0 * d_FTAr
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
    # You can customize this however you want later.
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
# Schedule / games
# -----------------------------

def fetch_games_for_date(game_date_str, stats_df, max_retries=3, pause_seconds=5):
    """
    game_date_str format: 'MM/DD/YYYY'
    Returns a DataFrame with GAME_ID, HOME_TEAM_NAME, AWAY_TEAM_NAME

    Retries a few times if stats.nba.com times out.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            sb = ScoreboardV2(
                game_date=game_date_str,
                league_id="00",
                day_offset=0,
            )
            games_header = sb.get_data_frames()[0]  # GameHeader
            last_exc = None
            break
        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            print(
                f"[fetch_games_for_date] ReadTimeout talking to stats.nba.com "
                f"(attempt {attempt}/{max_retries})."
            )
            if attempt < max_retries:
                print(f"Sleeping {pause_seconds} seconds before retry...")
                time.sleep(pause_seconds)
        except Exception as e:
            last_exc = e
            print(
                f"[fetch_games_for_date] Error talking to stats.nba.com "
                f"(attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                print(f"Sleeping {pause_seconds} seconds before retry...")
                time.sleep(pause_seconds)

    if last_exc is not None:
        raise RuntimeError(
            f"Failed to fetch games for {game_date_str} after {max_retries} attempts"
        ) from last_exc

    id_to_name = dict(zip(stats_df["TEAM_ID"], stats_df["TEAM_NAME"]))

    games_header["HOME_TEAM_NAME"] = games_header["HOME_TEAM_ID"].map(id_to_name)
    games_header["AWAY_TEAM_NAME"] = games_header["VISITOR_TEAM_ID"].map(id_to_name)

    games_df = games_header[["GAME_ID", "HOME_TEAM_NAME", "AWAY_TEAM_NAME"]].copy()
    return games_df


# -----------------------------
# Main daily engine
# -----------------------------

def run_daily_probs_for_date(
    game_date="12/04/2025",
    odds_dict=None,      # {(home, away): {"home_ml":..., "away_ml":..., "home_spread":...}}
    spreads_dict=None,   # OPTIONAL: {(home, away): home_spread}
    stats_df=None,
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
    if stats_df is None:
        stats_df = fetch_team_advanced_stats()

    if odds_dict is None:
        odds_dict = {}

    if spreads_dict is None:
        spreads_dict = {}

    # Fetch schedule
    games_df = fetch_games_for_date(game_date, stats_df)

    # Fetch injuries
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
        odds_info = odds_dict.get(key, {})
        home_ml = odds_info.get("home_ml")
        away_ml = odds_info.get("away_ml")

        if home_ml is not None and away_ml is not None:
            home_imp = american_to_implied_prob(home_ml)
            away_imp = american_to_implied_prob(away_ml)
        else:
            # If no odds, assume market is a coin flip
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
        # 1) If we have ML odds, base recommendation on ML edge.
        # 2) Else if we have spreads, base it on spread edge.
        # 3) Else simple threshold on model_home_prob vs 0.5.
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
                # edge for away
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
    # Sort by biggest model edge on home side
    df["abs_edge_home"] = df["edge_home"].abs()
    df = df.sort_values("abs_edge_home", ascending=False).reset_index(drop=True)
    return df


# -----------------------------
# CLI / entrypoint
# -----------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Run daily NBA betting model.")
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

    season = current_season_str()

    # Fetch team stats with retry; fail gracefully if NBA API is down
    try:
        stats_df = fetch_team_advanced_stats(season=season)
    except Exception as e:
        print(f"Error: Failed to fetch team stats: {e}")
        print("Exiting without predictions so the workflow can complete gracefully.")
        return

    # Run daily model; also fail gracefully if something blows up
    try:
        results_df = run_daily_probs_for_date(
            game_date=game_date,
            odds_dict=odds_dict,
            spreads_dict=spreads_dict,
            stats_df=stats_df,
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
