import math
from datetime import datetime, date

import numpy as np
import pandas as pd

from sports.common.util import safe_float
from sports.nba.bdl_client import fetch_games_for_date, season_start_year_for_date, get_team_last_game_date
from sports.nba.injuries import (
    fetch_injury_report_official_nba,
    build_injury_list_for_team_official,
    injury_adjustment,
)


SPREAD_SCALE_FACTOR = 4.0
RECENT_FORM_WEIGHT = 0.35
SEASON_FORM_WEIGHT = 1.0 - RECENT_FORM_WEIGHT


MATCHUP_WEIGHTS = np.array([0.2, 0.12, 0.12, 0.03, 4.0, 4.0])


def find_team_row(team_name_input, stats_df):
    name = team_name_input.strip().lower()
    full_match = stats_df[stats_df["TEAM_NAME"].str.lower() == name]
    if not full_match.empty:
        return full_match.iloc[0]

    contains_match = stats_df[stats_df["TEAM_NAME"].str.lower().str.contains(name)]
    if not contains_match.empty:
        return contains_match.iloc[0]

    raise ValueError(f"Could not find a team matching: {team_name_input}")


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


def compute_head_to_head_adjustment(home_team_id, away_team_id, season_year, api_key, max_seasons_back=3):
    # Placeholder (you can fill later)
    return 0.0


def run_daily_probs_for_date(
    game_date: str,
    odds_dict=None,
    spreads_dict=None,
    stats_df=None,
    api_key=None,
    lam=0.25,
):
    """
    Returns a DataFrame with columns needed by recommendations.py:
      date, home, away, model_home_prob, home_ml, away_ml, home_spread, model_spread_home
    """
    if api_key is None:
        raise ValueError("api_key is required for NBA (BallDontLie).")
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

        home_ml = odds_info.get("home_ml", np.nan)
        away_ml = odds_info.get("away_ml", np.nan)

        home_spread = spreads_dict.get(key, odds_info.get("home_spread", np.nan))
        home_spread = safe_float(home_spread)
        if home_spread is None:
            home_spread = np.nan

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

    return pd.DataFrame(rows)

