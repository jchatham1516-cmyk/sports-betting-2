# sports/nhl/model.py
from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.nhl.injuries import (
    fetch_espn_nhl_injuries,
    build_injury_list_for_team_nhl,
    injury_adjustment_points,
)

ELO_PATH = "results/elo_state_nhl.json"

# ----------------------------
# Tunables (NHL-specific)
# ----------------------------
HOME_ADV = 45.0
ELO_K = 18.0
ELO_PER_GOAL = 55.0
MAX_ABS_MODEL_SPREAD = 2.0
INJ_ELO_PER_POINT = 18.0
GOALIE_IMPACT_THRESHOLD = 3.0
GOALIE_EXTRA_ELO_PER_POINT = 16.0
B2B_PENALTY_ELO = -18.0
ONE_DAY_REST_ELO = -6.0
THREE_PLUS_BONUS_ELO = +3.0
BASE_COMPRESS = 0.65
GOALIE_UNCERTAINTY_MULT = 0.85
INJ_PTS_CLAMP = 6.0

# ----------------------------
# Helper functions
# ----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float("nan")

def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def _parse_iso_date(s: str) -> Optional[date]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date()
    except Exception:
        return None

def _rest_adjustment_days(days_rest: Optional[int]) -> float:
    if days_rest is None:
        return 0.0
    if days_rest <= 0:
        return float(B2B_PENALTY_ELO)
    if days_rest == 1:
        return float(ONE_DAY_REST_ELO)
    if days_rest >= 3:
        return float(THREE_PLUS_BONUS_ELO)
    return 0.0

def _build_last_game_date_map(days_back: int = 10) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nhl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 10))
    last_played: Dict[str, date] = {}
    for ev in events:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        if not home_raw or not away_raw:
            continue
        home = canon_team(home_raw)
        away = canon_team(away_raw)
        d = _parse_iso_date(ev.get("commence_time") or "")
        if d is None:
            continue
        if (home not in last_played) or (d > last_played[home]):
            last_played[home] = d
        if (away not in last_played) or (d > last_played[away]):
            last_played[away] = d
    return last_played

def update_elo_from_recent_scores(days_from: int = 3) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nhl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 3))
    for ev in events:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        scores = ev.get("scores")
        if not home_raw or not away_raw or not scores:
            continue
        home = canon_team(home_raw)
        away = canon_team(away_raw)
        game_key = f"{ev.get('id','')}|{ev.get('commence_time','')}|{home}|{away}"
        if st.is_processed(game_key):
            continue
        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = score_map.get(home_raw) or score_map.get(home)
            aw = score_map.get(away_raw) or score_map.get(away)
            hs = float(hs)
            aw = float(aw)
        except Exception:
            continue
        eh = st.get(home)
        ea = st.get(away)
        nh, na = elo_update(eh, ea, hs, aw, k=ELO_K, home_adv=HOME_ADV)
        st.set(home, nh)
        st.set(away, na)
        st.mark_processed(game_key)
    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)
    return st

def _goalie_cost(inj_list) -> float:
    if not inj_list:
        return 0.0
    role_w = {"starter": 1.0, "rotation": 0.55}
    total = 0.0
    for item in inj_list:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            continue
        _, role, mult, impact = item
        try:
            impact = float(impact)
            mult = float(mult)
        except Exception:
            continue
        if impact >= GOALIE_IMPACT_THRESHOLD:
            total += role_w.get(role, 0.6) * mult * impact
    return float(total)

# ----------------------------
# Main daily model function
# ----------------------------
def run_daily_nhl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=3)
    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    injuries_map = {}
    try:
        injuries_map = fetch_espn_nhl_injuries()
    except Exception as e:
        print(f"[nhl injuries] WARNING: failed to load ESPN injuries: {e}")

    last_played = _build_last_game_date_map(days_back=10)

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)

        eh = st.get(home)
        ea = st.get(away)

        # Rest/fatigue adjustments
        home_rest_days = away_rest_days = None
        if target_date is not None:
            h_last = last_played.get(home)
            a_last = last_played.get(away)
            if h_last is not None:
                home_rest_days = (target_date - h_last).days - 1
            if a_last is not None:
                away_rest_days = (target_date - a_last).days - 1
        home_rest_elo = _rest_adjustment_days(home_rest_days)
        away_rest_elo = _rest_adjustment_days(away_rest_days)
        rest_elo_adj = float(home_rest_elo - away_rest_elo)

        # Injuries + goalie
        home_inj = build_injury_list_for_team_nhl(home, injuries_map)
        away_inj = build_injury_list_for_team_nhl(away, injuries_map)
        inj_pts = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = _clamp(inj_pts, -INJ_PTS_CLAMP, INJ_PTS_CLAMP)
        inj_elo_adj = inj_pts * INJ_ELO_PER_POINT

        home_goalie_cost = _goalie_cost(home_inj)
        away_goalie_cost = _goalie_cost(away_inj)
        goalie_diff = float(away_goalie_cost - home_goalie_cost)
        goalie_elo_adj = goalie_diff * GOALIE_EXTRA_ELO_PER_POINT
        goalie_uncertain = (home_goalie_cost == 0.0 and away_goalie_cost == 0.0)

        eh_eff = eh + inj_elo_adj + goalie_elo_adj + rest_elo_adj
        ea_eff = ea
        p_home_raw = elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV)
        compress = BASE_COMPRESS * (GOALIE_UNCERTAINTY_MULT if goalie_uncertain else 1.0)
        p_home = 0.5 + compress * (float(p_home_raw) - 0.5)
        p_home = _clamp(p_home, 0.01, 0.99)

        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = -(float(elo_diff) / ELO_PER_GOAL)
        model_spread_home = _clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # Build row
        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "elo_diff": float(elo_diff),
            "inj_points": float(inj_pts),
            "goalie_diff": float(goalie_diff),
            "rest_days_home": np.nan if home_rest_days is None else float(home_rest_days),
            "rest_days_away": np.nan if away_rest_days is None else float(away_rest_days),
            "home_ml": _safe_float((oi or {}).get("home_ml")),
            "away_ml": _safe_float((oi or {}).get("away_ml")),
            "home_spread": _safe_float((oi or {}).get("home_spread")),
            "over_under": _safe_float((oi or {}).get("total")),  # added totals column
        })

    return pd.DataFrame(rows)

# Backwards-compatible alias
def run_daily_probs_for_date(
    game_date_str: str = None,
    *,
    game_date: str = None,
    odds_dict: dict = None,
    spreads_dict: dict = None,
    **kwargs,
) -> pd.DataFrame:
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")
    return run_daily_nhl(str(date_in), odds_dict=(odds_dict or {}))
