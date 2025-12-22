# sports/nba/model.py
from __future__ import annotations

import os
import math
from collections import defaultdict
from datetime import datetime, date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.nba.injuries import (
    fetch_official_nba_injuries as fetch_espn_nba_injuries,
    build_injury_list_for_team_nba,
    injury_adjustment_points,
)

ELO_PATH = "results/elo_state_nba.json"

# ----------------------------
# Tunables (NBA-specific)
# ----------------------------
HOME_ADV = 55.0
ELO_K = 20.0
ELO_PER_POINT = 40.0

# Injury scaling
MAX_ABS_INJ_ELO_ADJ = 45.0
MAX_ABS_INJ_POINTS = 6.0
INJ_ELO_PER_POINT = 6.0

MAX_ABS_MODEL_SPREAD = 17.0

# Rest effects
SHORT_REST_PENALTY_ELO = -14.0
NORMAL_REST_BONUS_ELO = 0.0
BYE_BONUS_ELO = +8.0

# Recent form
FORM_LOOKBACK_DAYS = 35
FORM_MIN_GAMES = 2
FORM_ELO_PER_POINT = 1.35
FORM_ELO_CLAMP = 40.0

# Prob compression
BASE_COMPRESS = 0.75

# ML threshold
MIN_ML_EDGE = 0.02

# ATS model
ATS_SD_PTS = 13.5
ATS_DEFAULT_PRICE = -110.0
ATS_MIN_EDGE_VS_BE = 0.03
ATS_MIN_PTS_EDGE = 2.0
ATS_BIG_LINE = 7.0
ATS_TINY_MODEL = 2.0
ATS_BIGLINE_FORCE_PASS = True

# ----------------------------
# Helpers
# ----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return float("nan")

def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
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

def _calc_days_off(target: Optional[date], last: Optional[date]) -> Optional[int]:
    if target is None or last is None:
        return None
    delta = (target - last).days - 1
    if delta < 0 or delta > 30:
        return None
    return delta

def _rest_elo(days_off: Optional[int]) -> float:
    if days_off is None:
        return 0.0
    if days_off <= 4:
        return float(SHORT_REST_PENALTY_ELO)
    if days_off >= 10:
        return float(BYE_BONUS_ELO)
    return float(NORMAL_REST_BONUS_ELO)

# ... (other helper functions remain unchanged) ...

# ----------------------------
# Main daily run
# ----------------------------
def run_daily_nba(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = EloState.load(ELO_PATH) if os.path.exists(ELO_PATH) else EloState()

    # Parse date
    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # Load injuries once
    try:
        injuries_map = fetch_espn_nba_injuries()
    except Exception as e:
        print(f"[nba injuries] WARNING: failed to load injuries: {e}")
        injuries_map = {}

    # Rest map
    last_played = _build_last_game_date_map(days_back=21)

    # Recent form
    form_map = _recent_form_adjustments(days_back=FORM_LOOKBACK_DAYS)

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)

        eh = st.get(home)
        ea = st.get(away)

        # Rest days
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Injuries
        home_inj = build_injury_list_for_team_nba(home, injuries_map)
        away_inj = build_injury_list_for_team_nba(away, injuries_map)
        inj_pts_raw = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = _clamp(inj_pts_raw, -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        # Form
        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = form_home - form_away

        # Effective Elo
        eh_eff = eh + rest_adj + 0.5 * inj_pts + 0.5 * form_diff
        ea_eff = ea - 0.5 * inj_pts - 0.5 * form_diff

        # Win probability
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_home = _clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99)

        # Spread-ish
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(-(elo_diff / ELO_PER_POINT), -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # Market odds
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=-110.0)

        # Market no-vig prob
        mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)
        edge_home = p_home - mkt_home_p if not np.isnan(mkt_home_p) else float("nan")
        edge_away = -edge_home if not np.isnan(edge_home) else float("nan")

        ml_pick = _ml_recommendation(p_home, mkt_home_p)
        value_tier = _pick_value_tier(abs(edge_home)) if not np.isnan(edge_home) else "UNKNOWN"

        # ATS calculations
        spread_edge_home = home_spread - model_spread_home if not np.isnan(home_spread) else float("nan")
        p_home_cover = _cover_prob_from_edge(spread_edge_home, sd_pts=ATS_SD_PTS)
        ats_side, ats_p_win, ats_edge_vs_be, ats_be = _ats_pick_and_edge(p_home_cover, spread_price=spread_price)

        # ATS gating
        ats_pass_reason = ""
        ats_allowed = True
        if np.isnan(home_spread) or np.isnan(model_spread_home):
            ats_allowed = False
            ats_pass_reason = "missing spread"
        elif ATS_BIGLINE_FORCE_PASS and abs(home_spread) >= 7.0 and abs(model_spread_home) <= 2.0:
            ats_allowed = False
            ats_pass_reason = "big market line but tiny model line"
        elif ats_allowed and (np.isnan(ats_edge_vs_be) or ats_edge_vs_be < 0.03):
            ats_allowed = False
            ats_pass_reason = f"ats_edge_vs_be<{0.03}"
        elif ats_allowed and (np.isnan(spread_edge_home) or abs(spread_edge_home) < 2.0):
            ats_allowed = False
            ats_pass_reason = f"|spread_edge|<2.0"

        if not ats_allowed:
            ats_strength = "pass"
            spread_reco = "No ATS bet (gated)"
        else:
            ats_strength = _ats_strength_label(ats_edge_vs_be)
            spread_reco = _ats_reco(ats_side, ats_strength)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": p_home,
            "model_spread_home": model_spread_home,
            "market_home_prob": mkt_home_p,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_edge_home": spread_edge_home,
            "ml_recommendation": ml_pick,
            "spread_recommendation": spread_reco,
            "value_tier": value_tier,
            "ats_side": ats_side,
            "ats_strength": ats_strength,
            "ats_edge_vs_be": ats_edge_vs_be,
            "ats_be": ats_be,
        })

    return pd.DataFrame(rows)

def run_daily_probs_for_date(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    return run_daily_nba(game_date_str, odds_dict=odds_dict)
