from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.nba.injuries import (
    fetch_espn_nba_injuries,
    build_injury_list_for_team_nba,
    injury_adjustment_points,
)

ELO_PATH = "results/elo_state_nba.json"

# ----------------------------
# Tunables (NBA-specific)
# ----------------------------
HOME_ADV = 100.0
ELO_K = 20.0
ELO_PER_POINT = 20.0
MAX_ABS_MODEL_SPREAD = 15.0
INJ_ELO_PER_POINT = 5.0
BASE_COMPRESS = 0.7
GOALIE_UNCERTAINTY_MULT = 1.0  # Not used for NBA
INJ_PTS_CLAMP = 6.0

# ----------------------------
# Helpers
# ----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
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


def _build_last_game_date_map(days_back: int = 10) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nba"]
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
    sport_key = SPORT_TO_ODDS_KEY["nba"]

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


# ----------------------------
# Main daily run
# ----------------------------
def run_daily_nba(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=3)

    # Parse date
    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # Load injuries once
    try:
        injuries_map = fetch_espn_nba_injuries()
    except Exception as e:
        print(f"[nba injuries] WARNING: failed to load ESPN injuries: {e}")
        injuries_map = {}

    # Rest map once
    last_played = _build_last_game_date_map(days_back=10)

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)

        eh = st.get(home)
        ea = st.get(away)

        # Rest days
        home_rest_days = None
        away_rest_days = None
        if target_date is not None:
            h_last = last_played.get(home)
            a_last = last_played.get(away)
            if h_last is not None:
                home_rest_days = (target_date - h_last).days - 1
            if a_last is not None:
                away_rest_days = (target_date - a_last).days - 1

        rest_adj = 0.0
        if home_rest_days is not None or away_rest_days is not None:
            home_adj = -1.0 * (home_rest_days or 0)
            away_adj = -1.0 * (away_rest_days or 0)
            rest_adj = home_adj - away_adj

        # Injuries
        home_inj = build_injury_list_for_team_nba(home, injuries_map)
        away_inj = build_injury_list_for_team_nba(away, injuries_map)

        inj_pts = _clamp(float(injury_adjustment_points(home_inj, away_inj)), -INJ_PTS_CLAMP, INJ_PTS_CLAMP)
        eh_eff = eh + rest_adj + inj_pts
        ea_eff = ea

        # Win prob + compression
        p_home_raw = elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV)
        p_home = _clamp(0.5 + BASE_COMPRESS * (p_home_raw - 0.5), 0.01, 0.99)

        # Spread-ish output
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(-(elo_diff / ELO_PER_POINT), -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # Market
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))
        model_total = _safe_float((oi or {}).get("model_total"))
        total_ou = _safe_float((oi or {}).get("total_ou"))

        if total_ou is not None and model_total is not None:
            total_edge = float(model_total - total_ou)
        else:
            total_edge = float("nan")

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "elo_diff": float(elo_diff),
            "inj_points": float(inj_pts),
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "model_total": model_total,
            "total_ou": total_ou,
            "total_edge": total_edge,
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
    return run_daily_nba(str(date_in), odds_dict=(odds_dict or {}))
