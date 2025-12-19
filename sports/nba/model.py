# sports/nba/model.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.calibration import load_nba_calibrator, update_and_save_nba_calibration

ELO_PATH = "results/elo_state_nba.json"

# ---- Tunables ----
HOME_ADV = 65.0           # Elo home advantage (tunable)
ELO_K = 20.0              # Elo update aggressiveness (tunable)

# Convert elo_diff -> spread uses calibration; still sanity-clamp output:
MAX_ABS_MODEL_SPREAD = 15.0

# Injury impact mapping (inj_points -> elo points)
INJ_ELO_PER_POINT = 18.0  # tune this


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(np.nan)


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Injuries (auto-detected)
# -----------------------------
def _load_nba_injuries():
    """
    Tries to load your existing NBA injury implementation.
    Falls back to no injuries if it doesn't exist / fails.
    Expected (if present) in sports/nba/injuries.py:
      - fetch_official_nba_injuries()
      - build_injury_list_for_team_nba(team, injuries_map)
      - injury_adjustment_points(home_inj, away_inj)
    """
    try:
        from sports.nba.injuries import (
            fetch_official_nba_injuries,
            build_injury_list_for_team_nba,
            injury_adjustment_points,
        )
        return fetch_official_nba_injuries, build_injury_list_for_team_nba, injury_adjustment_points
    except Exception:
        return None, None, None


# -----------------------------
# Elo update from recent scores
# -----------------------------
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

        # Score map might use raw names; try raw first then canonical.
        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = score_map.get(home_raw)
            aw = score_map.get(away_raw)
            if hs is None:
                hs = score_map.get(home)
            if aw is None:
                aw = score_map.get(away)

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


# -----------------------------
# Daily run
# -----------------------------
def run_daily_nba(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with at least:
      date, home, away,
      model_home_prob, model_spread_home,
      elo_diff (REQUIRED for calibration training),
      home_ml, away_ml, home_spread
    """
    st = update_elo_from_recent_scores(days_from=3)

    # Update calibration from your historical saved prediction CSVs
    cal = load_nba_calibrator()
    cal = update_and_save_nba_calibration()

    # Load injuries once
    fetch_inj, build_list, inj_points_fn = _load_nba_injuries()
    injuries_map = {}
    if fetch_inj is not None:
        try:
            injuries_map = fetch_inj()
        except Exception as e:
            print(f"[nba injuries] WARNING: failed to load injuries: {e}")
            injuries_map = {}

    # Clean empty output on off-days / missing odds
    if not odds_dict:
        return pd.DataFrame(columns=[
            "date", "home", "away",
            "model_home_prob", "model_spread_home", "elo_diff", "inj_points",
            "home_ml", "away_ml", "home_spread",
        ])

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)

        eh = st.get(home)
        ea = st.get(away)

        # Injuries -> injury points -> elo adjustment
        inj_pts = 0.0
        inj_elo_adj = 0.0
        if build_list is not None and inj_points_fn is not None and injuries_map:
            try:
                home_inj = build_list(home, injuries_map)
                away_inj = build_list(away, injuries_map)
                inj_pts = float(inj_points_fn(home_inj, away_inj))  # + means away more hurt
                inj_pts = _clamp(inj_pts, -8.0, 8.0)  # stability clamp
                inj_elo_adj = inj_pts * INJ_ELO_PER_POINT
            except Exception:
                inj_pts = 0.0
                inj_elo_adj = 0.0

        # Elo-based win prob (with injury adjustment)
        p_home = elo_win_prob(eh + inj_elo_adj, ea, home_adv=HOME_ADV)

        # Elo diff (store this so calibration can learn)
        elo_diff = ((eh + inj_elo_adj) - ea) + HOME_ADV

        # Calibrated spread to historical closing lines
        model_spread_home = cal.predict_spread(elo_diff)

        # Sanity clamp for NBA spreads
        model_spread_home = _clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "elo_diff": float(elo_diff),
            "inj_points": float(inj_pts),
            "home_ml": _safe_float((oi or {}).get("home_ml")),
            "away_ml": _safe_float((oi or {}).get("away_ml")),
            "home_spread": _safe_float((oi or {}).get("home_spread")),
        })

    return pd.DataFrame(rows)
def run_daily_probs_for_date(
    game_date_str: str = None,
    *,
    game_date: str = None,
    odds_dict: dict,
) -> pd.DataFrame:
    """
    Backwards-compatible alias for older code paths.

    Some callers pass:
      run_daily_probs_for_date(game_date="12/19/2025", odds_dict=...)

    Others may pass:
      run_daily_probs_for_date(game_date_str="12/19/2025", odds_dict=...)
    """
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")
    return run_daily_nba(str(date_in), odds_dict=odds_dict)
