# sports/nfl/model.py
from __future__ import annotations

import os
from datetime import datetime, date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.nfl.injuries import (
    fetch_espn_nfl_injuries,
    build_injury_list_for_team_nfl,
    injury_adjustment_points,
)

ELO_PATH = "results/elo_state_nfl.json"

# ----------------------------
# Tunables (NFL-specific)
# ----------------------------
HOME_ADV = 55.0
ELO_K = 20.0

# Elo -> points (spread-ish)
ELO_PER_POINT = 28.0

# Caps for sanity
MAX_ABS_MODEL_SPREAD = 17.0
MAX_ABS_INJ_ELO_ADJ = 80.0  # cap injury Elo swing

# Base injuries -> Elo
INJ_ELO_PER_POINT = 14.0

# Extra QB weighting (QB matters most)
QB_EXTRA_ELO = 18.0  # added if QB is out/doubtful etc (via injury tuples)

# Rest / short-week effects (Elo)
SHORT_REST_PENALTY_ELO = -14.0   # <=4 days off
NORMAL_REST_BONUS_ELO = 0.0
BYE_BONUS_ELO = +8.0             # 10+ days off

# Probability compression: NFL has variance; shrink toward 0.5
BASE_COMPRESS = 0.75


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


def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nfl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 21))

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


def _rest_elo(days_off: Optional[int]) -> float:
    """
    days_off = days between games (not counting game days).
    NFL: short rest matters; byes help a bit.
    """
    if days_off is None:
        return 0.0
    if days_off <= 4:
        return float(SHORT_REST_PENALTY_ELO)
    if days_off >= 10:
        return float(BYE_BONUS_ELO)
    return float(NORMAL_REST_BONUS_ELO)


def _qb_cost(inj_list) -> float:
    """
    Detect QB-ish injuries from your injury tuples (player, role, mult, impact).
    In your NFL injuries mapping, QB impact is 4.0.
    We treat impact >= 3.7 as QB-like.
    """
    if not inj_list:
        return 0.0

    role_w = {"starter": 1.0, "rotation": 0.55}
    total = 0.0
    for item in inj_list:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            continue
        _, role, mult, impact = item
        try:
            mult = float(mult)
            impact = float(impact)
        except Exception:
            continue
        if impact >= 3.7:
            total += role_w.get(role, 0.6) * mult * impact
    return float(total)


def update_elo_from_recent_scores(days_from: int = 10) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nfl"]

    # NFL games are weekly; look back further than NHL/NBA
    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 21))

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


def run_daily_nfl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    """
    ML-first NFL model:
      - canonical teams always
      - weekly Elo updates
      - rest penalties (short rest / bye)
      - injury adjustment with QB weighting
      - probability compression
      - spread cap
    """
    st = update_elo_from_recent_scores(days_from=14)

    # Parse date
    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # Load injuries once
    injuries_map = {}
    try:
        injuries_map = fetch_espn_nfl_injuries()
    except Exception as e:
        print(f"[nfl injuries] WARNING: failed to load ESPN injuries: {e}")
        injuries_map = {}

    # Rest map once
    last_played = _build_last_game_date_map(days_back=21)

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)

        eh = st.get(home)
        ea = st.get(away)

        # Rest days (days off between games)
        home_days_off = None
        away_days_off = None
        if target_date is not None:
            hl = last_played.get(home)
            al = last_played.get(away)
            if hl is not None:
                home_days_off = (target_date - hl).days - 1
            if al is not None:
                away_days_off = (target_date - al).days - 1

        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Injuries
        home_inj = build_injury_list_for_team_nfl(home, injuries_map)
        away_inj = build_injury_list_for_team_nfl(away, injuries_map)

        inj_pts = float(injury_adjustment_points(home_inj, away_inj))  # + means away more hurt
        inj_elo_adj = inj_pts * INJ_ELO_PER_POINT

        # QB extra weighting (difference in QB injury cost)
        qb_diff = _qb_cost(away_inj) - _qb_cost(home_inj)
        qb_elo_adj = qb_diff * QB_EXTRA_ELO

        # Cap injury Elo swings (cap the Elo, not the raw points)
        inj_total_elo = _clamp(inj_elo_adj + qb_elo_adj, -MAX_ABS_INJ_ELO_ADJ, MAX_ABS_INJ_ELO_ADJ)

        eh_eff = eh + rest_adj + inj_total_elo
        ea_eff = ea

        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))

        # Probability compression
        p_home = 0.5 + BASE_COMPRESS * (p_raw - 0.5)
        p_home = _clamp(p_home, 0.01, 0.99)

        # Spread-ish output (for ATS comparisons)
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = -(elo_diff / ELO_PER_POINT)
        model_spread_home = _clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),

            # Debug columns
            "elo_diff": float(elo_diff),
            "inj_points": float(inj_pts),
            "inj_elo_total": float(inj_total_elo),
            "qb_diff": float(qb_diff),
            "rest_days_home": np.nan if home_days_off is None else float(home_days_off),
            "rest_days_away": np.nan if away_days_off is None else float(away_days_off),

            # Market
            "home_ml": _safe_float((oi or {}).get("home_ml")),
            "away_ml": _safe_float((oi or {}).get("away_ml")),
            "home_spread": _safe_float((oi or {}).get("home_spread")),
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
    return run_daily_nfl(str(date_in), odds_dict=(odds_dict or {}))
