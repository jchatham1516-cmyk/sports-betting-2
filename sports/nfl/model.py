# sports/nfl/model.py
from __future__ import annotations

import os
from datetime import datetime, date
from typing import Dict, Optional, Tuple

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

# Elo -> points (spread-ish). Higher number = less aggressive spreads.
# If your spreads are too extreme, increase this (e.g., 40.0).
ELO_PER_POINT = 40.0

# Caps for sanity
MAX_ABS_MODEL_SPREAD = 17.0
MAX_ABS_INJ_ELO_ADJ = 80.0  # cap injury Elo swing

# Base injuries -> Elo
INJ_ELO_PER_POINT = 14.0

# Extra QB weighting
QB_EXTRA_ELO = 18.0  # multiplier applied to QB-ish cost differential

# Rest / short-week effects (Elo)
SHORT_REST_PENALTY_ELO = -14.0   # <=4 days off
NORMAL_REST_BONUS_ELO = 0.0
BYE_BONUS_ELO = +8.0             # 10+ days off

# Probability compression: NFL has variance; shrink toward 0.5
BASE_COMPRESS = 0.75

# Betting thresholds
MIN_ML_EDGE = 0.02       # 2% no-vig edge for ML
MIN_ATS_EDGE_PTS = 1.5   # 1.5 points edge for ATS


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


def _calc_days_off(target: Optional[date], last: Optional[date]) -> Optional[int]:
    """
    days_off = days between games (not counting game days).
    Returns None if data looks invalid.
    """
    if target is None or last is None:
        return None
    delta = (target - last).days - 1
    # Ignore bad ranges (future last_game, or crazy long gap)
    if delta < 0 or delta > 30:
        return None
    return int(delta)


def _rest_elo(days_off: Optional[int]) -> float:
    """
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

    Your current NFL injuries mapping uses QB impact ~4.0.
    We treat impact >= 3.7 as QB-like. Also supports a light "QB" hint
    if the player string contains ' qb' or 'quarterback'.
    """
    if not inj_list:
        return 0.0

    role_w = {"starter": 1.0, "rotation": 0.55}
    total = 0.0

    for item in inj_list:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            continue

        player, role, mult, impact = item
        try:
            mult = float(mult)
            impact = float(impact)
        except Exception:
            continue

        player_s = str(player).lower()
        qb_hint = (" qb" in player_s) or ("quarterback" in player_s)

        if impact >= 3.7 or qb_hint:
            total += role_w.get(role, 0.6) * mult * max(impact, 3.7)

    return float(total)


def _american_to_prob(ml: float) -> float:
    """
    Converts American odds to implied probability (with vig).
    """
    ml = float(ml)
    if ml == 0:
        return float("nan")
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)


def _no_vig_probs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    """
    Returns (home_prob_no_vig, away_prob_no_vig).
    If missing/invalid, returns (nan, nan).
    """
    try:
        hp = _american_to_prob(home_ml)
        ap = _american_to_prob(away_ml)
        if np.isnan(hp) or np.isnan(ap):
            return (float("nan"), float("nan"))
        s = hp + ap
        if s <= 0:
            return (float("nan"), float("nan"))
        return (hp / s, ap / s)
    except Exception:
        return (float("nan"), float("nan"))


def _pick_value_tier(abs_edge: float) -> str:
    if np.isnan(abs_edge):
        return "UNKNOWN"
    if abs_edge >= 0.08:
        return "HIGH VALUE"
    if abs_edge >= 0.04:
        return "MED VALUE"
    if abs_edge >= 0.02:
        return "LOW VALUE"
    return "NO EDGE"


def _ml_recommendation(model_p: float, market_p: float, min_edge: float = MIN_ML_EDGE) -> str:
    if np.isnan(model_p) or np.isnan(market_p):
        return "No ML bet (missing market prob)"
    edge = model_p - market_p
    if edge >= min_edge:
        return "Model PICK: HOME ML"
    if edge <= -min_edge:
        return "Model PICK: AWAY ML"
    return "No ML bet (edge too small)"


def _ats_recommendation(spread_edge_home: float, min_pts: float = MIN_ATS_EDGE_PTS) -> str:
    """
    spread_edge_home = (market_home_spread - model_spread_home)
      + means market is giving home MORE points than model thinks -> value HOME
      - means value AWAY
    """
    if np.isnan(spread_edge_home):
        return "No ATS bet (missing spread)"
    if spread_edge_home >= min_pts:
        return "Model PICK ATS: HOME"
    if spread_edge_home <= -min_pts:
        return "Model PICK ATS: AWAY"
    return "No ATS bet (edge too small)"


# ----------------------------
# Data builders
# ----------------------------
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


def update_elo_from_recent_scores(days_from: int = 10) -> EloState:
    """
    Updates Elo state from recent completed games.
    NFL is weekly; we look back up to ~3 weeks max for safety.
    """
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nfl"]

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


# ----------------------------
# Main daily run
# ----------------------------
def run_daily_nfl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    """
    NFL daily model outputs:
      - model_home_prob (compressed Elo prob)
      - market_home_prob (no-vig from ML)
      - edges (model - market)
      - model_spread_home (Elo -> points)
      - spread edge + ATS rec
      - ML rec + value tier
      - debug columns (elo_diff, injuries, qb, rest)
    """
    st = update_elo_from_recent_scores(days_from=14)

    # Parse date
    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # Load injuries once
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
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Injuries (tuples come from your sports/nfl/injuries.py)
        home_inj = build_injury_list_for_team_nfl(home, injuries_map)
        away_inj = build_injury_list_for_team_nfl(away, injuries_map)

        # injury_adjustment_points: + means away more hurt (per your comment)
        inj_pts = float(injury_adjustment_points(home_inj, away_inj))
        inj_elo_adj = inj_pts * INJ_ELO_PER_POINT

        # QB extra weighting (difference in QB injury cost)
        qb_diff = _qb_cost(away_inj) - _qb_cost(home_inj)
        qb_elo_adj = qb_diff * QB_EXTRA_ELO

        # Cap injury Elo swing (cap in Elo space)
        inj_total_elo = _clamp(inj_elo_adj + qb_elo_adj, -MAX_ABS_INJ_ELO_ADJ, MAX_ABS_INJ_ELO_ADJ)

        # Apply injury symmetrically (cleaner than applying only to home)
        eh_eff = eh + rest_adj + 0.5 * inj_total_elo
        ea_eff = ea - 0.5 * inj_total_elo

        # Win prob
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))

        # Probability compression
        p_home = 0.5 + BASE_COMPRESS * (p_raw - 0.5)
        p_home = _clamp(p_home, 0.01, 0.99)

        # Spread-ish output (for ATS comparisons)
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = -(elo_diff / ELO_PER_POINT)
        model_spread_home = _clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # Market odds/spread
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))

        # Market no-vig prob + edges
        mkt_home_p, mkt_away_p = (float("nan"), float("nan"))
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, mkt_away_p = _no_vig_probs(home_ml, away_ml)

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")

        ml_pick = _ml_recommendation(float(p_home), float(mkt_home_p), min_edge=MIN_ML_EDGE)
        value_tier = _pick_value_tier(abs(edge_home)) if not np.isnan(edge_home) else "UNKNOWN"

        # ATS edge/reco
        # spread_edge_home = market - model ; + => value HOME, - => value AWAY
        spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) else float("nan")
        ats_pick = _ats_recommendation(spread_edge_home, min_pts=MIN_ATS_EDGE_PTS)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,

            # Model
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),

            # Market (processed)
            "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
            "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
            "edge_away": float(edge_away) if not np.isnan(edge_away) else np.nan,

            # ATS
            "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
            "ml_recommendation": ml_pick,
            "spread_recommendation": ats_pick,
            "value_tier": value_tier,

            # Debug columns
            "elo_diff": float(elo_diff),
            "inj_points": float(inj_pts),
            "inj_elo_total": float(inj_total_elo),
            "qb_diff": float(qb_diff),
            "rest_days_home": np.nan if home_days_off is None else float(home_days_off),
            "rest_days_away": np.nan if away_days_off is None else float(away_days_off),

            # Raw market fields
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
        })

    return pd.DataFrame(rows)


# Backwards-compatible alias
def run_daily_probs_for_date(
    game_date_str: str = None,
    *,
    game_date: str = None,
    odds_dict: dict = None,
    spreads_dict: dict = None,  # kept for compatibility (unused)
    **kwargs,
) -> pd.DataFrame:
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")
    return run_daily_nfl(str(date_in), odds_dict=(odds_dict or {}))
