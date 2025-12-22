# sports/nfl/model.py
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
ELO_PER_POINT = 40.0
MAX_ABS_INJ_ELO_ADJ = 45.0
MAX_ABS_INJ_POINTS = 6.0
INJ_ELO_PER_POINT = 6.0
QB_EXTRA_ELO = 10.0
MAX_ABS_MODEL_SPREAD = 17.0
SHORT_REST_PENALTY_ELO = -14.0
NORMAL_REST_BONUS_ELO = 0.0
BYE_BONUS_ELO = +8.0
FORM_LOOKBACK_DAYS = 35
FORM_MIN_GAMES = 2
FORM_ELO_PER_POINT = 1.35
FORM_ELO_CLAMP = 40.0
BASE_COMPRESS = 0.75
MIN_ML_EDGE = 0.02
ATS_SD_PTS = 13.5
ATS_DEFAULT_PRICE = -110.0
ATS_MIN_EDGE_VS_BE = 0.03
ATS_MIN_PTS_EDGE = 2.0
ATS_BIG_LINE = 7.0
ATS_TINY_MODEL = 2.0
ATS_BIGLINE_FORCE_PASS = True
MAX_ATS_PLAYS_PER_DAY = 3

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
    if target is None or last is None:
        return None
    delta = (target - last).days - 1
    if delta < 0 or delta > 30:
        return None
    return int(delta)

def _rest_elo(days_off: Optional[int]) -> float:
    if days_off is None:
        return 0.0
    if days_off <= 4:
        return float(SHORT_REST_PENALTY_ELO)
    if days_off >= 10:
        return float(BYE_BONUS_ELO)
    return float(NORMAL_REST_BONUS_ELO)

def _recent_form_adjustments(days_back: int = FORM_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    sport_key = SPORT_TO_ODDS_KEY.get("nfl")
    if not sport_key:
        return {}

    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 45))
    except Exception:
        return {}

    margins = defaultdict(list)
    for ev in events:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        scores = ev.get("scores")
        if not home_raw or not away_raw or not scores:
            continue

        d = _parse_iso_date(ev.get("commence_time") or "")
        if d is None:
            continue

        home = canon_team(home_raw)
        away = canon_team(away_raw)
        if not home or not away:
            continue

        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = score_map.get(home_raw) or score_map.get(home)
            aw = score_map.get(away_raw) or score_map.get(away)
            hs = float(hs)
            aw = float(aw)
        except Exception:
            continue

        margin = float(hs - aw)
        margins[home].append((d, margin))
        margins[away].append((d, -margin))

    out: Dict[str, Dict[str, float]] = {}
    for team, lst in margins.items():
        lst = sorted(lst, key=lambda x: x[0], reverse=True)
        margins_only = [m for _, m in lst]
        games = len(margins_only)
        if games < FORM_MIN_GAMES:
            continue

        avg_margin = float(np.mean(margins_only))
        elo_adj = _clamp(avg_margin * FORM_ELO_PER_POINT, -FORM_ELO_CLAMP, FORM_ELO_CLAMP)

        out[team] = {
            "avg_margin": float(avg_margin),
            "games": int(games),
            "elo_adj": float(elo_adj),
        }

    return out

def _american_to_prob(ml: float) -> float:
    ml = float(ml)
    if ml == 0:
        return float("nan")
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)

def _no_vig_probs(home_ml: float, away_ml: float) -> Tuple[float, float]:
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
        return "Model PICK: HOME ML (strong)" if edge >= 0.06 else "Model lean: HOME ML"
    if edge <= -min_edge:
        return "Model PICK: AWAY ML (strong)" if edge <= -0.06 else "Model lean: AWAY ML"
    return "No ML bet (edge too small)"

# ... keep all your existing ATS helper functions here ...

# ----------------------------
# Main daily run
# ----------------------------
def run_daily_nfl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=14)
    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    try:
        injuries_map = fetch_espn_nfl_injuries()
    except Exception as e:
        print(f"[nfl injuries] WARNING: failed to load ESPN injuries: {e}")
        injuries_map = {}

    last_played = _build_last_game_date_map(days_back=21)
    form_map = _recent_form_adjustments(days_back=FORM_LOOKBACK_DAYS)

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        eh = st.get(home)
        ea = st.get(away)

        # ----------------------------
        # Rest, injuries, form
        # ----------------------------
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        home_inj = build_injury_list_for_team_nfl(home, injuries_map)
        away_inj = build_injury_list_for_team_nfl(away, injuries_map)
        inj_pts_raw = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = _clamp(inj_pts_raw, -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        def qb_cost(lst) -> float:
            s = 0.0
            for (player, role, mult, impact) in (lst or []):
                try:
                    player_s = str(player).lower()
                    qb_like = ((str(player_s).find(" qb") >= 0) or ("quarterback" in player_s) or (float(impact) >= 6.0))
                    if not qb_like:
                        continue
                    rw = 1.0 if role == "starter" else 0.55
                    s += rw * float(mult) * float(impact)
                except Exception:
                    continue
            return float(s)

        qb_home = qb_cost(home_inj)
        qb_away = qb_cost(away_inj)
        qb_diff = float(qb_away - qb_home)

        inj_elo_adj = float(inj_pts) * float(INJ_ELO_PER_POINT)
        qb_elo_adj = float(qb_diff) * float(QB_EXTRA_ELO)
        inj_total_elo = _clamp(inj_elo_adj + qb_elo_adj, -MAX_ABS_INJ_ELO_ADJ, MAX_ABS_INJ_ELO_ADJ)

        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = float(form_home - form_away)

        eh_eff = eh + rest_adj + 0.5 * inj_total_elo + 0.5 * form_diff
        ea_eff = ea - 0.5 * inj_total_elo - 0.5 * form_diff

        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_home = _clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99)
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(-(elo_diff / ELO_PER_POINT), -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=-110.0)

        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)
        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")

        ml_pick = _ml_recommendation(float(p_home), float(mkt_home_p), min_edge=MIN_ML_EDGE)
        value_tier = _pick_value_tier(abs(edge_home)) if not np.isnan(edge_home) else "UNKNOWN"

        # ----------------------------
        # TOTAL / OU recommendations
        # ----------------------------
        total_ou = _safe_float((oi or {}).get("total"))
        model_total = _safe_float((oi or {}).get("model_total"))
        total_edge = float(model_total - total_ou) if total_ou else float("nan")
        if np.isnan(total_edge):
            total_reco = "No total bet"
        elif total_edge > 2.0:
            total_reco = "OVER"
        elif total_edge < -2.0:
            total_reco = "UNDER"
        else:
            total_reco = "Too close to call OU"

        # ... existing ATS calculations here ...

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
            "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
            "edge_away": float(edge_away) if not np.isnan(edge_home) else np.nan,
            "ml_recommendation": ml_pick,
            "value_tier": value_tier,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
            "total_line": total_ou,
            "model_total": model_total,
            "total_edge": total_edge,
            "total_recommendation": total_reco,
            # ... keep your other debug columns ...
        })

    return pd.DataFrame(rows)

# Backwards-compatible alias
def run_daily_probs_for_date(game_date_str: str = None, *, game_date: str = None, odds_dict: dict = None, spreads_dict: dict = None, **kwargs) -> pd.DataFrame:
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")
    return run_daily_nfl(str(date_in), odds_dict=(odds_dict or {}))
