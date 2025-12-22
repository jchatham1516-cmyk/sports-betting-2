# sports/nba/model.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.calibration import load_nba_calibrator, update_and_save_nba_calibration

ELO_PATH = "results/elo_state_nba.json"

HOME_ADV = 65.0
ELO_K = 20.0
MAX_ABS_MODEL_SPREAD = 15.0
INJ_ELO_PER_POINT = 18.0
MIN_ML_EDGE = 0.02
BASE_COMPRESS = 0.85

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(np.nan)

def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default

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
        return "HIGH CONFIDENCE"
    if abs_edge >= 0.04:
        return "MEDIUM CONFIDENCE"
    if abs_edge >= 0.02:
        return "LOW CONFIDENCE"
    return "NO EDGE"

def _ml_recommendation(model_p: float, market_p: float, min_edge: float = MIN_ML_EDGE) -> str:
    if np.isnan(model_p) or np.isnan(market_p):
        return "No ML bet (missing market prob)"
    edge = model_p - market_p
    if edge >= min_edge:
        return "Model PICK: HOME (strong)" if edge >= 0.06 else "Model lean: HOME"
    if edge <= -min_edge:
        return "Model PICK: AWAY (strong)" if edge <= -0.06 else "Model lean: AWAY"
    return "No ML bet (edge too small)"

def _parse_game_datetime(commence_time: Optional[str]) -> Optional[datetime]:
    if not commence_time:
        return None
    try:
        s = str(commence_time).strip()
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def update_elo_from_recent_scores(days_from: int = 14) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY.get("nba")
    if not sport_key:
        return st
    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 21))
    except Exception:
        return st
    for ev in events:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        scores = ev.get("scores")
        if not home_raw or not away_raw or not scores:
            continue
        home = canon_team(home_raw)
        away = canon_team(away_raw)
        if not home or not away:
            continue
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

def run_daily_nba(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=10)
    try:
        calibrator = load_nba_calibrator()
    except Exception:
        calibrator = None

    rows: List[dict] = []

    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        eh = st.get(home)
        ea = st.get(away)

        inj_home_pts = _safe_float((oi or {}).get("inj_points_home"), default=0.0)
        inj_away_pts = _safe_float((oi or {}).get("inj_points_away"), default=0.0)
        inj_diff_pts = float(inj_away_pts - inj_home_pts)
        inj_elo = float(inj_diff_pts) * float(INJ_ELO_PER_POINT)

        eh_eff = float(eh) + 0.5 * inj_elo
        ea_eff = float(ea) - 0.5 * inj_elo

        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_home = _clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99)
        elo_diff = (eh_eff - ea_eff) + HOME_ADV

        if calibrator is not None:
            try:
                model_spread_home = float(calibrator.elo_diff_to_spread(float(elo_diff)))
            except Exception:
                model_spread_home = float(-(float(elo_diff) / 25.0))
        else:
            model_spread_home = float(-(float(elo_diff) / 25.0))
        model_spread_home = _clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))
        total_ou = _safe_float((oi or {}).get("total"))      # New: total points line
        ou_over_ml = _safe_float((oi or {}).get("over_ml"))  # New: over ML
        ou_under_ml = _safe_float((oi or {}).get("under_ml"))# New: under ML

        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")
        ml_pick = _ml_recommendation(float(p_home), float(mkt_home_p), min_edge=MIN_ML_EDGE)
        value_tier = _pick_value_tier(abs(edge_home)) if not np.isnan(edge_home) else "UNKNOWN"

        spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) else float("nan")
        if np.isnan(spread_edge_home):
            spread_reco = "No ATS bet (missing spread)"
        else:
            if spread_edge_home >= 2.0:
                spread_reco = "Model lean ATS: HOME"
            elif spread_edge_home <= -2.0:
                spread_reco = "Model lean ATS: AWAY"
            else:
                spread_reco = "Too close to call ATS (edge too small)"

        # --- NEW: totals recommendation ---
        total_edge = float((oi or {}).get("model_total") - total_ou) if total_ou else float("nan")
        if np.isnan(total_edge):
            total_reco = "No total bet"
        elif total_edge > 2.0:
            total_reco = "OVER"
        elif total_edge < -2.0:
            total_reco = "UNDER"
        else:
            total_reco = "Too close to call OU"

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
            "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
            "edge_away": float(edge_away) if not np.isnan(edge_home) else np.nan,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
            "ml_recommendation": ml_pick,
            "spread_recommendation": spread_reco,
            "total_recommendation": total_reco,
            "primary_recommendation": (
                spread_reco if ("Model" in spread_reco and "ATS" in spread_reco) else ml_pick
            ),
            "abs_edge_home": float(abs(edge_home)) if not np.isnan(edge_home) else np.nan,
            "value_tier": value_tier,
            "elo_diff": float(elo_diff),
            "inj_points_home": float(inj_home_pts) if not np.isnan(inj_home_pts) else 0.0,
            "inj_points_away": float(inj_away_pts) if not np.isnan(inj_away_pts) else 0.0,
            "inj_elo": float(inj_elo),
            "total_line": total_ou,
            "over_ml": ou_over_ml,
            "under_ml": ou_under_ml,
        })

    df = pd.DataFrame(rows)
    try:
        if not df.empty:
            update_and_save_nba_calibration(df)
    except Exception:
        pass

    return df

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
