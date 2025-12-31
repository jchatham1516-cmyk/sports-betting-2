# sports/nba/model.py
from __future__ import annotations

import math
import os
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.scores_sources import fetch_recent_scores
from sports.common.historical_totals import build_team_historical_total_lines
from sports.nba.injuries import (
    fetch_official_nba_injuries,
    build_injury_list_for_team_nba,
    injury_adjustment_points,
)

from sports.common.prob_calibration import load as load_platt, save as save_platt, fit_platt
from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin

# BallDontLie client (already in your repo)
from sports.nba.bdl_client import bdl_get, season_start_year_for_date, get_bdl_api_key

ELO_PATH = "results/elo_state_nba.json"
PLATT_PATH = "results/prob_cal_nba.json"
MARGIN_CAL_PATH = "results/margin_cal_nba.json"

# ----------------------------
# Tunables
# ----------------------------
HOME_ADV = float(os.getenv("NBA_HOME_ADV", "35.0"))
ELO_K = float(os.getenv("NBA_ELO_K", "20.0"))

ELO_TRAIN_DAYS = int(os.getenv("NBA_ELO_TRAIN_DAYS", "200"))
CAL_MIN_GAMES = int(os.getenv("NBA_CAL_MIN_GAMES", "80"))

# injuries -> margin adjust
MAX_ABS_INJ_POINTS = float(os.getenv("NBA_MAX_ABS_INJ_POINTS", "6.0"))
INJ_DAMP = float(os.getenv("NBA_INJ_DAMP", "0.60"))
ELO_PER_POINT = float(os.getenv("NBA_ELO_PER_POINT", "40.0"))

# spread sanity
MAX_ABS_MODEL_SPREAD = float(os.getenv("NBA_MAX_ABS_MODEL_SPREAD", "17.0"))

# basic regularization
BASE_COMPRESS = float(os.getenv("NBA_BASE_COMPRESS", "0.95"))
MIN_ML_EDGE = float(os.getenv("NBA_MIN_ML_EDGE", "0.02"))

# totals model
PTS_LOOKBACK_DAYS = int(os.getenv("NBA_PTS_LOOKBACK_DAYS", "60"))
PTS_MIN_GAMES = int(os.getenv("NBA_PTS_MIN_GAMES", "3"))
PTS_REGRESS = float(os.getenv("NBA_PTS_REGRESS", "0.30"))
PTS_LEAGUE_CLAMP_MIN = float(os.getenv("NBA_PTS_LEAGUE_CLAMP_MIN", "104.0"))
PTS_LEAGUE_CLAMP_MAX = float(os.getenv("NBA_PTS_LEAGUE_CLAMP_MAX", "122.0"))

TOTAL_HIST_DAYS = int(os.getenv("NBA_TOTAL_HIST_DAYS", "14"))
TOTAL_LINE_BLEND = float(os.getenv("NBA_TOTAL_LINE_BLEND", "0.35"))
TOTAL_REGRESS_WEIGHT = float(os.getenv("NBA_TOTAL_REGRESS_WEIGHT", "0.45"))
TOTAL_SD_FLOOR = float(os.getenv("NBA_TOTAL_SD_FLOOR", "9.0"))
TOTAL_SD_CEIL = float(os.getenv("NBA_TOTAL_SD_CEIL", "20.0"))
TOTAL_MIN_EDGE_VS_BE = float(os.getenv("NBA_TOTAL_MIN_EDGE_VS_BE", "0.015"))
TOTAL_MIN_PTS_EDGE = float(os.getenv("NBA_TOTAL_MIN_PTS_EDGE", "2.5"))
TOTAL_DEFAULT_PRICE = float(os.getenv("NBA_TOTAL_DEFAULT_PRICE", "-110.0"))

# ATS (kept simple)
ATS_SD_PTS = float(os.getenv("NBA_ATS_SD_PTS", "13.5"))
ATS_DEFAULT_PRICE = float(os.getenv("NBA_ATS_DEFAULT_PRICE", "-110.0"))

# blowout-aware margin distribution (mixture normal)
MARGIN_SD_BASE = float(os.getenv("NBA_MARGIN_SD_BASE", str(ATS_SD_PTS)))
BLOWOUT_MIX_W = float(os.getenv("NBA_BLOWOUT_MIX_W", "0.18"))
BLOWOUT_SD_MULT = float(os.getenv("NBA_BLOWOUT_SD_MULT", "1.75"))


# ----------------------------
# Small utilities
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


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _normal_ci(mu: float, sd: float, z: float = 1.96) -> Tuple[float, float]:
    if np.isnan(mu) or np.isnan(sd) or sd <= 0:
        return (float("nan"), float("nan"))
    return (float(mu - z * sd), float(mu + z * sd))


def _american_to_prob(price: float) -> float:
    price = float(price)
    if price == 0:
        return float("nan")
    if price > 0:
        return 100.0 / (price + 100.0)
    return (-price) / ((-price) + 100.0)


def _no_vig_probs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    hp = _american_to_prob(home_ml)
    ap = _american_to_prob(away_ml)
    if np.isnan(hp) or np.isnan(ap) or (hp + ap) <= 0:
        return (float("nan"), float("nan"))
    s = hp + ap
    return (hp / s, ap / s)


def _breakeven_prob_from_american(price: float) -> float:
    try:
        price = float(price)
    except Exception:
        return float("nan")
    if price == 0:
        return float("nan")
    if price > 0:
        return 100.0 / (price + 100.0)
    return (-price) / ((-price) + 100.0)


def _mix_norm_win_prob(mu: float, sd: float) -> float:
    """P(margin > 0) under mixture of normals (blowout-tail)."""
    if sd <= 1e-9 or np.isnan(mu) or np.isnan(sd):
        return float("nan")
    w = float(_clamp(BLOWOUT_MIX_W, 0.0, 0.49))
    k = float(_clamp(BLOWOUT_SD_MULT, 1.0, 3.0))
    z1 = mu / sd
    z2 = mu / (sd * k)
    return float(_clamp((1.0 - w) * _phi(z1) + w * _phi(z2), 0.001, 0.999))


def _mix_norm_tail_prob_abs_ge(thresh: float, mu: float, sd: float) -> float:
    """P(|margin| >= thresh) under same mixture."""
    if sd <= 1e-9 or np.isnan(mu) or np.isnan(sd):
        return float("nan")
    w = float(_clamp(BLOWOUT_MIX_W, 0.0, 0.49))
    k = float(_clamp(BLOWOUT_SD_MULT, 1.0, 3.0))

    def tail(sd0: float) -> float:
        z_hi = (thresh - mu) / sd0
        z_lo = (-thresh - mu) / sd0
        return float((1.0 - _phi(z_hi)) + _phi(z_lo))

    return float(_clamp((1.0 - w) * tail(sd) + w * tail(sd * k), 0.0, 1.0))


def _pace_proxy_from_total(exp_total: float, league_total: float) -> float:
    """Objective pace proxy derived from scoring environment only."""
    try:
        if np.isnan(exp_total) or np.isnan(league_total) or league_total <= 1e-6:
            return 1.0
        return float(_clamp(exp_total / league_total, 0.85, 1.20))
    except Exception:
        return 1.0

def _team_game_total_mean(team: str, team_tbl: pd.DataFrame) -> float:
    if team_tbl is None or team_tbl.empty:
        return float("nan")
    sub = team_tbl[team_tbl["team"] == team]
    if sub.empty:
        return float("nan")
    return float((sub["pts_for"] + sub["pts_against"]).mean())

def _ml_recommendation(p_home: float, mkt_home_p: float, *, min_edge: float = 0.02) -> str:
    if np.isnan(p_home) or np.isnan(mkt_home_p):
        return "NONE"
    edge = p_home - mkt_home_p
    if abs(edge) < float(min_edge):
        return "No ML bet (edge/conf too small)"
    if edge > 0:
        return "Model PICK: HOME ML (strong)"
    return "Model PICK: AWAY ML (strong)"

def _margin_model_spread_from_elo_diff(elo_diff: float) -> float:
    """
    Convert elo_diff -> model_spread_home (negative means home favored).
    Use margin calibrator only if it behaves sensibly; otherwise fallback linear.
    """
    # Linear fallback: positive elo_diff => home stronger => home favored => negative spread
    fallback = float(-float(elo_diff) / 30.0)

    try:
        cal = load_margin_cal(MARGIN_CAL_PATH)
        if cal is None:
            return fallback

        y = float(cal.predict(float(elo_diff)))

        # Reject broken calibrator outputs (this is what’s happening to you now)
        if np.isnan(y):
            return fallback

        # If it always spits near 0 even for big elo_diff, it's useless
        if abs(float(elo_diff)) >= 60 and abs(y) < 0.75:
            return fallback

        # Sanity clamp (protect against crazy predictions)
        if abs(y) > 40:
            return fallback

        return y
    except Exception:
        return fallback
# ----------------------------
# Historical totals (your repo format)
# ----------------------------
def _team_hist_total_stats(team: str, hist: Dict[str, Dict[str, float]]) -> Tuple[float, float, int]:
    t = canon_team(team) or team
    if not hist or t not in hist:
        return (float("nan"), float("nan"), 0)
    d = hist.get(t) or {}
    avg = _safe_float(d.get("avg"))
    sd = _safe_float(d.get("sd"))
    n = int(d.get("n") or 0)
    return (float(avg), float(sd), int(n))
def _lookup_hist(team: str, hist: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not hist:
        return None
    t = canon_team(team) or team
    if t in hist:
        return hist[t]
    # fallback: case-insensitive search
    tl = t.lower()
    for k, v in hist.items():
        if str(k).lower() == tl:
            return v
    return None


# ----------------------------
# BDL scoring table (key totals fix)
# ----------------------------
def _build_team_scoring_table(days_back: int, as_of_date: date) -> pd.DataFrame:
    """
    Columns: team, pts_for, pts_against.
    Two rows per game (one for each team).
    """
    try:
        api_key = get_bdl_api_key()
    except Exception:
        return pd.DataFrame(columns=["team", "pts_for", "pts_against"])

    start_date = (as_of_date - timedelta(days=int(days_back) + 1)).strftime("%Y-%m-%d")
    end_date = as_of_date.strftime("%Y-%m-%d")
    season_year = season_start_year_for_date(as_of_date)

    params = {
        "seasons[]": season_year,
        "start_date": start_date,
        "end_date": end_date,
        "per_page": 100,
    }

    rows = []
    cursor = None

    while True:
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        games_json = bdl_get("games", params=params, api_key=api_key)
        games = (games_json or {}).get("data", []) or []
        meta = (games_json or {}).get("meta", {}) or {}
        cursor = meta.get("next_cursor")

        for g in games:
            home_team = (g or {}).get("home_team") or {}
            away_team = (g or {}).get("visitor_team") or {}

            home_name = home_team.get("full_name")
            away_name = away_team.get("full_name")
            hs = g.get("home_team_score", 0) or 0
            av = g.get("visitor_team_score", 0) or 0

            # skip unplayed / not final-ish
            if hs == 0 and av == 0 and (g.get("period", 0) or 0) == 0:
                continue

            home = canon_team(home_name) or str(home_name or "")
            away = canon_team(away_name) or str(away_name or "")
            if not home or not away:
                continue

            rows.append({"team": home, "pts_for": float(hs), "pts_against": float(av)})
            rows.append({"team": away, "pts_for": float(av), "pts_against": float(hs)})

        if not cursor:
            break

    if not rows:
        return pd.DataFrame(columns=["team", "pts_for", "pts_against"])

    return pd.DataFrame(rows, columns=["team", "pts_for", "pts_against"])


def _expected_points_total(home: str, away: str, league_pts: float, team_tbl: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Objective expected points using recent scoring table.
    Returns (exp_home_pts, exp_away_pts, exp_total).
    """
    if team_tbl is None or team_tbl.empty or np.isnan(league_pts) or league_pts <= 1e-6:
        return (league_pts, league_pts, 2.0 * league_pts)

    def _team_means(team: str) -> Tuple[Optional[float], Optional[float], int]:
        # IMPORTANT: define sub here so it always exists
        sub = team_tbl[team_tbl["team"] == team]
        if sub.empty:
            return (None, None, 0)
        pf = float(sub["pts_for"].mean())
        pa = float(sub["pts_against"].mean())
        n = int(len(sub))
        return (pf, pa, n)

    h_pf, h_pa, _ = _team_means(home)
    a_pf, a_pa, _ = _team_means(away)

    # convert to relative strengths, regressed toward league
    def _rel(v: Optional[float]) -> float:
        if v is None or np.isnan(v):
            return 1.0
        raw = float(v) / float(league_pts)
        return float((1.0 - PTS_REGRESS) * raw + PTS_REGRESS * 1.0)

    home_off = _rel(h_pf)
    home_def = _rel(h_pa)   # higher allowed => weaker defense (we keep it multiplicative)
    away_off = _rel(a_pf)
    away_def = _rel(a_pa)

    exp_home = float(league_pts) * home_off * away_def
    exp_away = float(league_pts) * away_off * home_def

    exp_home = _clamp(exp_home, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX)
    exp_away = _clamp(exp_away, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX)

    return (float(exp_home), float(exp_away), float(exp_home + exp_away))

# ----------------------------
# Elo + calibration
# ----------------------------
def update_elo_from_recent_scores(days_from: int = 10) -> EloState:
    """
    Uses odds-api scores endpoint (via fetch_recent_scores) to update Elo and (optionally) fit calibrators.
    NOTE: fetch_recent_scores in your repo clamps daysFrom to <=3. We keep that behavior for now.
    """
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nba"]

    train_days = int(days_from) if days_from is not None else int(ELO_TRAIN_DAYS)
    train_days = int(max(7, train_days))

    events = fetch_recent_scores(sport_key=sport_key, days_from=train_days)

    train_ps: list[float] = []
    train_ys: list[float] = []
    train_xs: list[float] = []
    train_margins: list[float] = []

    for ev in events:
        try:
            home_raw = ev.get("home_team")
            away_raw = ev.get("away_team")
            scores = ev.get("scores")
            if not home_raw or not away_raw or not scores:
                continue

            home = canon_team(home_raw)
            away = canon_team(away_raw)
            if not home or not away:
                continue

            # avoid double-processing if EloState supports it
            game_key = f"{ev.get('id','')}|{ev.get('commence_time','')}|{home}|{away}"
            if hasattr(st, "is_processed") and st.is_processed(game_key):
                continue

            score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))

            eh = st.get(home)
            ea = st.get(away)

            p_raw = float(elo_win_prob(eh + HOME_ADV, ea))
            p_comp = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))

            train_ps.append(p_comp)
            train_ys.append(1.0 if hs > aw else 0.0)

            elo_diff = (float(eh) + float(HOME_ADV)) - float(ea)
            train_xs.append(float(elo_diff))
            train_margins.append(float(hs - aw))

            nh, na = elo_update(eh, ea, hs, aw, k=ELO_K, home_adv=HOME_ADV)
            st.set(home, nh)
            st.set(away, na)

            if hasattr(st, "mark_processed"):
                st.mark_processed(game_key)

        except Exception:
            continue

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)

    # fit calibrators if enough samples
    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            cal = fit_platt(np.array(train_ps, dtype=float), np.array(train_ys, dtype=float))
            save_platt(PLATT_PATH, cal)

            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nba calibration] WARNING: calibration fit failed: {e}")

    return st


def _load_calibrators():
    platt = None
    margin = None
    try:
        platt = load_platt(PLATT_PATH)
    except Exception:
        platt = None
    try:
        margin = load_margin_cal(MARGIN_CAL_PATH)
    except Exception:
        margin = None
    return platt, margin


# ----------------------------
# Main runner
# ----------------------------
def run_daily_nba(game_date_str: str, *, odds_dict: dict, stats_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Objective-only predictions:
      - win prob + precise edge vs no-vig market
      - expected margin + CI + blowout tail probs
      - totals model + CI (fixes "totals all same" by using per-team scoring table)
    """
    game_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()

    st = update_elo_from_recent_scores(days_from=ELO_TRAIN_DAYS)
    platt, margin_cal = _load_calibrators()

    # Build scoring table (as-of yesterday)
    as_of = game_date - timedelta(days=1)
    team_tbl = _build_team_scoring_table(PTS_LOOKBACK_DAYS, as_of)

    league_pts = float("nan")
    league_avg_total = float("nan")
    league_sd_total = float("nan")
    if team_tbl is not None and not team_tbl.empty:
        league_pts = float(team_tbl["pts_for"].mean())
        league_avg_total = float((team_tbl["pts_for"] + team_tbl["pts_against"]).mean())
        league_sd_total = float((team_tbl["pts_for"] + team_tbl["pts_against"]).std(ddof=0))

    # historical totals lines (your repo signature)
    try:
        hist_lines = build_team_historical_total_lines(
            sport_key="basketball_nba",
            days_back=TOTAL_HIST_DAYS,
            minutes_before_commence=10,
        )
    except Exception as e:
        print(f"[nba totals] WARNING: failed to build historical totals lines: {e}")
        hist_lines = {}

    # injuries
    try:
        injury_data = fetch_official_nba_injuries()
    except Exception:
        injury_data = []

    rows = []

    # IMPORTANT: odds_dict may store teams in the KEY (tuple), not in oi
    for matchup, oi in (odds_dict or {}).items():
        # recover home/away from value dict or fallback to tuple key
        home_raw = (oi or {}).get("home")
        away_raw = (oi or {}).get("away")
        if (home_raw is None or away_raw is None) and isinstance(matchup, (tuple, list)) and len(matchup) == 2:
            home_raw, away_raw = matchup[0], matchup[1]

        home = canon_team(str(home_raw)) if home_raw is not None else None
        away = canon_team(str(away_raw)) if away_raw is not None else None
        if not home or not away:
            continue

        # market ML -> no-vig
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)

        # injuries -> elo shift
        inj_list_home = build_injury_list_for_team_nba(home, injury_data)
        inj_list_away = build_injury_list_for_team_nba(away, injury_data)
        inj_pts_home = _clamp(injury_adjustment_points(inj_list_home), -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS) * INJ_DAMP
        inj_pts_away = _clamp(injury_adjustment_points(inj_list_away), -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS) * INJ_DAMP
        inj_elo_home = float(inj_pts_home) * float(ELO_PER_POINT)
        inj_elo_away = float(inj_pts_away) * float(ELO_PER_POINT)

        eh = float(st.get(home)) + inj_elo_home
        ea = float(st.get(away)) + inj_elo_away

        # base win prob from Elo
        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        p_home = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))

        # platt calibration if available
        if platt is not None:
            try:
                p_home = float(platt.predict_proba(p_home))
                p_home = float(_clamp(p_home, 0.01, 0.99))
            except Exception:
                pass

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")
        ml_reco = _ml_recommendation(p_home, mkt_home_p, min_edge=MIN_ML_EDGE)

        # margin / spread
        elo_diff = (eh + HOME_ADV) - ea
        model_spread_home = _clamp(_margin_model_spread_from_elo_diff(float(elo_diff)), -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # home margin in points
        mu_margin_home = float(-model_spread_home)

        # variance scaled by pace proxy from totals
        exp_home_pts, exp_away_pts, exp_total = _expected_points_total(home, away, league_pts, team_tbl)
        pace_proxy = _pace_proxy_from_total(exp_total, league_avg_total) if not np.isnan(league_avg_total) else 1.0
        margin_sd = float(MARGIN_SD_BASE) * float(_clamp(0.92 + 0.20 * pace_proxy, 0.85, 1.15))
        home_gt = _team_game_total_mean(home, team_tbl)
        away_gt = _team_game_total_mean(away, team_tbl)

        pace_mult = 1.0
        if not np.isnan(home_gt) and not np.isnan(away_gt) and not np.isnan(league_avg_total) and league_avg_total > 1e-6:
            pace_mult = float(_clamp(0.5 * (home_gt + away_gt) / league_avg_total, 0.90, 1.12))

        exp_total = float(exp_total) * pace_mult
        win_prob_home = _mix_norm_win_prob(mu_margin_home, margin_sd)
        blowout_prob_abs15 = _mix_norm_tail_prob_abs_ge(15.0, mu_margin_home, margin_sd)
        blowout_prob_abs25 = _mix_norm_tail_prob_abs_ge(25.0, mu_margin_home, margin_sd)

        margin_ci95_low, margin_ci95_high = _normal_ci(mu_margin_home, margin_sd, z=1.96)
        margin_ci80_low, margin_ci80_high = _normal_ci(mu_margin_home, margin_sd, z=1.2816)

        # totals
        total_points = _safe_float((oi or {}).get("total_points"))
        total_over_price = _safe_float((oi or {}).get("over_price"), default=TOTAL_DEFAULT_PRICE)
        total_under_price = _safe_float((oi or {}).get("under_price"), default=TOTAL_DEFAULT_PRICE)

        h_avg, h_sd, h_n = _team_hist_total_stats(home, hist_lines)
        a_avg, a_sd, a_n = _team_hist_total_stats(away, hist_lines)

        hist_base = float("nan")
        if not np.isnan(h_avg) and not np.isnan(a_avg):
            hist_base = 0.5 * (h_avg + a_avg)
        elif not np.isnan(league_avg_total):
            hist_base = float(league_avg_total)

        base_total = float(exp_total)
        if not np.isnan(hist_base):
            w = _clamp(TOTAL_LINE_BLEND, 0.20, 0.25)
            base_total = float((1.0 - w) * base_total + w * hist_base)

        league_anchor_total = league_avg_total
        if np.isnan(league_anchor_total) and not np.isnan(league_pts):
            league_anchor_total = float(2.0 * league_pts)

        if not np.isnan(base_total) and not np.isnan(league_anchor_total):
            model_total = float((1.0 - TOTAL_REGRESS_WEIGHT) * base_total + TOTAL_REGRESS_WEIGHT * league_anchor_total)
        else:
            model_total = float(base_total)

        # total sd: prefer hist sd, else league sd
        sd = float("nan")
        if not np.isnan(h_sd) and not np.isnan(a_sd):
            sd = 0.5 * (h_sd + a_sd)
        elif not np.isnan(h_sd):
            sd = h_sd
        elif not np.isnan(a_sd):
            sd = a_sd
        else:
            sd = league_sd_total

        sd = _clamp(sd, TOTAL_SD_FLOOR, TOTAL_SD_CEIL)

        total_ci95_low, total_ci95_high = _normal_ci(model_total, sd, z=1.96)
        total_ci80_low, total_ci80_high = _normal_ci(model_total, sd, z=1.2816)

        total_pick_side = "NONE"
        total_edge_points = float("nan")
        total_edge_vs_be = float("nan")
        total_reco = "No total bet (missing total/model)"

        if not np.isnan(model_total) and not np.isnan(total_points) and sd > 0:
            # P(over) from Normal(model_total, sd)
            z = (model_total - total_points) / sd
            p_over = float(_clamp(_phi(z), 0.001, 0.999))
            p_under = 1.0 - p_over

            be_over = _breakeven_prob_from_american(total_over_price)
            be_under = _breakeven_prob_from_american(total_under_price)
            if np.isnan(be_over):
                be_over = 0.5238
            if np.isnan(be_under):
                be_under = 0.5238

            edge_over = p_over - be_over
            edge_under = p_under - be_under

            total_edge_points = float(model_total - total_points)
            if edge_over >= edge_under:
                total_pick_side = "OVER"
                total_edge_vs_be = float(edge_over)
            else:
                total_pick_side = "UNDER"
                total_edge_vs_be = float(edge_under)

            if abs(total_edge_points) >= TOTAL_MIN_PTS_EDGE and total_edge_vs_be >= TOTAL_MIN_EDGE_VS_BE:
                total_reco = f"Model PICK TOTAL: {total_pick_side}"
            else:
                total_reco = "No total bet (edge too small)"

        # optional ATS fields if present in odds
        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=ATS_DEFAULT_PRICE)
        spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) and not np.isnan(model_spread_home) else float("nan")
        p_home_cover = float(_clamp(_phi((spread_edge_home / ATS_SD_PTS)), 0.001, 0.999)) if not np.isnan(spread_edge_home) else float("nan")

        rows.append(
            {
                "date": game_date_str,
                "home": home,
                "away": away,
                "model_home_prob": float(p_home),
                "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
                "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
                "edge_away": float(edge_away) if not np.isnan(edge_away) else np.nan,
                "ml_recommendation": str(ml_reco),
                "home_ml": float(home_ml) if not np.isnan(home_ml) else np.nan,
                "away_ml": float(away_ml) if not np.isnan(away_ml) else np.nan,
                "inj_points_home": float(inj_pts_home),
                "inj_points_away": float(inj_pts_away),
                "elo_diff": float(elo_diff),
                "model_spread_home": float(model_spread_home),
                "model_margin_home": float(mu_margin_home),
                "margin_sd": float(margin_sd),
                "margin_ci95_low": float(margin_ci95_low),
                "margin_ci95_high": float(margin_ci95_high),
                "margin_ci80_low": float(margin_ci80_low),
                "margin_ci80_high": float(margin_ci80_high),
                "win_prob_home": float(win_prob_home),
                "blowout_prob_abs15": float(blowout_prob_abs15),
                "blowout_prob_abs25": float(blowout_prob_abs25),
                "home_spread": float(home_spread) if not np.isnan(home_spread) else np.nan,
                "spread_price": float(spread_price) if not np.isnan(spread_price) else np.nan,
                "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
                "p_home_cover": float(p_home_cover) if not np.isnan(p_home_cover) else np.nan,
                "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
                "total_over_price": float(total_over_price),
                "total_under_price": float(total_under_price),
                "model_total": float(model_total),
                "total_sd": float(sd) if not np.isnan(sd) else np.nan,
                "total_ci95_low": float(total_ci95_low),
                "total_ci95_high": float(total_ci95_high),
                "total_ci80_low": float(total_ci80_low),
                "total_ci80_high": float(total_ci80_high),
                "total_edge_points": float(total_edge_points) if not np.isnan(total_edge_points) else np.nan,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_pick_side": str(total_pick_side),
                "total_recommendation": str(total_reco),
                "hist_total_home_avg": float(h_avg) if not np.isnan(h_avg) else np.nan,
                "hist_total_away_avg": float(a_avg) if not np.isnan(a_avg) else np.nan,
                "hist_total_home_n": int(h_n),
                "hist_total_away_n": int(a_n),
            }
        )

    return pd.DataFrame(rows)


def run_daily_probs_for_date(
    game_date_str: str = None,
    *,
    game_date: str = None,
    odds_dict: dict = None,
    spreads_dict: dict = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper used by current_of_sports_betting_algorithm.py
    """
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")
    return run_daily_nba(str(date_in), odds_dict=(odds_dict or {}))
