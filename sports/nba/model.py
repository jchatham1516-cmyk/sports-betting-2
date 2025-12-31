# sports/nba/model.py
from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.historical_totals import build_team_historical_total_lines
from sports.nba.injuries import (
    fetch_official_nba_injuries,
    build_injury_list_for_team_nba,
    injury_adjustment_points,
)

from sports.common.prob_calibration import load as load_platt, save as save_platt, fit_platt
from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin

# NEW: use BallDontLie for last-N-days scoring table (fixes constant totals)
from sports.nba.bdl_client import bdl_get, season_start_year_for_date, get_bdl_api_key

ELO_PATH = "results/elo_state_nba.json"
PLATT_PATH = "results/prob_cal_nba.json"
MARGIN_CAL_PATH = "results/margin_cal_nba.json"

# ----------------------------
# Tunables (NBA-specific)
# ----------------------------
HOME_ADV = float(os.getenv("NBA_HOME_ADV", "55.0"))
ELO_K = float(os.getenv("NBA_ELO_K", "20.0"))

ELO_TRAIN_DAYS = int(os.getenv("NBA_ELO_TRAIN_DAYS", "200"))
ELO_PER_POINT = float(os.getenv("NBA_ELO_PER_POINT", "40.0"))

MAX_ABS_INJ_POINTS = float(os.getenv("NBA_MAX_ABS_INJ_POINTS", "6.0"))
INJ_DAMP = float(os.getenv("NBA_INJ_DAMP", "0.60"))

MAX_ABS_MODEL_SPREAD = float(os.getenv("NBA_MAX_ABS_MODEL_SPREAD", "17.0"))

SHORT_REST_PENALTY_ELO = float(os.getenv("NBA_SHORT_REST_PENALTY_ELO", "-14.0"))
NORMAL_REST_BONUS_ELO = float(os.getenv("NBA_NORMAL_REST_BONUS_ELO", "0.0"))

FORM_LOOKBACK_DAYS = int(os.getenv("NBA_FORM_LOOKBACK_DAYS", "35"))
FORM_MIN_GAMES = int(os.getenv("NBA_FORM_MIN_GAMES", "2"))
FORM_ELO_PER_POINT = float(os.getenv("NBA_FORM_ELO_PER_POINT", "1.35"))
FORM_ELO_PER_NET = float(os.getenv("NBA_FORM_ELO_PER_NET", "1.35"))
FORM_ELO_CLAMP = float(os.getenv("NBA_FORM_ELO_CLAMP", "40.0"))

BASE_COMPRESS = float(os.getenv("NBA_BASE_COMPRESS", "0.95"))
MIN_ML_EDGE = float(os.getenv("NBA_MIN_ML_EDGE", "0.02"))

# Missing-Elo handling (do NOT copy market; just soften extremes)
MISSING_ELO_SHRINK = float(os.getenv("NBA_MISSING_ELO_SHRINK", "0.35"))
MISSING_ELO_MARKET_BLEND = float(os.getenv("NBA_MISSING_ELO_MARKET_BLEND", "0.15"))

CAL_MIN_GAMES = int(os.getenv("NBA_CAL_MIN_GAMES", "80"))

# ATS model
ATS_SD_PTS = float(os.getenv("NBA_ATS_SD_PTS", "13.5"))
ATS_DEFAULT_PRICE = float(os.getenv("NBA_ATS_DEFAULT_PRICE", "-110.0"))
ATS_MIN_EDGE_VS_BE = float(os.getenv("NBA_ATS_MIN_EDGE_VS_BE", "0.02"))
ATS_MIN_PTS_EDGE = float(os.getenv("NBA_ATS_MIN_PTS_EDGE", "1.5"))
ATS_BIG_LINE = float(os.getenv("NBA_ATS_BIG_LINE", "7.0"))
ATS_TINY_MODEL = float(os.getenv("NBA_ATS_TINY_MODEL", "2.0"))
ATS_BIGLINE_FORCE_PASS = os.getenv("NBA_ATS_BIGLINE_FORCE_PASS", "1") == "1"
MAX_ATS_PLAYS_PER_DAY = int(os.getenv("NBA_MAX_ATS_PLAYS_PER_DAY", "3"))

# Totals model (hybrid of recent scoring + historical MARKET totals lines)
TOTAL_DEFAULT_PRICE = float(os.getenv("NBA_TOTAL_DEFAULT_PRICE", "-110.0"))
TOTAL_HIST_DAYS = int(os.getenv("NBA_TOTAL_HIST_DAYS", "14"))
TOTAL_REGRESS_WEIGHT = float(os.getenv("NBA_TOTAL_REGRESS_WEIGHT", "0.45"))
TOTAL_SD_FLOOR = float(os.getenv("NBA_TOTAL_SD_FLOOR", "9.0"))
TOTAL_SD_CEIL = float(os.getenv("NBA_TOTAL_SD_CEIL", "20.0"))
TOTAL_MIN_EDGE_VS_BE = float(os.getenv("NBA_TOTAL_MIN_EDGE_VS_BE", "0.015"))
TOTAL_MIN_PTS_EDGE = float(os.getenv("NBA_TOTAL_MIN_PTS_EDGE", "2.5"))
TOTAL_USE_MARKET_FALLBACK = os.getenv("NBA_TOTAL_USE_MARKET_FALLBACK", "0") == "1"
TOTAL_LINE_BLEND = float(os.getenv("NBA_TOTAL_LINE_BLEND", "0.35"))  # weight on historical lines vs scoring model

# Recent scoring model
PTS_LOOKBACK_DAYS = int(os.getenv("NBA_PTS_LOOKBACK_DAYS", "60"))  # you wanted 60
PTS_MIN_GAMES = int(os.getenv("NBA_PTS_MIN_GAMES", "3"))
PTS_REGRESS = float(os.getenv("NBA_PTS_REGRESS", "0.30"))
PTS_LEAGUE_CLAMP_MIN = float(os.getenv("NBA_PTS_LEAGUE_CLAMP_MIN", "104.0"))
PTS_LEAGUE_CLAMP_MAX = float(os.getenv("NBA_PTS_LEAGUE_CLAMP_MAX", "122.0"))

STRICT_SANITY = os.getenv("NBA_STRICT_SANITY", "0") == "1"


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
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
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
    if days_off <= 1:
        return float(SHORT_REST_PENALTY_ELO)
    return float(NORMAL_REST_BONUS_ELO)


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


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _breakeven_prob_from_american(price: float) -> float:
    try:
        price = float(price)
        if price == 0:
            return float("nan")
        if price < 0:
            return (-price) / ((-price) + 100.0)
        return 100.0 / (price + 100.0)
    except Exception:
        return float("nan")


def _ml_recommendation(model_p: float, market_p: float, min_edge: float = MIN_ML_EDGE) -> str:
    if np.isnan(model_p) or np.isnan(market_p):
        return "No ML bet (missing market prob)"
    edge = model_p - market_p
    if edge >= min_edge:
        return "Model PICK: HOME ML (strong)" if edge >= 0.06 else "Model lean: HOME ML"
    if edge <= -min_edge:
        return "Model PICK: AWAY ML (strong)" if edge <= -0.06 else "Model lean: AWAY ML"
    return "No ML bet (edge too small)"


def _cover_prob_from_edge(spread_edge_pts: float, sd_pts: float) -> float:
    if spread_edge_pts is None or np.isnan(spread_edge_pts):
        return float("nan")
    z = float(spread_edge_pts) / float(sd_pts)
    return float(_clamp(_phi(z), 0.001, 0.999))


def _ats_pick_and_edge(p_home_cover: float, spread_price: float) -> Tuple[str, float, float, float]:
    be = _breakeven_prob_from_american(spread_price)
    if np.isnan(p_home_cover) or np.isnan(be):
        return ("NONE", float("nan"), float("nan"), float("nan"))

    p_away_cover = 1.0 - p_home_cover
    if p_home_cover >= p_away_cover:
        side = "HOME"
        p_win = p_home_cover
    else:
        side = "AWAY"
        p_win = p_away_cover

    edge = float(p_win - be)
    return (side, float(p_win), float(edge), float(be))


def _ats_strength_label(edge_vs_be: float) -> str:
    if np.isnan(edge_vs_be):
        return "UNKNOWN"
    if edge_vs_be >= 0.06:
        return "strong"
    if edge_vs_be >= 0.03:
        return "medium"
    if edge_vs_be >= 0.015:
        return "lean"
    return "too_close"


def _ats_reco(side: str, strength: str) -> str:
    if side == "NONE" or strength == "UNKNOWN":
        return "No ATS bet (missing spread/price)"
    if strength == "too_close":
        return "Too close to call ATS (edge too small)"
    return f"Model PICK ATS: {side} ({strength})"


def _total_pick_and_edge(model_total: float, market_total: float, over_price: float, under_price: float, sd: float):
    if np.isnan(model_total) or np.isnan(market_total) or np.isnan(sd) or sd <= 0:
        return ("NONE", float("nan"), float("nan"), float("nan"), float("nan"))

    z = (float(model_total) - float(market_total)) / float(sd)
    p_over = float(_clamp(_phi(z), 0.001, 0.999))
    p_under = 1.0 - p_over

    be_over = _breakeven_prob_from_american(over_price)
    be_under = _breakeven_prob_from_american(under_price)

    edge_over = (p_over - be_over) if not np.isnan(be_over) else float("nan")
    edge_under = (p_under - be_under) if not np.isnan(be_under) else float("nan")

    if np.isnan(edge_over) or np.isnan(edge_under):
        return ("NONE", float("nan"), float("nan"), float("nan"), float(model_total - market_total))

    if edge_over >= edge_under:
        return ("OVER", p_over, be_over, edge_over, float(model_total - market_total))
    return ("UNDER", p_under, be_under, edge_under, float(model_total - market_total))


def _total_reco(side: str, edge_vs_be: float, edge_points: float) -> str:
    if side == "NONE":
        return "No total bet (missing total/model)"
    if np.isnan(edge_vs_be) or np.isnan(edge_points):
        return "No total bet (missing price/model)"
    if abs(edge_points) < TOTAL_MIN_PTS_EDGE:
        return "No total bet (edge too small)"
    if edge_vs_be < TOTAL_MIN_EDGE_VS_BE:
        return f"No total bet (edge_vs_be<{TOTAL_MIN_EDGE_VS_BE:.3f})"
    return f"Model PICK TOTAL: {side}"


# -------------------------------------------------------------------
# FIX: Build team scoring table from BallDontLie (last N days)
# This is the key fix for "all totals are the same".
# -------------------------------------------------------------------
def _build_team_scoring_table(days_back: int, as_of_date: date) -> pd.DataFrame:
    """
    Returns columns: team, pts_for, pts_against (two rows per game: one for each team).
    Uses BallDontLie results so we can actually look back 45/60 days.
    """
    try:
        api_key = get_bdl_api_key()
    except Exception:
        # If missing BDL key, return empty -> totals may fall back
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

            # Skip unplayed / not final
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


# -------------------------------------------------------------------
# FIX: indentation + actually compute team-based expected points
# -------------------------------------------------------------------
def _expected_points_total(home: str, away: str, league_pts: float, team_tbl: pd.DataFrame) -> Tuple[float, float, float]:
    if team_tbl is None or team_tbl.empty or league_pts <= 1e-6:
        return (league_pts, league_pts, 2.0 * league_pts)

    def _team_means(t: str) -> Tuple[Optional[float], Optional[float]]:
        if team_tbl is None or team_tbl.empty:
            return (None, None)

        t0 = (t or "").strip()
        t1 = canon_team(t0) or t0

        sub = team_tbl[(team_tbl["team"] == t0) | (team_tbl["team"] == t1)]
        if sub.empty:
            sub = team_tbl[team_tbl["team"].astype(str).str.lower() == t1.lower()]

        if sub.empty or len(sub) < PTS_MIN_GAMES:
            return (None, None)

        return (float(sub["pts_for"].mean()), float(sub["pts_against"].mean()))

    hf, ha = _team_means(home)
    af, aa = _team_means(away)

    def _strength(x: Optional[float]) -> float:
        if x is None or np.isnan(x):
            return 1.0
        raw = float(x) / float(league_pts)
        return float((1.0 - PTS_REGRESS) * raw + PTS_REGRESS * 1.0)

    home_off = _strength(hf)
    home_def = _strength(ha)
    away_off = _strength(af)
    away_def = _strength(aa)

    exp_home = float(league_pts * home_off * away_def)
    exp_away = float(league_pts * away_off * home_def)
    return (exp_home, exp_away, float(exp_home + exp_away))


def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    """
    NOTE: OddsAPI scores endpoint clamps days_from to 1..3 in your codebase.
    This function is only used for rest/fatigue, so that's fine.
    """
    from sports.common.scores_sources import fetch_recent_scores  # keep local to avoid confusion
    sport_key = SPORT_TO_ODDS_KEY["nba"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=int(min(3, max(1, days_back)))) or []

    last_played: Dict[str, date] = {}
    for ev in events:
        home_raw = ev.get("home")
        away_raw = ev.get("away")
        if not home_raw or not away_raw:
            continue

        home = canon_team(home_raw) or str(home_raw)
        away = canon_team(away_raw) or str(away_raw)

        d = _parse_iso_date(ev.get("commence_time") or "")
        if d is None:
            continue

        if (home not in last_played) or (d > last_played[home]):
            last_played[home] = d
        if (away not in last_played) or (d > last_played[away]):
            last_played[away] = d

    return last_played


def _recent_form_adjustments(days_back: int = FORM_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    try:
        st = update_elo_from_recent_scores(days_from=max(1, min(3, int(days_back))))
    except Exception:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    try:
        items = getattr(st, "ratings", {})
        for team, elo in items.items():
            mult = 1.0 + (float(elo) - 1500.0) / 15000.0
            out[str(team)] = {"elo_adj": float((float(elo) - 1500.0) / 50.0), "off": float(mult), "def": float(mult)}
    except Exception:
        return {}

    return out


def update_elo_from_recent_scores(days_from: int = 10) -> EloState:
    from sports.common.scores_sources import fetch_recent_scores  # existing function (clamped 1..3)
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nba"]

    train_days = int(days_from) if days_from is not None else int(ELO_TRAIN_DAYS)
    train_days = int(max(7, train_days))

    # NOTE: this still clamps to 3; leaving as-is to keep changes minimal
    events = fetch_recent_scores(sport_key=sport_key, days_from=train_days)

    train_ps: list[float] = []
    train_ys: list[float] = []
    train_xs: list[float] = []
    train_margins: list[float] = []

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
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))
        except Exception:
            continue

        eh = st.get(home)
        ea = st.get(away)

        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        p_comp = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))
        train_ps.append(p_comp)
        train_ys.append(1.0 if hs > aw else 0.0)

        elo_diff = (float(eh) + float(HOME_ADV)) - float(ea)
        train_xs.append(float(elo_diff))
        train_margins.append(float(hs - aw))

        nh, na = elo_update(eh, ea, hs, aw, k=ELO_K, home_adv=HOME_ADV)
        st.set(home, nh)
        st.set(away, na)
        st.mark_processed(game_key)

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)

    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            cal = fit_platt(np.array(train_ps, dtype=float), np.array(train_ys, dtype=float))
            save_platt(PLATT_PATH, cal)

            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nba calibration] WARNING: calibration fit failed: {e}")

    return st


def load_nba_calibrator():
    try:
        return load_margin_cal(MARGIN_CAL_PATH)
    except Exception:
        return None


def run_daily_nba(game_date_str: str, *, odds_dict: dict, stats_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=ELO_TRAIN_DAYS)
    platt = load_platt(PLATT_PATH)
    margin_cal = load_nba_calibrator()

    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = datetime.utcnow().date()

    try:
        injuries_map = fetch_official_nba_injuries()
    except Exception as e:
        print(f"[nba injuries] WARNING: failed to load injuries: {e}")
        injuries_map = {}

    last_played = _build_last_game_date_map(days_back=21)
    form_map = _recent_form_adjustments(days_back=FORM_LOOKBACK_DAYS)

    # Historical MARKET totals lines
    sport_key = SPORT_TO_ODDS_KEY.get("nba")
    team_total_lines: Dict[str, Dict[str, float]] = {}
    if sport_key:
        try:
            team_total_lines = build_team_historical_total_lines(
                sport_key=sport_key,
                days_back=TOTAL_HIST_DAYS,
                minutes_before_commence=10,
            )
        except Exception as e:
            print(f"[nba totals] WARNING: failed to build historical totals lines: {e}")
            team_total_lines = {}

    league_avgs: list[float] = []
    league_sds: list[float] = []
    for v in (team_total_lines or {}).values():
        try:
            if v.get("avg") is not None:
                league_avgs.append(float(v.get("avg")))
            if v.get("sd") is not None and not np.isnan(float(v.get("sd"))):
                league_sds.append(float(v.get("sd")))
        except Exception:
            continue

    league_avg_total = float(np.mean(league_avgs)) if league_avgs else float("nan")
    league_sd_total = float(np.mean(league_sds)) if league_sds else 14.0

    # FIX: scoring table from BallDontLie for last N days (this prevents constant totals)
    team_tbl = _build_team_scoring_table(days_back=PTS_LOOKBACK_DAYS, as_of_date=target_date)

    # league per-team scoring anchor
    league_pts = 110.0
    try:
        if not np.isnan(league_avg_total) and league_avg_total > 25:
            league_pts = float(league_avg_total / 2.0)
        elif team_tbl is not None and not team_tbl.empty:
            league_pts = float(team_tbl["pts_for"].mean())
        league_pts = _clamp(league_pts, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX)
    except Exception:
        league_pts = 110.0

    def _team_line_avg_sd(team_canon: str, team_raw: str) -> Tuple[float, float]:
        for k in [team_canon, team_raw, (team_raw or "").strip(), (team_canon or "").strip()]:
            if not k:
                continue
            v = (team_total_lines or {}).get(k)
            if isinstance(v, dict) and v.get("avg") is not None:
                return (_safe_float(v.get("avg")), _safe_float(v.get("sd"), default=np.nan))
        return (float("nan"), float("nan"))

    def _margin_model_spread_from_elo_diff(elo_diff: float) -> float:
        try:
            if margin_cal is None:
                return float(-(elo_diff / ELO_PER_POINT))
            if abs(getattr(margin_cal, "a", 0.0)) < 1e-9 and abs(getattr(margin_cal, "b", 0.0)) < 1e-9:
                return float(-(elo_diff / ELO_PER_POINT))
            pred_margin = float(margin_cal.predict(float(elo_diff)))
            return float(-pred_margin)
        except Exception:
            return float(-(elo_diff / ELO_PER_POINT))

    rows: list[dict] = []

    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        eh = st.get(home)
        ea = st.get(away)

        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        home_inj = build_injury_list_for_team_nba(home, injuries_map)
        away_inj = build_injury_list_for_team_nba(away, injuries_map)
        inj_pts_raw = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = float(INJ_DAMP) * _clamp(inj_pts_raw, -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = float(form_home - form_away)

        eh_eff = float(eh) + float(rest_adj) + 0.5 * float(inj_pts) + 0.5 * float(form_diff)
        ea_eff = float(ea) - 0.5 * float(inj_pts) - 0.5 * float(form_diff)

        # Market
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)

        # Win prob
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_comp = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))
        try:
            p_home = float(_clamp(float(platt.predict(float(p_comp))), 0.01, 0.99))
        except Exception:
            p_home = p_comp

        # Missing-Elo softening ONLY (no “copy market”)
        home_missing = home not in (getattr(st, "ratings", {}) or {})
        away_missing = away not in (getattr(st, "ratings", {}) or {})
        if home_missing and away_missing:
            neutralized = 0.5 + float(MISSING_ELO_SHRINK) * (p_home - 0.5)
            neutralized = float(_clamp(neutralized, 0.05, 0.95))

            if not np.isnan(mkt_home_p):
                w = float(_clamp(MISSING_ELO_MARKET_BLEND, 0.0, 0.85))
                p_home = float(_clamp((1.0 - w) * neutralized + w * float(mkt_home_p), 0.01, 0.99))
            else:
                p_home = neutralized

        # Spread
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(
            _margin_model_spread_from_elo_diff(float(elo_diff)),
            -MAX_ABS_MODEL_SPREAD,
            MAX_ABS_MODEL_SPREAD
        )

        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=ATS_DEFAULT_PRICE)

        total_points = _safe_float((oi or {}).get("total_points"))
        total_over_price = _safe_float((oi or {}).get("over_price"), default=TOTAL_DEFAULT_PRICE)
        total_under_price = _safe_float((oi or {}).get("under_price"), default=TOTAL_DEFAULT_PRICE)

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")

        ml_pick = _ml_recommendation(float(p_home), float(mkt_home_p), min_edge=MIN_ML_EDGE)

        # ATS
        spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) else float("nan")
        p_home_cover = _cover_prob_from_edge(spread_edge_home, sd_pts=ATS_SD_PTS)
        ats_side, ats_p_win, ats_edge_vs_be, ats_be = _ats_pick_and_edge(p_home_cover, spread_price)

        ats_allowed = True
        ats_pass_reason = ""
        if np.isnan(home_spread) or np.isnan(model_spread_home):
            ats_allowed = False
            ats_pass_reason = "missing spread"
        else:
            if ATS_BIGLINE_FORCE_PASS and abs(home_spread) >= ATS_BIG_LINE and abs(model_spread_home) <= ATS_TINY_MODEL:
                ats_allowed = False
                ats_pass_reason = "big market line but tiny model line"
            if ats_allowed and (np.isnan(ats_edge_vs_be) or ats_edge_vs_be < ATS_MIN_EDGE_VS_BE):
                ats_allowed = False
                ats_pass_reason = f"ats_edge_vs_be<{ATS_MIN_EDGE_VS_BE:.3f}"
            if ats_allowed and (np.isnan(spread_edge_home) or abs(spread_edge_home) < ATS_MIN_PTS_EDGE):
                ats_allowed = False
                ats_pass_reason = f"|spread_edge|<{ATS_MIN_PTS_EDGE:.1f}"

        if not ats_allowed:
            spread_reco = f"No ATS bet (gated): {ats_pass_reason}"
        else:
            ats_strength = _ats_strength_label(ats_edge_vs_be)
            spread_reco = _ats_reco(ats_side, ats_strength)

        # TOTALS
        home_avg, home_sd = _team_line_avg_sd(home, home_in)
        away_avg, away_sd = _team_line_avg_sd(away, away_in)

        exp_home, exp_away, exp_total = _expected_points_total(home, away, league_pts, team_tbl)

        hist_base = float("nan")
        if not np.isnan(home_avg) and not np.isnan(away_avg):
            hist_base = 0.5 * (home_avg + away_avg)
        elif not np.isnan(league_avg_total):
            hist_base = float(league_avg_total)

        base_total = float(exp_total)
        if not np.isnan(hist_base):
            w = _clamp(TOTAL_LINE_BLEND, 0.0, 1.0)
            base_total = float((1.0 - w) * base_total + w * hist_base)

        league_anchor_total = league_avg_total
        if np.isnan(league_anchor_total):
            league_anchor_total = float(2.0 * league_pts) if league_pts > 0 else float("nan")

        if not np.isnan(base_total) and not np.isnan(league_anchor_total):
            model_total = float((1.0 - TOTAL_REGRESS_WEIGHT) * base_total + TOTAL_REGRESS_WEIGHT * league_anchor_total)
        else:
            model_total = float(base_total)

        if np.isnan(model_total) and TOTAL_USE_MARKET_FALLBACK and not np.isnan(total_points):
            model_total = float(total_points)

        sd = float("nan")
        if not np.isnan(home_sd) and not np.isnan(away_sd):
            sd = 0.5 * (home_sd + away_sd)
        elif not np.isnan(home_sd):
            sd = home_sd
        elif not np.isnan(away_sd):
            sd = away_sd
        else:
            sd = league_sd_total

        sd = _clamp(sd, TOTAL_SD_FLOOR, TOTAL_SD_CEIL)

        total_side, total_p_win, total_be, total_edge_vs_be, total_edge_pts = _total_pick_and_edge(
            model_total=float(model_total),
            market_total=float(total_points) if not np.isnan(total_points) else float("nan"),
            over_price=float(total_over_price),
            under_price=float(total_under_price),
            sd=float(sd),
        )
        total_recommendation = _total_reco(total_side, total_edge_vs_be, total_edge_pts)

        rows.append(
            {
                "date": game_date_str,
                "home": home,
                "away": away,
                "model_home_prob": float(p_home),
                "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
                "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
                "edge_away": float(edge_away) if not np.isnan(edge_home) else np.nan,
                "elo_diff": float(elo_diff),
                "model_spread_home": float(model_spread_home),
                "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
                "ml_recommendation": ml_pick,
                "spread_recommendation": spread_reco,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_spread": home_spread,
                "spread_price": spread_price,
                "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
                "total_over_price": float(total_over_price),
                "total_under_price": float(total_under_price),
                "model_total": float(model_total) if not np.isnan(model_total) else np.nan,
                "total_edge_points": float(total_edge_pts) if not np.isnan(total_edge_pts) else np.nan,
                "total_pick_side": total_side,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_recommendation": str(total_recommendation),
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
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")
    return run_daily_nba(str(date_in), odds_dict=(odds_dict or {}))
