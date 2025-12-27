# sports/nba/model.py
from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import datetime, date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.historical_totals import build_team_historical_total_lines
from sports.nba.injuries import (
    fetch_official_nba_injuries,
    build_injury_list_for_team_nba,
    injury_adjustment_points,
)

# probability + margin calibrators
from sports.common.prob_calibration import load as load_platt, save as save_platt, fit_platt
from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin

ELO_PATH = "results/elo_state_nba.json"
PLATT_PATH = "results/prob_cal_nba.json"
MARGIN_CAL_PATH = "results/margin_cal_nba.json"

# ----------------------------
# Tunables (NBA-specific)
# ----------------------------
HOME_ADV = float(os.getenv("NBA_HOME_ADV", "55.0"))
ELO_K = float(os.getenv("NBA_ELO_K", "20.0"))

# How far back to train Elo each run (THIS WAS YOUR BIG PROBLEM WHEN CAPPED TO ~21 DAYS)
ELO_TRAIN_DAYS = int(os.getenv("NBA_ELO_TRAIN_DAYS", "200"))

# Fallback only once margin calibrator isn't trained
ELO_PER_POINT = float(os.getenv("NBA_ELO_PER_POINT", "40.0"))

MAX_ABS_INJ_POINTS = float(os.getenv("NBA_MAX_ABS_INJ_POINTS", "6.0"))
MAX_ABS_MODEL_SPREAD = float(os.getenv("NBA_MAX_ABS_MODEL_SPREAD", "17.0"))

SHORT_REST_PENALTY_ELO = float(os.getenv("NBA_SHORT_REST_PENALTY_ELO", "-14.0"))
NORMAL_REST_BONUS_ELO = float(os.getenv("NBA_NORMAL_REST_BONUS_ELO", "0.0"))

FORM_LOOKBACK_DAYS = int(os.getenv("NBA_FORM_LOOKBACK_DAYS", "35"))
FORM_MIN_GAMES = int(os.getenv("NBA_FORM_MIN_GAMES", "2"))
FORM_ELO_PER_POINT = float(os.getenv("NBA_FORM_ELO_PER_POINT", "1.35"))
FORM_ELO_CLAMP = float(os.getenv("NBA_FORM_ELO_CLAMP", "40.0"))

# Less compression (closer to 1.0) => less “everything looks 52/48”
BASE_COMPRESS = float(os.getenv("NBA_BASE_COMPRESS", "0.95"))
MIN_ML_EDGE = float(os.getenv("NBA_MIN_ML_EDGE", "0.02"))

# Calibration minimum games
CAL_MIN_GAMES = int(os.getenv("NBA_CAL_MIN_GAMES", "80"))

# ATS model
ATS_SD_PTS = float(os.getenv("NBA_ATS_SD_PTS", "13.5"))
ATS_DEFAULT_PRICE = float(os.getenv("NBA_ATS_DEFAULT_PRICE", "-110.0"))
ATS_MIN_EDGE_VS_BE = float(os.getenv("NBA_ATS_MIN_EDGE_VS_BE", "0.03"))
ATS_MIN_PTS_EDGE = float(os.getenv("NBA_ATS_MIN_PTS_EDGE", "2.0"))
ATS_BIG_LINE = float(os.getenv("NBA_ATS_BIG_LINE", "7.0"))
ATS_TINY_MODEL = float(os.getenv("NBA_ATS_TINY_MODEL", "2.0"))
ATS_BIGLINE_FORCE_PASS = os.getenv("NBA_ATS_BIGLINE_FORCE_PASS", "1") == "1"
MAX_ATS_PLAYS_PER_DAY = int(os.getenv("NBA_MAX_ATS_PLAYS_PER_DAY", "3"))  # set None manually to disable

# ----------------------------
# Totals model (historical MARKET totals lines)
# ----------------------------
TOTAL_DEFAULT_PRICE = float(os.getenv("NBA_TOTAL_DEFAULT_PRICE", "-110.0"))

TOTAL_HIST_DAYS = int(os.getenv("NBA_TOTAL_HIST_DAYS", "14"))
TOTAL_REGRESS_WEIGHT = float(os.getenv("NBA_TOTAL_REGRESS_WEIGHT", "0.45"))

TOTAL_SD_FLOOR = float(os.getenv("NBA_TOTAL_SD_FLOOR", "9.0"))
TOTAL_SD_CEIL = float(os.getenv("NBA_TOTAL_SD_CEIL", "20.0"))

TOTAL_MIN_EDGE_VS_BE = float(os.getenv("NBA_TOTAL_MIN_EDGE_VS_BE", "0.02"))
TOTAL_MIN_PTS_EDGE = float(os.getenv("NBA_TOTAL_MIN_PTS_EDGE", "3.0"))


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


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ----------------------------
# ATS helpers
# ----------------------------
def _cover_prob_from_edge(spread_edge_pts: float, sd_pts: float = ATS_SD_PTS) -> float:
    if spread_edge_pts is None or np.isnan(spread_edge_pts):
        return float("nan")
    z = float(spread_edge_pts) / float(sd_pts)
    return float(_clamp(_phi(z), 0.001, 0.999))


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


# ----------------------------
# Totals helpers
# ----------------------------
def _total_pick_and_edge(
    model_total: float,
    market_total: float,
    over_price: float,
    under_price: float,
    sd: float,
) -> Tuple[str, float, float, float, float]:
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
        return ("OVER", float(p_over), float(be_over), float(edge_over), float(model_total - market_total))
    return ("UNDER", float(p_under), float(be_under), float(edge_under), float(model_total - market_total))


def _total_gate_reason(side: str, edge_vs_be: float, edge_points: float) -> str:
    if side == "NONE":
        return "missing total/model"
    if np.isnan(edge_vs_be) or np.isnan(edge_points):
        return "missing price/model"
    if abs(edge_points) < TOTAL_MIN_PTS_EDGE:
        return "edge too small"
    if edge_vs_be < TOTAL_MIN_EDGE_VS_BE:
        return f"edge_vs_be<{TOTAL_MIN_EDGE_VS_BE:.3f}"
    return ""


def _total_reco(side: str, edge_vs_be: float, edge_points: float) -> str:
    reason = _total_gate_reason(side, edge_vs_be, edge_points)
    if reason:
        return f"No total bet ({reason})"
    return f"Model PICK TOTAL: {side}"


def _choose_primary_from_fields(
    *,
    ml_reco: str,
    spread_reco: str,
    total_reco: str,
    edge_home: float,
    ats_edge_vs_be: float,
    total_edge_vs_be: float,
    total_edge_points: float,
) -> Tuple[str, str]:
    ml_score = float(abs(edge_home)) if edge_home is not None and not np.isnan(edge_home) else -999.0

    ats_score = -999.0
    if isinstance(spread_reco, str) and spread_reco.startswith("Model PICK ATS:"):
        if ats_edge_vs_be is not None and not np.isnan(ats_edge_vs_be):
            ats_score = float(ats_edge_vs_be)

    tot_score = -999.0
    if isinstance(total_reco, str) and total_reco.startswith("Model PICK TOTAL:"):
        a = float(total_edge_vs_be) if total_edge_vs_be is not None and not np.isnan(total_edge_vs_be) else -999.0
        b = (
            float(abs(total_edge_points) / 10.0)
            if total_edge_points is not None and not np.isnan(total_edge_points)
            else -999.0
        )
        tot_score = float(max(a, b))

    primary = str(ml_reco)
    why = f"Primary=ML (score={ml_score:+.3f})"
    best = ml_score

    if ats_score > best:
        best = ats_score
        primary = str(spread_reco)
        why = f"Primary=ATS (edge_vs_be={ats_score:+.3f})"

    if tot_score > best:
        best = tot_score
        primary = str(total_reco)
        why = f"Primary=TOTAL (score={tot_score:+.3f}; edge_vs_be={float(total_edge_vs_be):+.3f})"

    return primary, why


# ----------------------------
# Builders
# ----------------------------
def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nba"]
    # DO NOT CAP HARD — we want enough history to find last-played reliably
    events = fetch_recent_scores(sport_key=sport_key, days_from=int(days_back))

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


def _recent_form_adjustments(days_back: int = FORM_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    sport_key = SPORT_TO_ODDS_KEY.get("nba")
    if not sport_key:
        return {}
    try:
        # DO NOT CAP HARD — if you set NBA_FORM_LOOKBACK_DAYS higher, we should use it
        events = fetch_recent_scores(sport_key=sport_key, days_from=int(days_back))
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
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))
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
        out[team] = {"avg_margin": avg_margin, "games": int(games), "elo_adj": float(elo_adj)}
    return out


def update_elo_from_recent_scores(days_from: int = 10) -> EloState:
    """
    Updates Elo ratings from recent completed games.
    ALSO trains:
      - Platt probability calibrator (COMPRESSED Elo prob -> calibrated prob)
      - Margin calibrator (elo_diff -> expected score margin)
    """
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nba"]

    # IMPORTANT: use a real training window (default 200 days), no tiny cap
    train_days = int(days_from) if days_from is not None else int(ELO_TRAIN_DAYS)
    train_days = int(max(7, train_days))

    events = fetch_recent_scores(sport_key=sport_key, days_from=train_days)

    train_ps: list = []
    train_ys: list = []
    train_xs: list = []
    train_margins: list = []

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
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))
        except Exception:
            continue

        eh = st.get(home)
        ea = st.get(away)
# Sanity check: default Elo should never be used silently
if eh == 1500 or ea == 1500:
    raise RuntimeError(f"Default Elo used: {home} vs {away}")

        # ---- collect calibration signal BEFORE updating Elo ----
        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        # compress FIRST, because we also use compressed values at prediction time
        p_comp = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))

        train_ps.append(p_comp)
        train_ys.append(1.0 if hs > aw else 0.0)

        elo_diff = (float(eh) + float(HOME_ADV)) - float(ea)
        train_xs.append(elo_diff)
        train_margins.append(float(hs - aw))

        # ---- update Elo ----
        nh, na = elo_update(eh, ea, hs, aw, k=ELO_K, home_adv=HOME_ADV)
        st.set(home, nh)
        st.set(away, na)
        st.mark_processed(game_key)

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)

    # ---- fit + save calibrators when enough samples ----
    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            cal = fit_platt(np.array(train_ps, dtype=float), np.array(train_ys, dtype=float))
            save_platt(PLATT_PATH, cal)

            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nba calibration] WARNING: calibration fit failed: {e}")

    return st


# ----------------------------
# Main daily run
# ----------------------------
def run_daily_nba(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    # IMPORTANT: train Elo on a real window, not 14 days
    st = update_elo_from_recent_scores(days_from=ELO_TRAIN_DAYS)

    platt = load_platt(PLATT_PATH)
    margin_cal = load_margin_cal(MARGIN_CAL_PATH)

    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # injuries
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

    league_avgs = []
    league_sds = []
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

    def _team_line_avg_sd(team_canon: str, team_raw: str) -> Tuple[float, float]:
        candidates = []
        if team_raw:
            candidates.append(str(team_raw))
            candidates.append(str(team_raw).strip())
        if team_canon:
            candidates.append(str(team_canon))
            candidates.append(str(team_canon).strip())

        for k in candidates:
            v = (team_total_lines or {}).get(k)
            if isinstance(v, dict) and v.get("avg") is not None:
                avg = _safe_float(v.get("avg"))
                sd = _safe_float(v.get("sd"), default=np.nan)
                return (avg, sd)

        return (float("nan"), float("nan"))

    def _margin_model_spread_from_elo_diff(elo_diff: float) -> float:
        try:
            if abs(getattr(margin_cal, "a", 0.0)) < 1e-9 and abs(getattr(margin_cal, "b", 0.0)) < 1e-9:
                return float(-(elo_diff / ELO_PER_POINT))
            pred_margin = float(margin_cal.predict(float(elo_diff)))  # home_score - away_score
            return float(-pred_margin)
        except Exception:
            return float(-(elo_diff / ELO_PER_POINT))

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        eh = st.get(home)
        ea = st.get(away)

        # Rest
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Injuries (+ => home advantage)
        home_inj = build_injury_list_for_team_nba(home, injuries_map)
        away_inj = build_injury_list_for_team_nba(away, injuries_map)
        inj_pts_raw = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = 0.6 * _clamp(inj_pts_raw, -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        # Form
        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = float(form_home - form_away)

        # Effective elos
        eh_eff = float(eh) + float(rest_adj) + 0.5 * float(inj_pts) + 0.5 * float(form_diff)
        ea_eff = float(ea) - 0.5 * float(inj_pts) - 0.5 * float(form_diff)

        # Win prob (raw -> compressed -> calibrated; calibrator trained on p_comp now)
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_comp = _clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99)
        try:
            p_home = _clamp(float(platt.predict(float(p_comp))), 0.01, 0.99)
        except Exception:
            p_home = p_comp

        # Spread
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(
            _margin_model_spread_from_elo_diff(float(elo_diff)),
            -MAX_ABS_MODEL_SPREAD,
            MAX_ABS_MODEL_SPREAD,
        )

        # Market
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=ATS_DEFAULT_PRICE)

        total_points = _safe_float((oi or {}).get("total_points"))
        total_over_price = _safe_float((oi or {}).get("over_price"), default=TOTAL_DEFAULT_PRICE)
        total_under_price = _safe_float((oi or {}).get("under_price"), default=TOTAL_DEFAULT_PRICE)

        # Market no-vig
        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")

        ml_pick = _ml_recommendation(float(p_home), float(mkt_home_p), min_edge=MIN_ML_EDGE)
        value_tier = _pick_value_tier(abs(edge_home)) if not np.isnan(edge_home) else "UNKNOWN"

        # ATS
        spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) else float("nan")
        p_home_cover = _cover_prob_from_edge(spread_edge_home, sd_pts=ATS_SD_PTS)
        ats_side, ats_p_win, ats_edge_vs_be, ats_be = _ats_pick_and_edge(p_home_cover, spread_price)

        ats_pass_reason = ""
        ats_allowed = True
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
            ats_strength = "pass"
            spread_reco = f"No ATS bet (gated): {ats_pass_reason}" if ats_pass_reason else "No ATS bet (gated)"
        else:
            ats_strength = _ats_strength_label(ats_edge_vs_be)
            spread_reco = _ats_reco(ats_side, ats_strength)

        # TOTALS (from historical MARKET total lines)
        home_avg, home_sd = _team_line_avg_sd(home, home_in)
        away_avg, away_sd = _team_line_avg_sd(away, away_in)

        base_total = float("nan")
        if not np.isnan(home_avg) and not np.isnan(away_avg):
            base_total = 0.5 * (home_avg + away_avg)
        elif not np.isnan(league_avg_total):
            base_total = float(league_avg_total)

        if not np.isnan(base_total) and not np.isnan(league_avg_total):
            model_total = float((1.0 - TOTAL_REGRESS_WEIGHT) * base_total + TOTAL_REGRESS_WEIGHT * league_avg_total)
        else:
            model_total = float("nan")

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

        total_pass_reason = _total_gate_reason(total_side, total_edge_vs_be, total_edge_pts)
        total_recommendation = _total_reco(total_side, total_edge_vs_be, total_edge_pts)

        primary_pre, why_pre = _choose_primary_from_fields(
            ml_reco=ml_pick,
            spread_reco=spread_reco,
            total_reco=total_recommendation,
            edge_home=edge_home,
            ats_edge_vs_be=ats_edge_vs_be,
            total_edge_vs_be=total_edge_vs_be,
            total_edge_points=total_edge_pts,
        )

        rows.append(
            {
                "date": game_date_str,
                "home": home,
                "away": away,
                "model_home_prob": float(p_home),
                "model_spread_home": float(model_spread_home),
                "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
                "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
                "edge_away": float(edge_away) if not np.isnan(edge_home) else np.nan,
                "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
                "ats_home_cover_prob": float(p_home_cover) if not np.isnan(p_home_cover) else np.nan,
                "ats_pick_side": ats_side,
                "ats_pick_prob": float(ats_p_win) if not np.isnan(ats_p_win) else np.nan,
                "ats_breakeven_prob": float(ats_be) if not np.isnan(ats_be) else np.nan,
                "ats_edge_vs_be": float(ats_edge_vs_be) if not np.isnan(ats_edge_vs_be) else np.nan,
                "ats_strength": ats_strength,
                "ats_pass_reason": ats_pass_reason,
                "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
                "total_over_price": float(total_over_price),
                "total_under_price": float(total_under_price),
                "model_total": float(model_total) if not np.isnan(model_total) else np.nan,
                "total_edge_points": float(total_edge_pts) if not np.isnan(total_edge_pts) else np.nan,
                "total_pick_side": total_side,
                "total_pick_prob": float(total_p_win) if not np.isnan(total_p_win) else np.nan,
                "total_breakeven_prob": float(total_be) if not np.isnan(total_be) else np.nan,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_pass_reason": str(total_pass_reason),
                "total_recommendation": str(total_recommendation),
                "ml_recommendation": ml_pick,
                "spread_recommendation": spread_reco,
                "primary_recommendation": primary_pre,
                "why_primary": why_pre,
                "value_tier": value_tier,
                "elo_diff": float(elo_diff),
                "inj_points_raw": float(inj_pts_raw),
                "inj_points": float(inj_pts),
                "form_elo_diff": float(form_diff),
                "form_home_elo": float(form_home),
                "form_away_elo": float(form_away),
                "rest_days_home": np.nan if home_days_off is None else float(home_days_off),
                "rest_days_away": np.nan if away_days_off is None else float(away_days_off),
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_spread": home_spread,
                "spread_price": spread_price,
            }
        )
# Sanity check: constant-probability detector
if len(rows) >= 5:
    probs = [round(r["model_home_prob"], 3) for r in rows if not np.isnan(r.get("model_home_prob", np.nan))]
    if len(set(probs)) <= 2:
        raise RuntimeError("Model produced near-constant probabilities — check Elo/team mapping.")

    df = pd.DataFrame(rows)

    # Top-N ATS filter
    try:
        if MAX_ATS_PLAYS_PER_DAY is not None and not df.empty and "ats_edge_vs_be" in df.columns:
            elig = df["spread_recommendation"].astype(str).str.contains("Model PICK ATS:", na=False)
            df["ats_rank_score"] = np.where(elig, df["ats_edge_vs_be"].astype(float), -999.0)
            top_idx = df.sort_values("ats_rank_score", ascending=False).head(int(MAX_ATS_PLAYS_PER_DAY)).index
            keep = set(top_idx.tolist())

            for i in df.index:
                if bool(elig.loc[i]) and i not in keep:
                    df.loc[i, "spread_recommendation"] = "No ATS bet (top-N filter)"
                    df.loc[i, "ats_strength"] = "pass"
                    df.loc[i, "ats_pass_reason"] = "top-N filter"

            df.drop(columns=["ats_rank_score"], inplace=True, errors="ignore")

            # recompute PRIMARY after ATS filter
            for i in df.index:
                ml_pick = str(df.loc[i, "ml_recommendation"])
                spread_reco = str(df.loc[i, "spread_recommendation"])
                total_reco = str(df.loc[i, "total_recommendation"])

                edge_home = float(df.loc[i, "edge_home"]) if not pd.isna(df.loc[i, "edge_home"]) else float("nan")
                ats_edge_vs_be = float(df.loc[i, "ats_edge_vs_be"]) if not pd.isna(df.loc[i, "ats_edge_vs_be"]) else float("nan")
                total_edge_vs_be = float(df.loc[i, "total_edge_vs_be"]) if not pd.isna(df.loc[i, "total_edge_vs_be"]) else float("nan")
                total_edge_pts = float(df.loc[i, "total_edge_points"]) if not pd.isna(df.loc[i, "total_edge_points"]) else float("nan")

                primary, why = _choose_primary_from_fields(
                    ml_reco=ml_pick,
                    spread_reco=spread_reco,
                    total_reco=total_reco,
                    edge_home=edge_home,
                    ats_edge_vs_be=ats_edge_vs_be,
                    total_edge_vs_be=total_edge_vs_be,
                    total_edge_points=total_edge_pts,
                )
                df.loc[i, "primary_recommendation"] = primary
                df.loc[i, "why_primary"] = why
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
