# sports/nfl/model.py
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

from sports.nfl.injuries import (
    fetch_espn_nfl_injuries,
    build_injury_list_for_team_nfl,
    injury_adjustment_points,
)

from sports.common.prob_calibration import load as load_platt, save as save_platt, fit_platt
from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin

# OPTIONAL weather
from sports.common.weather_sources import fetch_game_weather

ELO_PATH = "results/elo_state_nfl.json"
PLATT_PATH = "results/prob_cal_nfl.json"
MARGIN_CAL_PATH = "results/margin_cal_nfl.json"

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

FORM_LOOKBACK_DAYS = 70
FORM_MIN_GAMES = 2
FORM_ELO_PER_POINT = 1.35
FORM_ELO_CLAMP = 40.0

BASE_COMPRESS = 0.75
MIN_ML_EDGE = 0.02
CAL_MIN_GAMES = 60

# ATS
ATS_SD_PTS = 13.5
ATS_DEFAULT_PRICE = -110.0
ATS_MIN_EDGE_VS_BE = 0.03
ATS_MIN_PTS_EDGE = 2.0
ATS_BIG_LINE = 7.0
ATS_TINY_MODEL = 2.0
ATS_BIGLINE_FORCE_PASS = True
MAX_ATS_PLAYS_PER_DAY = 3  # None to disable

# ----------------------------
# Totals
# ----------------------------
TOTAL_DEFAULT_PRICE = -110.0

# Anchor weight: higher = trust your expected-points model more
TOTAL_ANCHOR_W = float(os.getenv("NFL_TOTAL_ANCHOR_W", "0.60"))

# Expected points model settings
PTS_LOOKBACK_DAYS = int(os.getenv("NFL_PTS_LOOKBACK_DAYS", "70"))
PTS_RECENT_WEIGHT = float(os.getenv("NFL_PTS_RECENT_WEIGHT", "0.60"))  # recent vs older mix
PTS_REGRESS = float(os.getenv("NFL_PTS_REGRESS", "0.35"))             # shrink team strengths to league mean
PTS_MIN_GAMES = int(os.getenv("NFL_PTS_MIN_GAMES", "4"))

# Convert injuries/QB/rest into point adjustments
INJ_POINTS_TO_TOTAL_PTS = float(os.getenv("NFL_INJ_POINTS_TO_TOTAL_PTS", "0.45"))
QB_POINTS_PER_QB_IMPACT = float(os.getenv("NFL_QB_POINTS_PER_QB_IMPACT", "0.55"))
REST_POINTS_PER_ELO = float(os.getenv("NFL_REST_POINTS_PER_ELO", "0.02"))  # tiny

# Weather impact
ENABLE_WEATHER = os.getenv("NFL_ENABLE_WEATHER", "1") == "1"
WIND_PTS_PER_MPH_OVER_10 = float(os.getenv("NFL_WIND_PTS_PER_MPH_OVER_10", "0.35"))
COLD_PTS_IF_UNDER_35F = float(os.getenv("NFL_COLD_PTS_IF_UNDER_35F", "1.25"))

# Totals bet gating
TOTAL_MIN_EDGE_VS_BE = 0.02
TOTAL_MIN_PTS_EDGE = 2.0
TOTAL_SD_FLOOR = 5.5
TOTAL_SD_CEIL = 14.5
TOTAL_REGRESS_WEIGHT = 0.40  # historical-market-line baseline shrink (still used for SD baseline)

TOTAL_PRIMARY_BOOST = float(os.getenv("NFL_TOTAL_PRIMARY_BOOST", "1.25"))

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


def _parse_iso_datetime(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _calc_days_off(target: Optional[date], last: Optional[date]) -> Optional[int]:
    if target is None or last is None:
        return None
    delta = (target - last).days - 1
    if delta < 0 or delta > 40:
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


# ---------- ATS helpers ----------
def _cover_prob_from_edge(spread_edge_pts: float, sd_pts: float = ATS_SD_PTS) -> float:
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


# ---------- Totals helpers ----------
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

    edge_pts = float(model_total - market_total)

    if np.isnan(edge_over) or np.isnan(edge_under):
        return ("NONE", float("nan"), float("nan"), float("nan"), float(edge_pts))

    if edge_over >= edge_under:
        return ("OVER", float(p_over), float(be_over), float(edge_over), float(edge_pts))
    return ("UNDER", float(p_under), float(be_under), float(edge_under), float(edge_pts))


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


# ----------------------------
# Scoring-strength model (expected points)
# ----------------------------
def _build_team_scoring_table() -> pd.DataFrame:
    """
    Uses OddsAPI scores endpoint (daysFrom<=3 per call).
    This is often sparse — so totals MUST have a stable fallback (handled below).
    """
    sport_key = SPORT_TO_ODDS_KEY["nfl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=3)

    rows = []
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

        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))
        except Exception:
            continue

        rows.append({"team": home, "opp": away, "pts_for": hs, "pts_against": aw})
        rows.append({"team": away, "opp": home, "pts_for": aw, "pts_against": hs})

    return pd.DataFrame(rows)


def _expected_points_total(
    home: str,
    away: str,
    *,
    league_pts: float,
    team_tbl: pd.DataFrame,
    fallback_total_line: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Expected points:
      - If team_tbl has enough games for BOTH teams: use strengths model.
      - Otherwise: fallback to a reasonable TOTAL LINE baseline (historical market avg),
        split 50/50 across teams.
    """
    # ---- Fallback path (this is the key fix) ----
    if team_tbl is None or team_tbl.empty:
        base_total = fallback_total_line
        if base_total is None or np.isnan(float(base_total)):
            base_total = 2.0 * float(league_pts)
        base_total = float(base_total)
        return (0.5 * base_total, 0.5 * base_total, base_total)

    def _team_means(t: str) -> Tuple[Optional[float], Optional[float], int]:
        sub = team_tbl[team_tbl["team"] == t]
        if sub.empty:
            return (None, None, 0)
        return (float(sub["pts_for"].mean()), float(sub["pts_against"].mean()), int(len(sub)))

    hf, ha, hn = _team_means(home)
    af, aa, an = _team_means(away)

    # If either team is too sparse, fallback to baseline total line
    if hn < PTS_MIN_GAMES or an < PTS_MIN_GAMES:
        base_total = fallback_total_line
        if base_total is None or np.isnan(float(base_total)):
            base_total = 2.0 * float(league_pts)
        base_total = float(base_total)
        return (0.5 * base_total, 0.5 * base_total, base_total)

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

    total = float(exp_home + exp_away)
    return (exp_home, exp_away, total)


# ----------------------------
# Builders
# ----------------------------
def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nfl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=3)

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
    sport_key = SPORT_TO_ODDS_KEY.get("nfl")
    if not sport_key:
        return {}
    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=3)
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
        out[team] = {"avg_margin": float(avg_margin), "games": int(games), "elo_adj": float(elo_adj)}
    return out


def update_elo_from_recent_scores(days_from: int = 10) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nfl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=3)

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

        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        train_ps.append(p_raw)
        train_ys.append(1.0 if hs > aw else 0.0)

        elo_diff = (float(eh) + float(HOME_ADV)) - float(ea)
        train_xs.append(elo_diff)
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
        print(f"[nfl calibration] WARNING: calibration fit failed: {e}")

    return st


# ----------------------------
# Main daily run
# ----------------------------
def run_daily_nfl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=14)
    platt = load_platt(PLATT_PATH)
    margin_cal = load_margin_cal(MARGIN_CAL_PATH)

    def _margin_model_spread_from_elo_diff(elo_diff: float) -> float:
        try:
            if abs(getattr(margin_cal, "a", 0.0)) < 1e-9 and abs(getattr(margin_cal, "b", 0.0)) < 1e-9:
                return float(-(elo_diff / ELO_PER_POINT))
            pred_margin = float(margin_cal.predict(float(elo_diff)))
            return float(-pred_margin)
        except Exception:
            return float(-(elo_diff / ELO_PER_POINT))

    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # injuries
    try:
        injuries_map = fetch_espn_nfl_injuries()
    except Exception as e:
        print(f"[nfl injuries] WARNING: failed to load ESPN injuries: {e}")
        injuries_map = {}

    last_played = _build_last_game_date_map(days_back=21)
    form_map = _recent_form_adjustments(days_back=FORM_LOOKBACK_DAYS)

    # Totals market-line history (for SD baseline + stable baseline total)
    sport_key = SPORT_TO_ODDS_KEY.get("nfl")
    team_total_lines = {}
    if sport_key:
        try:
            team_total_lines = build_team_historical_total_lines(
                sport_key=sport_key,
                days_back=21,
                minutes_before_commence=10,
            )
        except Exception as e:
            print(f"[nfl totals] WARNING: failed to build historical totals lines: {e}")
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

    league_avg_total_line = float(np.mean(league_avgs)) if league_avgs else float("nan")
    league_sd_total = float(np.mean(league_sds)) if league_sds else 11.0

    def _team_line_avg_sd(team_name: str) -> Tuple[float, float]:
        v = (team_total_lines or {}).get(team_name)
        if isinstance(v, dict) and v.get("avg") is not None:
            return (_safe_float(v.get("avg")), _safe_float(v.get("sd"), default=np.nan))
        return (float("nan"), float("nan"))

    # Build scoring table for expected points model (often sparse)
    team_tbl = _build_team_scoring_table()

    # ---- KEY FIX: stable league_pts baseline ----
    # If we have a league avg total line, per-team avg points is roughly half of it.
    if not np.isnan(league_avg_total_line):
        league_pts = float(_clamp(0.5 * league_avg_total_line, 21.0, 26.0))
    else:
        # reasonable NFL per-team baseline if history isn't available
        league_pts = 23.0

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
        rest_adj_elo = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Injuries + QB
        home_inj = build_injury_list_for_team_nfl(home, injuries_map)
        away_inj = build_injury_list_for_team_nfl(away, injuries_map)

        inj_pts_raw = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = _clamp(inj_pts_raw, -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        def qb_cost(lst) -> float:
            s = 0.0
            for (player, role, mult, impact) in (lst or []):
                try:
                    player_s = str(player).lower()
                    qb_like = (("quarterback" in player_s) or (player_s.find(" qb") >= 0) or (float(impact) >= 6.0))
                    if not qb_like:
                        continue
                    rw = 1.0 if role == "starter" else 0.55
                    s += rw * float(mult) * float(impact)
                except Exception:
                    continue
            return float(s)

        qb_home = qb_cost(home_inj)
        qb_away = qb_cost(away_inj)
        qb_diff = float(qb_away - qb_home)  # positive => away QB worse => home advantage

        inj_elo_adj = float(inj_pts) * float(INJ_ELO_PER_POINT)
        qb_elo_adj = float(qb_diff) * float(QB_EXTRA_ELO)
        inj_total_elo = _clamp(inj_elo_adj + qb_elo_adj, -MAX_ABS_INJ_ELO_ADJ, MAX_ABS_INJ_ELO_ADJ)

        # Form
        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = float(form_home - form_away)

        # Effective elos
        eh_eff = float(eh) + float(rest_adj_elo) + 0.5 * float(inj_total_elo) + 0.5 * float(form_diff)
        ea_eff = float(ea) - 0.5 * float(inj_total_elo) - 0.5 * float(form_diff)

        # Win prob
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

        # Market no-vig ML
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

        # ----------------------------
        # TOTALS: expected points model + market anchor
        # ----------------------------
        # Build a stable fallback total line baseline from historical market totals
        home_avg, home_sd = _team_line_avg_sd(home)
        away_avg, away_sd = _team_line_avg_sd(away)

        fallback_total_line = float("nan")
        if not np.isnan(home_avg) and not np.isnan(away_avg):
            fallback_total_line = 0.5 * (home_avg + away_avg)
        elif not np.isnan(home_avg):
            fallback_total_line = home_avg
        elif not np.isnan(away_avg):
            fallback_total_line = away_avg
        elif not np.isnan(league_avg_total_line):
            fallback_total_line = league_avg_total_line
        else:
            fallback_total_line = 46.0  # safe NFL-ish default

        # A little regression toward league average total line (keeps weird team avgs from dominating)
        if not np.isnan(league_avg_total_line) and not np.isnan(fallback_total_line):
            fallback_total_line = float(
                (1.0 - TOTAL_REGRESS_WEIGHT) * fallback_total_line + TOTAL_REGRESS_WEIGHT * league_avg_total_line
            )

        # 1) expected points from scoring strengths (or fallback baseline total line)
        exp_home, exp_away, model_total_outcome = _expected_points_total(
            home=home,
            away=away,
            league_pts=float(league_pts),
            team_tbl=team_tbl,
            fallback_total_line=float(fallback_total_line),
        )

        # 2) injuries/QB/rest adjustments (small, but directional)
        total_adj_inj = -INJ_POINTS_TO_TOTAL_PTS * (abs(float(inj_pts)))
        total_adj_qb = -QB_POINTS_PER_QB_IMPACT * (abs(float(qb_home)) + abs(float(qb_away)))
        total_adj_rest = REST_POINTS_PER_ELO * float(rest_adj_elo)

        model_total_outcome = float(model_total_outcome + total_adj_inj + total_adj_qb + total_adj_rest)

        # 3) weather adjustment (leave your existing integration alone)
        weather_temp = np.nan
        weather_wind = np.nan
        if ENABLE_WEATHER:
            kickoff_dt = _parse_iso_datetime((oi or {}).get("commence_time", "")) or None
            if kickoff_dt is None:
                kickoff_dt = datetime.utcnow()
            w = fetch_game_weather(home_team=home, game_dt_utc=kickoff_dt)

            # support dict return
            if isinstance(w, dict):
                t = w.get("temp_f")
                wd = w.get("wind_mph")
            else:
                t = getattr(w, "temp_f", None)
                wd = getattr(w, "wind_mph", None)

            if t is not None and t == t:
                weather_temp = float(t)
                if weather_temp < 35.0:
                    model_total_outcome -= float(COLD_PTS_IF_UNDER_35F)

            if wd is not None and wd == wd:
                weather_wind = float(wd)
                if weather_wind > 10.0:
                    model_total_outcome -= float((weather_wind - 10.0) * WIND_PTS_PER_MPH_OVER_10)

        # 4) anchor to market
        market_total = float(total_points) if not np.isnan(total_points) else float("nan")
        if not np.isnan(market_total):
            model_total = float(TOTAL_ANCHOR_W * model_total_outcome + (1.0 - TOTAL_ANCHOR_W) * market_total)
        else:
            model_total = float(model_total_outcome)

        # SD baseline from historical market totals
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
            market_total=float(market_total) if not np.isnan(market_total) else float("nan"),
            over_price=float(total_over_price),
            under_price=float(total_under_price),
            sd=float(sd),
        )

        total_pass_reason = _total_gate_reason(total_side, total_edge_vs_be, total_edge_pts)
        total_recommendation = _total_reco(total_side, total_edge_vs_be, total_edge_pts)

        # ----------------------------
        # PRIMARY selection (NFL: push totals first when value exists)
        # ----------------------------
        ml_score = float(abs(edge_home)) if not np.isnan(edge_home) else -999.0
        ats_score = float(ats_edge_vs_be) if str(spread_reco).startswith("Model PICK ATS:") else -999.0
        tot_score = float(total_edge_vs_be) if str(total_recommendation).startswith("Model PICK TOTAL:") else -999.0
        if tot_score > -900:
            tot_score *= float(TOTAL_PRIMARY_BOOST)

        primary = ml_pick
        why_primary = f"Primary=ML (abs_edge={ml_score:+.3f})"
        best = ml_score

        if ats_score > best:
            best = ats_score
            primary = spread_reco
            why_primary = f"Primary=ATS (edge_vs_be={ats_score:+.3f})"

        if tot_score > best:
            best = tot_score
            primary = total_recommendation
            why_primary = f"Primary=TOTAL (edge_vs_be={float(total_edge_vs_be):+.3f}; boosted)"

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
                "model_total_outcome": float(model_total_outcome),
                "model_total": float(model_total),
                "total_edge_points": float(total_edge_pts) if not np.isnan(total_edge_pts) else np.nan,
                "total_pick_side": total_side,
                "total_pick_prob": float(total_p_win) if not np.isnan(total_p_win) else np.nan,
                "total_breakeven_prob": float(total_be) if not np.isnan(total_be) else np.nan,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_pass_reason": str(total_pass_reason),
                "total_recommendation": str(total_recommendation),
                "ml_recommendation": ml_pick,
                "spread_recommendation": spread_reco,
                "primary_recommendation": primary,
                "why_primary": why_primary,
                "value_tier": value_tier,
                "elo_diff": float(elo_diff),
                "inj_points_raw": float(inj_pts_raw),
                "inj_points": float(inj_pts),
                "inj_elo_total": float(inj_total_elo),
                "qb_diff": float(qb_diff),
                "form_elo_diff": float(form_diff),
                "form_home_elo": float(form_home),
                "form_away_elo": float(form_away),
                "rest_days_home": np.nan if home_days_off is None else float(home_days_off),
                "rest_days_away": np.nan if away_days_off is None else float(away_days_off),
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_spread": home_spread,
                "spread_price": spread_price,
                "weather_temp_f": weather_temp,
                "weather_wind_mph": weather_wind,
            }
        )

    df = pd.DataFrame(rows)

    # top-N ATS filter (optional)
    if MAX_ATS_PLAYS_PER_DAY is not None and not df.empty:
        elig = df["spread_recommendation"].astype(str).str.contains("Model PICK ATS:", na=False)
        df["ats_rank_score"] = np.where(elig, df["ats_edge_vs_be"].astype(float), -999.0)
        top_idx = df.sort_values("ats_rank_score", ascending=False).head(MAX_ATS_PLAYS_PER_DAY).index
        keep = set(top_idx.tolist())

        for i in df.index:
            if bool(elig.loc[i]) and i not in keep:
                df.loc[i, "spread_recommendation"] = "No ATS bet (top-N filter)"
                df.loc[i, "ats_strength"] = "pass"
                df.loc[i, "ats_pass_reason"] = "top-N filter"

        df.drop(columns=["ats_rank_score"], inplace=True, errors="ignore")

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
    return run_daily_nfl(str(date_in), odds_dict=(odds_dict or {}))
