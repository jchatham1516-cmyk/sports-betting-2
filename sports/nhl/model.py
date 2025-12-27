# sports/nhl/model.py
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

# probability + margin calibrators
from sports.common.prob_calibration import load as load_platt, save as save_platt, fit_platt
from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin

ELO_PATH = "results/elo_state_nhl.json"
PLATT_PATH = "results/prob_cal_nhl.json"
MARGIN_CAL_PATH = "results/margin_cal_nhl.json"

# ----------------------------
# Tunables (NHL-specific)
# ----------------------------
HOME_ADV = 45.0
ELO_K = 18.0

# Elo -> goals (fallback until margin calibrator is trained)
ELO_PER_GOAL = 55.0

# Spread cap
MAX_ABS_MODEL_SPREAD = 2.5

# Prob compression
BASE_COMPRESS = 0.78

# ML threshold
MIN_ML_EDGE = 0.02

# Calibration minimum games
CAL_MIN_GAMES = 120

# Rest effects (simple)
SHORT_REST_PENALTY_ELO = -10.0
NORMAL_REST_BONUS_ELO = 0.0

# Recent form (margin = goals for - goals against)
FORM_LOOKBACK_DAYS = 35
FORM_MIN_GAMES = 2
FORM_ELO_PER_GOAL = 7.0
FORM_ELO_CLAMP = 35.0

# ----------------------------
# ATS (puck line) settings
# ----------------------------
ENABLE_ATS = os.getenv("NHL_ENABLE_ATS", "0") == "1"

ATS_SD_GOALS = 1.35
ATS_DEFAULT_PRICE = -110.0
ATS_MIN_EDGE_VS_BE = 0.03
ATS_MIN_GOALS_EDGE = 0.35
ATS_BIG_LINE = 1.5
ATS_TINY_MODEL = 0.35
ATS_BIGLINE_FORCE_PASS = False
MAX_ATS_PLAYS_PER_DAY = 3  # set None to disable

# ----------------------------
# Totals (historical market totals lines)
# ----------------------------
TOTAL_DEFAULT_PRICE = -110.0
TOTAL_HIST_DAYS = 21
TOTAL_REGRESS_WEIGHT = 0.40
TOTAL_SD_FLOOR = 0.55
TOTAL_SD_CEIL = 1.35
TOTAL_MIN_EDGE_VS_BE = 0.02
TOTAL_MIN_GOALS_EDGE = 0.35


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
    if days_off <= 0:
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


def _cover_prob_from_edge(spread_edge_goals: float, sd_goals: float = ATS_SD_GOALS) -> float:
    if spread_edge_goals is None or np.isnan(spread_edge_goals):
        return float("nan")
    z = float(spread_edge_goals) / float(sd_goals)
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


def _total_gate_reason(side: str, edge_vs_be: float, edge_goals: float) -> str:
    if side == "NONE":
        return "missing total/model"
    if np.isnan(edge_vs_be) or np.isnan(edge_goals):
        return "missing price/model"
    if abs(edge_goals) < TOTAL_MIN_GOALS_EDGE:
        return "edge too small"
    if edge_vs_be < TOTAL_MIN_EDGE_VS_BE:
        return f"edge_vs_be<{TOTAL_MIN_EDGE_VS_BE:.3f}"
    return ""


def _total_reco(side: str, edge_vs_be: float, edge_goals: float) -> str:
    reason = _total_gate_reason(side, edge_vs_be, edge_goals)
    if reason:
        return f"No total bet ({reason})"
    return f"Model PICK TOTAL: {side}"


def _elo_diff_from_prob(p: float) -> float:
    # Elo diff such that p = 1/(1+10^(-diff/400))
    p = float(_clamp(p, 0.01, 0.99))
    return float(400.0 * math.log10(p / (1.0 - p)))


def _seed_missing_elos_from_market(st: EloState, odds_dict: dict) -> None:
    """
    If Elo state is empty / missing teams (common for NHL because score history is limited),
    seed team ratings from today's market no-vig probabilities so we don't get all-1500.
    """
    if odds_dict is None:
        return

    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        # Need MLs to seed
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        if np.isnan(home_ml) or np.isnan(away_ml):
            continue

        mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)
        if np.isnan(mkt_home_p):
            continue

        # If either team missing, seed both around 1500 with a diff that matches market
        if (not st.has(home)) or (not st.has(away)):
            diff = _elo_diff_from_prob(float(mkt_home_p))
            # market probability already includes home edge; remove HOME_ADV from the diff for team-only ratings
            team_only_diff = diff - float(HOME_ADV)

            # set around 1500 baseline
            base = 1500.0
            st.set(home, base + 0.5 * team_only_diff)
            st.set(away, base - 0.5 * team_only_diff)


# ----------------------------
# Builders
# ----------------------------
def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nhl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 3))

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
    sport_key = SPORT_TO_ODDS_KEY.get("nhl")
    if not sport_key:
        return {}

    try:
        # NHL scores endpoint is limited; keep modest
        events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 3))
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
        elo_adj = _clamp(avg_margin * FORM_ELO_PER_GOAL, -FORM_ELO_CLAMP, FORM_ELO_CLAMP)
        out[team] = {"avg_margin": avg_margin, "games": int(games), "elo_adj": float(elo_adj)}

    return out


def update_elo_from_recent_scores(days_from: int = 3) -> EloState:
    """
    Updates Elo from recent completed games (NHL scores feed is limited).
    ALSO trains:
      - Platt probability calibrator (compressed prob -> calibrated prob)
      - Margin calibrator (elo_diff -> expected goals margin)
    """
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nhl"]

    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 3))

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

        # collect calibration signal BEFORE updating Elo
        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        p_comp = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))

        train_ps.append(p_comp)
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

    # fit + save calibrators when enough samples
    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            cal = fit_platt(np.array(train_ps, dtype=float), np.array(train_ys, dtype=float))
            save_platt(PLATT_PATH, cal)

            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nhl calibration] WARNING: calibration fit failed: {e}")

    return st


# ----------------------------
# Main daily run
# ----------------------------
def run_daily_nhl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=180)

    # IMPORTANT: seed missing elos from today's market so we don't get all-1500
    _seed_missing_elos_from_market(st, odds_dict)

    platt = load_platt(PLATT_PATH)
    margin_cal = load_margin_cal(MARGIN_CAL_PATH)

    def _margin_model_spread_from_elo_diff(elo_diff: float) -> float:
        try:
            if abs(getattr(margin_cal, "a", 0.0)) < 1e-9 and abs(getattr(margin_cal, "b", 0.0)) < 1e-9:
                return float(-(elo_diff / ELO_PER_GOAL))
            pred_margin = float(margin_cal.predict(float(elo_diff)))  # home_goals - away_goals
            return float(-pred_margin)
        except Exception:
            return float(-(elo_diff / ELO_PER_GOAL))

    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    last_played = _build_last_game_date_map(days_back=21)
    form_map = _recent_form_adjustments(days_back=FORM_LOOKBACK_DAYS)

    # Historical MARKET totals lines (total goals)
    sport_key = SPORT_TO_ODDS_KEY.get("nhl")
    team_total_lines = {}
    if sport_key:
        try:
            team_total_lines = build_team_historical_total_lines(
                sport_key=sport_key,
                days_back=TOTAL_HIST_DAYS,
                minutes_before_commence=10,
            )
        except Exception as e:
            print(f"[nhl totals] WARNING: failed to build historical totals lines: {e}")
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
    league_sd_total = float(np.mean(league_sds)) if league_sds else 0.95
# ---- FALLBACK: if historical totals lines failed, use today's market totals to build a baseline ----
if (np.isnan(league_avg_total) or not league_avgs) and odds_dict:
    market_totals = []
    for _k, oi in (odds_dict or {}).items():
        tp = _safe_float((oi or {}).get("total_points"))
        if not np.isnan(tp):
            market_totals.append(float(tp))

    if market_totals:
        league_avg_total = float(np.mean(market_totals))
        league_sd_total = float(np.std(market_totals)) if len(market_totals) >= 3 else float(league_sd_total)
        if np.isnan(league_sd_total) or league_sd_total <= 0:
            league_sd_total = 0.95

    def _team_line_avg_sd(team_name: str) -> Tuple[float, float]:
        v = (team_total_lines or {}).get(team_name)
        if isinstance(v, dict) and v.get("avg") is not None:
            return (_safe_float(v.get("avg")), _safe_float(v.get("sd"), default=np.nan))
        return (float("nan"), float("nan"))

    rows = []
    default_elo_count = 0

    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        eh = st.get(home)
        ea = st.get(away)

        if eh == st.default_elo or ea == st.default_elo:
            default_elo_count += 1
            print(
                f"[NHL WARNING] Default Elo used (mapping/training issue): "
                f"home={home_in}->{home} eh={eh}, away={away_in}->{away} ea={ea}"
            )

        # Rest
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Form
        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = float(form_home - form_away)

        # Effective elos
        eh_eff = float(eh) + float(rest_adj) + 0.5 * float(form_diff)
        ea_eff = float(ea) - 0.5 * float(form_diff)

        # Win prob (raw -> compressed -> calibrated)
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_comp = _clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99)
        try:
            p_home = _clamp(float(platt.predict(float(p_comp))), 0.01, 0.99)
        except Exception:
            p_home = p_comp

        # Spread-ish (goals)
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(
            _margin_model_spread_from_elo_diff(float(elo_diff)),
            -MAX_ABS_MODEL_SPREAD,
            MAX_ABS_MODEL_SPREAD,
        )

        # Market inputs
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

        # ATS (disabled by default)
        if not ENABLE_ATS:
            spread_edge_home = float("nan")
            p_home_cover = float("nan")
            ats_side, ats_p_win, ats_edge_vs_be, ats_be = ("NONE", float("nan"), float("nan"), float("nan"))
            ats_strength = "pass"
            ats_pass_reason = "disabled"
            spread_reco = "No ATS bet (disabled)"
        else:
            spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) else float("nan")
            p_home_cover = _cover_prob_from_edge(spread_edge_home, sd_goals=ATS_SD_GOALS)
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
                if ats_allowed and (np.isnan(spread_edge_home) or abs(spread_edge_home) < ATS_MIN_GOALS_EDGE):
                    ats_allowed = False
                    ats_pass_reason = f"|spread_edge|<{ATS_MIN_GOALS_EDGE:.2f}"

            if not ats_allowed:
                ats_strength = "pass"
                spread_reco = f"No ATS bet (gated): {ats_pass_reason}" if ats_pass_reason else "No ATS bet (gated)"
            else:
                ats_strength = _ats_strength_label(ats_edge_vs_be)
                spread_reco = _ats_reco(ats_side, ats_strength)

        # Totals
        home_avg, home_sd = _team_line_avg_sd(home)
        away_avg, away_sd = _team_line_avg_sd(away)

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

        # Primary selection
        ml_edge_abs = float(abs(edge_home)) if not np.isnan(edge_home) else -999.0

        ats_edge_val = -999.0
        if ENABLE_ATS and str(spread_reco).startswith("Model PICK ATS:"):
            ats_edge_val = float(ats_edge_vs_be) if not np.isnan(ats_edge_vs_be) else -999.0

        tot_edge_val = float(total_edge_vs_be) if str(total_recommendation).startswith("Model PICK TOTAL:") else -999.0

        primary = ml_pick
        why_primary = f"Primary=ML (abs_edge={ml_edge_abs:+.3f})"
        best_edge = ml_edge_abs

        if ats_edge_val > best_edge:
            best_edge = ats_edge_val
            primary = spread_reco
            why_primary = f"Primary=ATS (edge_vs_be={ats_edge_val:+.3f})"
        if tot_edge_val > best_edge:
            best_edge = tot_edge_val
            primary = total_recommendation
            why_primary = f"Primary=TOTAL (edge_vs_be={tot_edge_val:+.3f})"

        rows.append(
            {
                "date": game_date_str,
                "home": home,
                "away": away,
                "model_home_prob": float(p_home),
                "model_spread_home": float(model_spread_home),
                "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
                "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
                "edge_away": float(edge_away) if not np.isnan(edge_away) else np.nan,
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
                "spread_recommendation": str(spread_reco),
                "primary_recommendation": str(primary),
                "why_primary": why_primary,
                "value_tier": value_tier,
                "elo_diff": float(elo_diff),
                "rest_days_home": np.nan if home_days_off is None else float(home_days_off),
                "rest_days_away": np.nan if away_days_off is None else float(away_days_off),
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_spread": home_spread,
                "spread_price": spread_price,
            }
        )

    df = pd.DataFrame(rows)

    # Sanity check: only hard-fail if we STILL look broken after seeding
    if len(rows) >= 5:
        probs = [round(r["model_home_prob"], 3) for r in rows if not np.isnan(r.get("model_home_prob", np.nan))]
        if len(set(probs)) <= 2:
            # Instead of crashing your whole run, warn and continue output
            print("[NHL WARNING] Model produced near-constant probabilities — likely Elo not trained; output still saved.")
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
    return run_daily_nhl(str(date_in), odds_dict=(odds_dict or {}))
