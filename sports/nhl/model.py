# sports/nhl/model.py
from __future__ import annotations

import os
import math
from datetime import datetime, date
from typing import Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.nhl.injuries import (
    fetch_espn_nhl_injuries,
    build_injury_list_for_team_nhl,
    injury_adjustment_points,
)

ELO_PATH = "results/elo_state_nhl.json"

# ----------------------------
# Tunables (NHL-specific)
# ----------------------------
HOME_ADV = 45.0
ELO_K = 18.0

# Elo -> goals (spread-ish)
ELO_PER_GOAL = 55.0
MAX_ABS_MODEL_SPREAD = 2.0

# Injuries -> Elo
INJ_ELO_PER_POINT = 18.0
INJ_PTS_CLAMP = 6.0

TOTAL_LOOKBACK_DAYS = 3
TOTAL_MIN_GAMES = 1

# Goalie weighting
GOALIE_IMPACT_THRESHOLD = 3.0
GOALIE_EXTRA_ELO_PER_POINT = 16.0

# Rest/fatigue (Elo)
B2B_PENALTY_ELO = -18.0
ONE_DAY_REST_ELO = -6.0
THREE_PLUS_BONUS_ELO = +3.0

# Prob compression
BASE_COMPRESS = 0.65
GOALIE_UNCERTAINTY_MULT = 0.85

# ----------------------------
# Totals model
# ----------------------------
TOTAL_LOOKBACK_DAYS = 28
TOTAL_MIN_GAMES = 3
TOTAL_DECAY = 0.90

TOTAL_SD_DEFAULT = 1.35
TOTAL_SD_CLAMP_LO = 0.90
TOTAL_SD_CLAMP_HI = 2.40

TOTAL_DEFAULT_PRICE = -110.0
TOTAL_MIN_EDGE_VS_BE = 0.03
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


def _rest_adjustment_days(days_rest: Optional[int]) -> float:
    if days_rest is None:
        return 0.0
    # clamp weird values
    try:
        days_rest = int(days_rest)
    except Exception:
        return 0.0
    if days_rest <= 0:
        return float(B2B_PENALTY_ELO)
    if days_rest == 1:
        return float(ONE_DAY_REST_ELO)
    if days_rest >= 3:
        return float(THREE_PLUS_BONUS_ELO)
    return 0.0


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


def _weighted_mean(values: list[float], decay: float = TOTAL_DECAY) -> float:
    if not values:
        return float("nan")
    w = 1.0
    num = 0.0
    den = 0.0
    for v in values:
        try:
            v = float(v)
        except Exception:
            continue
        if np.isnan(v):
            continue
        num += w * v
        den += w
        w *= float(decay)
    if den <= 0:
        return float("nan")
    return float(num / den)


def _totals_prob_over(model_total: float, market_total: float, sd: float) -> float:
    if np.isnan(model_total) or np.isnan(market_total) or np.isnan(sd) or sd <= 0:
        return float("nan")
    # P(total goals > market_total)
    z = (float(market_total) - float(model_total)) / float(sd)
    return float(_clamp(1.0 - _phi(z), 0.001, 0.999))


def _total_pick(
    *,
    model_total: float,
    market_total: float,
    over_price: float,
    under_price: float,
    sd: float,
) -> Tuple[str, float, float, float, float]:
    """
    Returns: (side, p_win, edge_vs_be, breakeven, edge_goals)
    edge_goals = model_total - market_total
    """
    if np.isnan(model_total) or np.isnan(market_total):
        return ("NONE", float("nan"), float("nan"), float("nan"), float("nan"))

    edge_goals = float(model_total - market_total)

    p_over = _totals_prob_over(model_total, market_total, sd=sd)
    p_under = float(1.0 - p_over) if not np.isnan(p_over) else float("nan")

    be_over = _breakeven_prob_from_american(over_price)
    be_under = _breakeven_prob_from_american(under_price)

    if np.isnan(p_over) or np.isnan(p_under) or np.isnan(be_over) or np.isnan(be_under):
        return ("NONE", float("nan"), float("nan"), float("nan"), float(edge_goals))

    edge_over = float(p_over - be_over)
    edge_under = float(p_under - be_under)

    if edge_over >= edge_under:
        return ("OVER", float(p_over), float(edge_over), float(be_over), float(edge_goals))
    return ("UNDER", float(p_under), float(edge_under), float(be_under), float(edge_goals))


def _goalie_cost(inj_list) -> float:
    if not inj_list:
        return 0.0
    role_w = {"starter": 1.0, "rotation": 0.55}
    total = 0.0
    for item in inj_list:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            continue
        _, role, mult, impact = item
        try:
            impact = float(impact)
            mult = float(mult)
        except Exception:
            continue
        if impact >= GOALIE_IMPACT_THRESHOLD:
            total += role_w.get(role, 0.6) * mult * impact
    return float(total)


# ----------------------------
# Builders
# ----------------------------
def _build_last_game_date_map(days_back: int = 10) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nhl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 10))

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


def _recent_goal_totals(days_back: int = TOTAL_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    sport_key = SPORT_TO_ODDS_KEY.get("nhl")
    if not sport_key:
        return {}

    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 45))
    except Exception:
        return {}

    per_team = defaultdict(list)  # team -> list[(date, gf, ga, total)]

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

        tot = float(hs + aw)
        per_team[home].append((d, float(hs), float(aw), tot))
        per_team[away].append((d, float(aw), float(hs), tot))

    out: Dict[str, Dict[str, float]] = {}
    for team, lst in per_team.items():
        lst = sorted(lst, key=lambda x: x[0], reverse=True)
        gf = [x[1] for x in lst]
        ga = [x[2] for x in lst]
        totals = [x[3] for x in lst]
        games = len(totals)
        if games < TOTAL_MIN_GAMES:
            continue

        gf_w = _weighted_mean(gf)
        ga_w = _weighted_mean(ga)
        tot_w = _weighted_mean(totals)

        try:
            sd = float(np.std([float(x) for x in totals[:12]], ddof=1)) if games >= 2 else float("nan")
        except Exception:
            sd = float("nan")

        out[team] = {
            "games": int(games),
            "gf_w": float(gf_w) if not np.isnan(gf_w) else np.nan,
            "ga_w": float(ga_w) if not np.isnan(ga_w) else np.nan,
            "tot_w": float(tot_w) if not np.isnan(tot_w) else np.nan,
            "tot_sd": float(sd) if not np.isnan(sd) else np.nan,
        }

    return out


def update_elo_from_recent_scores(days_from: int = 3) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nhl"]

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

        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))
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
def run_daily_nhl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=3)

    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # injuries
    try:
        injuries_map = fetch_espn_nhl_injuries()
    except Exception as e:
        print(f"[nhl injuries] WARNING: failed to load ESPN injuries: {e}")
        injuries_map = {}

    last_played = _build_last_game_date_map(days_back=10)
    totals_map = _recent_goal_totals(days_back=TOTAL_LOOKBACK_DAYS)

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        eh = st.get(home)
        ea = st.get(away)

        # Rest
        home_rest_days = None
        away_rest_days = None
        if target_date is not None:
            h_last = last_played.get(home)
            a_last = last_played.get(away)
            if h_last is not None:
                home_rest_days = (target_date - h_last).days - 1
            if a_last is not None:
                away_rest_days = (target_date - a_last).days - 1

        home_rest_elo = _rest_adjustment_days(home_rest_days)
        away_rest_elo = _rest_adjustment_days(away_rest_days)
        rest_elo_adj = float(home_rest_elo - away_rest_elo)

        # Injuries + goalies
        home_inj = build_injury_list_for_team_nhl(home, injuries_map)
        away_inj = build_injury_list_for_team_nhl(away, injuries_map)

        inj_pts = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = _clamp(inj_pts, -INJ_PTS_CLAMP, INJ_PTS_CLAMP)
        inj_elo_adj = float(inj_pts) * float(INJ_ELO_PER_POINT)

        home_goalie_cost = _goalie_cost(home_inj)
        away_goalie_cost = _goalie_cost(away_inj)
        goalie_diff = float(away_goalie_cost - home_goalie_cost)
        goalie_elo_adj = float(goalie_diff) * float(GOALIE_EXTRA_ELO_PER_POINT)

        goalie_uncertain = (home_goalie_cost == 0.0 and away_goalie_cost == 0.0)

        # Effective Elo
        eh_eff = float(eh) + float(inj_elo_adj) + float(goalie_elo_adj) + float(rest_elo_adj)
        ea_eff = float(ea)

        p_home_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        compress = BASE_COMPRESS * (GOALIE_UNCERTAINTY_MULT if goalie_uncertain else 1.0)
        p_home = _clamp(0.5 + compress * (p_home_raw - 0.5), 0.01, 0.99)

        # Spread-ish
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(-(elo_diff / ELO_PER_GOAL), -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # Market
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))

        total_points = _safe_float((oi or {}).get("total_points"))
        over_price = _safe_float((oi or {}).get("over_price"), default=TOTAL_DEFAULT_PRICE)
        under_price = _safe_float((oi or {}).get("under_price"), default=TOTAL_DEFAULT_PRICE)

        # ----------------
        # Totals model (goals)
        # ----------------
        hs = totals_map.get(home) or {}
        as_ = totals_map.get(away) or {}

        home_gf = _safe_float(hs.get("gf_w"))
        home_ga = _safe_float(hs.get("ga_w"))
        away_gf = _safe_float(as_.get("gf_w"))
        away_ga = _safe_float(as_.get("ga_w"))

        model_total = float("nan")
        if not np.isnan(home_gf) and not np.isnan(home_ga) and not np.isnan(away_gf) and not np.isnan(away_ga):
            exp_home = 0.5 * (home_gf + away_ga)
            exp_away = 0.5 * (away_gf + home_ga)
            model_total = float(exp_home + exp_away)

        if np.isnan(model_total):
            home_tot = _safe_float(hs.get("tot_w"))
            away_tot = _safe_float(as_.get("tot_w"))
            if not np.isnan(home_tot) and not np.isnan(away_tot):
                model_total = float(0.5 * (home_tot + away_tot))

        home_sd = _safe_float(hs.get("tot_sd"))
        away_sd = _safe_float(as_.get("tot_sd"))
        if not np.isnan(home_sd) and not np.isnan(away_sd):
            sd = float(0.5 * (home_sd + away_sd))
        elif not np.isnan(home_sd):
            sd = float(home_sd)
        elif not np.isnan(away_sd):
            sd = float(away_sd)
        else:
            sd = float(TOTAL_SD_DEFAULT)
        sd = _clamp(sd, TOTAL_SD_CLAMP_LO, TOTAL_SD_CLAMP_HI)

        total_side, total_p_win, total_edge_vs_be, total_be, total_edge_goals = _total_pick(
            model_total=float(model_total) if not np.isnan(model_total) else float("nan"),
            market_total=float(total_points) if not np.isnan(total_points) else float("nan"),
            over_price=float(over_price) if not np.isnan(over_price) else float(TOTAL_DEFAULT_PRICE),
            under_price=float(under_price) if not np.isnan(under_price) else float(TOTAL_DEFAULT_PRICE),
            sd=float(sd),
        )

        total_allowed = True
        total_pass_reason = ""
        if np.isnan(total_points) or np.isnan(model_total) or total_side == "NONE":
            total_allowed = False
            total_pass_reason = "missing total/model"
        else:
            if np.isnan(total_edge_vs_be) or total_edge_vs_be < TOTAL_MIN_EDGE_VS_BE:
                total_allowed = False
                total_pass_reason = f"total_edge_vs_be<{TOTAL_MIN_EDGE_VS_BE:.3f}"
            if total_allowed and (np.isnan(total_edge_goals) or abs(total_edge_goals) < TOTAL_MIN_GOALS_EDGE):
                total_allowed = False
                total_pass_reason = f"|total_edge_goals|<{TOTAL_MIN_GOALS_EDGE:.2f}"

        if not total_allowed:
            total_reco = f"No total bet (gated): {total_pass_reason}" if total_pass_reason else "No total bet (gated)"
        else:
            total_reco = f"Model PICK TOTAL: {total_side} ({total_edge_goals:+.2f} goals, edge_vs_be={total_edge_vs_be:+.3f})"

        # Primary recommendation: TOTALS (if pick) else ML-ish (you can expand later)
        primary = total_reco if isinstance(total_reco, str) and total_reco.startswith("Model PICK TOTAL:") else "No primary pick"

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,

            # Model
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),

            # Totals
            "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
            "total_over_price": float(over_price) if not np.isnan(over_price) else np.nan,
            "total_under_price": float(under_price) if not np.isnan(under_price) else np.nan,
            "model_total": float(model_total) if not np.isnan(model_total) else np.nan,
            "total_edge_goals": float(total_edge_goals) if not np.isnan(total_edge_goals) else np.nan,
            "total_pick_side": total_side,
            "total_pick_prob": float(total_p_win) if not np.isnan(total_p_win) else np.nan,
            "total_breakeven_prob": float(total_be) if not np.isnan(total_be) else np.nan,
            "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
            "total_recommendation": total_reco,
            "total_pass_reason": total_pass_reason,
            "total_sd": float(sd),

            # Market
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,

            # Recommendation
            "primary_recommendation": primary,

            # Debug
            "elo_diff": float(elo_diff),
            "inj_points": float(inj_pts),
            "goalie_diff": float(goalie_diff),
            "rest_days_home": np.nan if home_rest_days is None else float(home_rest_days),
            "rest_days_away": np.nan if away_rest_days is None else float(away_rest_days),
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
    return run_daily_nhl(str(date_in), odds_dict=(odds_dict or {}))
