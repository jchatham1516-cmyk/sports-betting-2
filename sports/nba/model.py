# sports/nba/model.py
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
from sports.nba.injuries import (
    fetch_official_nba_injuries,
    build_injury_list_for_team_nba,
    injury_adjustment_points,
)

ELO_PATH = "results/elo_state_nba.json"

# ----------------------------
# Tunables (NBA-specific)
# ----------------------------
HOME_ADV = 55.0
ELO_K = 20.0

# Elo -> points (spread-ish)
ELO_PER_POINT = 40.0

# Injury points clamp (your injuries module returns "points-ish")
MAX_ABS_INJ_POINTS = 6.0

# Spread cap
MAX_ABS_MODEL_SPREAD = 17.0

# Rest effects (Elo)
SHORT_REST_PENALTY_ELO = -14.0
NORMAL_REST_BONUS_ELO = 0.0
BYE_BONUS_ELO = +8.0

# Recent form (based on scoring margin)
FORM_LOOKBACK_DAYS = 35
FORM_MIN_GAMES = 2
FORM_ELO_PER_POINT = 1.35
FORM_ELO_CLAMP = 40.0

# Prob compression
BASE_COMPRESS = 0.75

# ML threshold
MIN_ML_EDGE = 0.02

# ATS model
ATS_SD_PTS = 13.5
ATS_DEFAULT_PRICE = -110.0
ATS_MIN_EDGE_VS_BE = 0.03
ATS_MIN_PTS_EDGE = 2.0
ATS_BIG_LINE = 7.0
ATS_TINY_MODEL = 2.0
ATS_BIGLINE_FORCE_PASS = True
MAX_ATS_PLAYS_PER_DAY = 3  # set None to disable

# ----------------------------
# Totals model (improved)
# ----------------------------
TOTAL_LOOKBACK_DAYS = 45
TOTAL_MIN_GAMES = 4
TOTAL_DECAY = 0.90

TOTAL_HOME_BUMP = 0.5  # small home bump
TOTAL_SD_DEFAULT = 14.0
TOTAL_SD_CLAMP_LO = 9.0
TOTAL_SD_CLAMP_HI = 22.0

TOTAL_DEFAULT_PRICE = -110.0
TOTAL_MIN_EDGE_VS_BE = 0.02  # must beat breakeven by 2%
TOTAL_MIN_PTS_EDGE = 3.0     # must have >= 3 points edge to play totals


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
    if days_off >= 10:
        return float(BYE_BONUS_ELO)
    return float(NORMAL_REST_BONUS_ELO)


def _recent_form_adjustments(days_back: int = FORM_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    """
    { team: {"avg_margin": float, "games": int, "elo_adj": float } }
    """
    sport_key = SPORT_TO_ODDS_KEY.get("nba")
    if not sport_key:
        return {}

    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 60))
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
        out[team] = {"avg_margin": float(avg_margin), "games": int(games), "elo_adj": float(elo_adj)}

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


# ----------------------------
# ATS helpers
# ----------------------------
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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
# Totals helpers (improved)
# ----------------------------
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
    # P(actual > market) with mean=model_total, sd=sd
    z = (float(model_total) - float(market_total)) / float(sd)
    return float(_clamp(_phi(z), 0.001, 0.999))


def _total_pick(
    *,
    model_total: float,
    market_total: float,
    over_price: float,
    under_price: float,
    sd: float,
) -> Tuple[str, float, float, float, float]:
    """
    Returns: (side, p_win, edge_vs_be, breakeven, edge_points)
    edge_points = model_total - market_total
    """
    if np.isnan(model_total) or np.isnan(market_total):
        return ("NONE", float("nan"), float("nan"), float("nan"), float("nan"))

    edge_pts = float(model_total - market_total)

    p_over = _totals_prob_over(model_total, market_total, sd=sd)
    p_under = float(1.0 - p_over) if not np.isnan(p_over) else float("nan")

    be_over = _breakeven_prob_from_american(over_price)
    be_under = _breakeven_prob_from_american(under_price)

    if np.isnan(p_over) or np.isnan(p_under) or np.isnan(be_over) or np.isnan(be_under):
        return ("NONE", float("nan"), float("nan"), float("nan"), float(edge_pts))

    edge_over = float(p_over - be_over)
    edge_under = float(p_under - be_under)

    if edge_over >= edge_under:
        return ("OVER", float(p_over), float(edge_over), float(be_over), float(edge_pts))
    return ("UNDER", float(p_under), float(edge_under), float(be_under), float(edge_pts))


def _recent_total_stats(days_back: int = TOTAL_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    """
    Builds per-team weighted PF/PA and totals using recent games.
    Returns:
      { team: {"pf_w", "pa_w", "tot_w", "tot_sd", "games"} }
    """
    sport_key = SPORT_TO_ODDS_KEY.get("nba")
    if not sport_key:
        return {}

    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 70))
    except Exception:
        return {}

    per_team_games = defaultdict(list)  # team -> list[(date, pf, pa, total)]

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

        tot = float(hs + aw)
        per_team_games[home].append((d, float(hs), float(aw), tot))
        per_team_games[away].append((d, float(aw), float(hs), tot))

    out: Dict[str, Dict[str, float]] = {}
    for team, lst in per_team_games.items():
        lst = sorted(lst, key=lambda x: x[0], reverse=True)
        pf = [x[1] for x in lst]
        pa = [x[2] for x in lst]
        totals = [x[3] for x in lst]
        games = len(totals)

        if games < TOTAL_MIN_GAMES:
            continue

        pf_w = _weighted_mean(pf)
        pa_w = _weighted_mean(pa)
        tot_w = _weighted_mean(totals)

        try:
            sd = float(np.std([float(x) for x in totals[:12]], ddof=1)) if games >= 2 else float("nan")
        except Exception:
            sd = float("nan")

        out[team] = {
            "games": int(games),
            "pf_w": float(pf_w) if not np.isnan(pf_w) else np.nan,
            "pa_w": float(pa_w) if not np.isnan(pa_w) else np.nan,
            "tot_w": float(tot_w) if not np.isnan(tot_w) else np.nan,
            "tot_sd": float(sd) if not np.isnan(sd) else np.nan,
        }

    return out


# ----------------------------
# Data builders
# ----------------------------
def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nba"]
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
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nba"]

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
def run_daily_nba(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=14)

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
    totals_map = _recent_total_stats(days_back=TOTAL_LOOKBACK_DAYS)

    rows = []

    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        eh = st.get(home)
        ea = st.get(away)

        # Rest days
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Injuries (+ => HOME advantage)
        home_inj = build_injury_list_for_team_nba(home, injuries_map)
        away_inj = build_injury_list_for_team_nba(away, injuries_map)
        inj_pts_raw = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = _clamp(inj_pts_raw, -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        # Recent form (Elo-style)
        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = float(form_home - form_away)

        # Effective elos (symmetric form + injuries)
        eh_eff = float(eh) + float(rest_adj) + 0.5 * float(inj_pts) + 0.5 * float(form_diff)
        ea_eff = float(ea) - 0.5 * float(inj_pts) - 0.5 * float(form_diff)

        # Win prob + compression
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_home = _clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99)

        # Spread-ish
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(-(elo_diff / ELO_PER_POINT), -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # Market inputs
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=ATS_DEFAULT_PRICE)

        total_points = _safe_float((oi or {}).get("total_points"))
        total_over_price = _safe_float((oi or {}).get("over_price"), default=TOTAL_DEFAULT_PRICE)
        total_under_price = _safe_float((oi or {}).get("under_price"), default=TOTAL_DEFAULT_PRICE)

        # Market no-vig probability
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

        # Totals model (PF/PA matchup)
        hs = totals_map.get(home) or {}
        as_ = totals_map.get(away) or {}

        home_pf = _safe_float(hs.get("pf_w"))
        home_pa = _safe_float(hs.get("pa_w"))
        away_pf = _safe_float(as_.get("pf_w"))
        away_pa = _safe_float(as_.get("pa_w"))

        model_total = float("nan")
        if not np.isnan(home_pf) and not np.isnan(home_pa) and not np.isnan(away_pf) and not np.isnan(away_pa):
            exp_home = 0.5 * (home_pf + away_pa)
            exp_away = 0.5 * (away_pf + home_pa)
            model_total = float(exp_home + exp_away + TOTAL_HOME_BUMP)

        if np.isnan(model_total):
            home_tot = _safe_float(hs.get("tot_w"))
            away_tot = _safe_float(as_.get("tot_w"))
            if not np.isnan(home_tot) and not np.isnan(away_tot):
                model_total = float(0.5 * (home_tot + away_tot) + TOTAL_HOME_BUMP)

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

        total_side, total_p_win, total_edge_vs_be, total_be, total_edge_pts = _total_pick(
            model_total=float(model_total) if not np.isnan(model_total) else float("nan"),
            market_total=float(total_points) if not np.isnan(total_points) else float("nan"),
            over_price=float(total_over_price) if not np.isnan(total_over_price) else float(TOTAL_DEFAULT_PRICE),
            under_price=float(total_under_price) if not np.isnan(total_under_price) else float(TOTAL_DEFAULT_PRICE),
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
            if total_allowed and (np.isnan(total_edge_pts) or abs(total_edge_pts) < TOTAL_MIN_PTS_EDGE):
                total_allowed = False
                total_pass_reason = f"|total_edge_pts|<{TOTAL_MIN_PTS_EDGE:.1f}"

        if not total_allowed:
            total_reco = f"No total bet (gated): {total_pass_reason}" if total_pass_reason else "No total bet (gated)"
        else:
            total_reco = f"Model PICK TOTAL: {total_side} ({total_edge_pts:+.1f} pts, edge_vs_be={total_edge_vs_be:+.3f})"

        # Primary recommendation preference order: ATS > totals > ML
        primary = ml_pick
        if isinstance(spread_reco, str) and spread_reco.startswith("Model PICK ATS:"):
            primary = spread_reco
        elif isinstance(total_reco, str) and total_reco.startswith("Model PICK TOTAL:"):
            primary = total_reco

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,

            # Model
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),

            # Market processed
            "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
            "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
            "edge_away": float(edge_away) if not np.isnan(edge_away) else np.nan,

            # ATS
            "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
            "ats_home_cover_prob": float(p_home_cover) if not np.isnan(p_home_cover) else np.nan,
            "ats_pick_side": ats_side,
            "ats_pick_prob": float(ats_p_win) if not np.isnan(ats_p_win) else np.nan,
            "ats_breakeven_prob": float(ats_be) if not np.isnan(ats_be) else np.nan,
            "ats_edge_vs_be": float(ats_edge_vs_be) if not np.isnan(ats_edge_vs_be) else np.nan,
            "ats_strength": ats_strength,
            "ats_pass_reason": ats_pass_reason,

            # Totals (O/U)
            "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
            "total_over_price": float(total_over_price) if not np.isnan(total_over_price) else np.nan,
            "total_under_price": float(total_under_price) if not np.isnan(total_under_price) else np.nan,
            "model_total": float(model_total) if not np.isnan(model_total) else np.nan,
            "total_edge_points": float(total_edge_pts) if not np.isnan(total_edge_pts) else np.nan,
            "total_pick_side": total_side,
            "total_pick_prob": float(total_p_win) if not np.isnan(total_p_win) else np.nan,
            "total_breakeven_prob": float(total_be) if not np.isnan(total_be) else np.nan,
            "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
            "total_pass_reason": total_pass_reason,
            "total_recommendation": total_reco,

            # Recos
            "ml_recommendation": ml_pick,
            "spread_recommendation": spread_reco,
            "primary_recommendation": primary,
            "value_tier": value_tier,

            # Debug
            "elo_diff": float(elo_diff),
            "inj_points_raw": float(inj_pts_raw),
            "inj_points": float(inj_pts),
            "form_elo_diff": float(form_diff),
            "form_home_elo": float(form_home),
            "form_away_elo": float(form_away),
            "form_home_avg_margin": float((form_map.get(home) or {}).get("avg_margin", np.nan)),
            "form_away_avg_margin": float((form_map.get(away) or {}).get("avg_margin", np.nan)),
            "form_home_games": float((form_map.get(home) or {}).get("games", np.nan)),
            "form_away_games": float((form_map.get(away) or {}).get("games", np.nan)),
            "rest_days_home": np.nan if home_days_off is None else float(home_days_off),
            "rest_days_away": np.nan if away_days_off is None else float(away_days_off),

            # Raw market
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "spread_price": spread_price,
        })

    df = pd.DataFrame(rows)

    # Optional: limit ATS picks to top N edges per slate
    if MAX_ATS_PLAYS_PER_DAY is not None and not df.empty and "ats_edge_vs_be" in df.columns:
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

        # refresh primary after filtering
        if "primary_recommendation" in df.columns:
            for i in df.index:
                primary2 = df.loc[i, "ml_recommendation"]
                sr = str(df.loc[i, "spread_recommendation"])
                tr = str(df.loc[i, "total_recommendation"]) if "total_recommendation" in df.columns else ""
                if sr.startswith("Model PICK ATS:"):
                    primary2 = sr
                elif tr.startswith("Model PICK TOTAL:"):
                    primary2 = tr
                df.loc[i, "primary_recommendation"] = primary2

    return df


# Backwards-compatible alias (IMPORTANT for your runner)
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
