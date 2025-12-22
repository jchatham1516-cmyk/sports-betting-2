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

FORM_LOOKBACK_DAYS = 70
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
# Totals model
# ----------------------------
TOTAL_LOOKBACK_DAYS = 3            # IMPORTANT: Odds API scores daysFrom max is 3
TOTAL_MIN_GAMES = 1
TOTAL_DECAY = 0.88

TOTAL_SD_DEFAULT = 13.0
TOTAL_MIN_EDGE_VS_BE = 0.03
TOTAL_MIN_PTS_EDGE = 1.5

TOTAL_SHRINK_TO_SLATE = 0.40


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


# ---------- Totals helpers ----------
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
    if np.isnan(model_total) or np.isnan(market_total):
        return ("NONE", float("nan"), float("nan"), float("nan"), float("nan"))

    total_edge_pts = float(model_total - market_total)

    p_over = _totals_prob_over(model_total, market_total, sd=sd)
    p_under = float(1.0 - p_over) if not np.isnan(p_over) else float("nan")

    be_over = _breakeven_prob_from_american(over_price)
    be_under = _breakeven_prob_from_american(under_price)

    if np.isnan(p_over) or np.isnan(p_under) or np.isnan(be_over) or np.isnan(be_under):
        return ("NONE", float("nan"), float("nan"), float("nan"), float(total_edge_pts))

    edge_over = float(p_over - be_over)
    edge_under = float(p_under - be_under)

    if edge_over >= edge_under:
        return ("OVER", float(p_over), float(edge_over), float(be_over), float(total_edge_pts))
    return ("UNDER", float(p_under), float(edge_under), float(be_under), float(total_edge_pts))


def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nfl"]
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


def _recent_total_stats(days_back: int = TOTAL_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    sport_key = SPORT_TO_ODDS_KEY.get("nfl")
    if not sport_key:
        return {}

    days = min(int(days_back), 3)
    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=days)
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
        totals = [x[3] for x in lst]
        pf = [x[1] for x in lst]
        pa = [x[2] for x in lst]
        games = len(totals)

        if games < TOTAL_MIN_GAMES:
            continue

        pf_w = _weighted_mean(pf)
        pa_w = _weighted_mean(pa)
        tot_w = _weighted_mean(totals)

        try:
            sd = float(np.std([float(x) for x in totals[:
