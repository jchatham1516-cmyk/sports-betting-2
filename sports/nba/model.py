# sports/nba/model.py
from __future__ import annotations

import json
import logging
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
    build_injury_list_for_team_nba,
    build_injury_detail_list_for_team_nba,
    injury_adjustment_points,
    _fetch_from_espn,
    _fetch_from_official_nba,
)

from sports.common.eval import build_game_key
from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin
from sports.common.calibration import load_nba_calibrator, update_and_save_nba_calibration
from sports.common.prob_calibration import update_prob_calibration
from sports.common.prob_uncertainty import update_uncertainty, load_uncertainty

# BallDontLie client (already in your repo)
from sports.nba.bdl_client import bdl_get, season_start_year_for_date, get_bdl_api_key

ELO_PATH = "results/elo_state_nba.json"
MARGIN_CAL_PATH = "results/margin_cal_nba.json"
TOTAL_CAL_PATH = "results/total_cal_nba.json"
UNCERTAINTY_PATH = "results/prob_uncertainty_nba.json"

# ----------------------------
# Tunables
# ----------------------------
HOME_ADV = float(os.getenv("NBA_HOME_ADV", "35.0"))
ELO_K = float(os.getenv("NBA_ELO_K", "20.0"))

ELO_TRAIN_DAYS = int(os.getenv("NBA_ELO_TRAIN_DAYS", "200"))
CAL_MIN_GAMES = int(os.getenv("NBA_CAL_MIN_GAMES", "80"))
PROB_CAL_WINDOW = int(os.getenv("NBA_PROB_CAL_WINDOW", "240"))
PROB_CAL_MIN_GAMES = int(os.getenv("NBA_PROB_CAL_MIN_GAMES", str(CAL_MIN_GAMES)))

# injuries -> margin adjust
MAX_ABS_INJ_POINTS = float(os.getenv("NBA_MAX_ABS_INJ_POINTS", "6.0"))
INJ_DAMP = float(os.getenv("NBA_INJ_DAMP", "0.60"))
INJURY_LOW_CONF_MULT = float(os.getenv("NBA_INJURY_LOW_CONF_MULT", "0.60"))
INJURY_MED_CONF_MULT = float(os.getenv("NBA_INJURY_MED_CONF_MULT", "0.85"))
ELO_PER_POINT = float(os.getenv("NBA_ELO_PER_POINT", "40.0"))

# basic regularization
BASE_COMPRESS = float(os.getenv("NBA_BASE_COMPRESS", "0.95"))
MIN_ML_EDGE = float(os.getenv("NBA_MIN_ML_EDGE", "0.02"))

# form/recency adjustments (derived from recent net ratings)
FORM_ELO_PER_NET = float(os.getenv("NBA_FORM_ELO_PER_NET", "2.0"))
FORM_ELO_MAX_ABS = float(os.getenv("NBA_FORM_ELO_MAX_ABS", "120.0"))

# totals model
PTS_LOOKBACK_DAYS = int(os.getenv("NBA_PTS_LOOKBACK_DAYS", "60"))
PTS_MIN_GAMES = int(os.getenv("NBA_PTS_MIN_GAMES", "3"))
PTS_REGRESS = float(os.getenv("NBA_PTS_REGRESS", "0.30"))
PTS_LEAGUE_CLAMP_MIN = float(os.getenv("NBA_PTS_LEAGUE_CLAMP_MIN", "104.0"))
PTS_LEAGUE_CLAMP_MAX = float(os.getenv("NBA_PTS_LEAGUE_CLAMP_MAX", "122.0"))

PTS_RECENCY_HALF_LIFE = float(os.getenv("NBA_PTS_RECENCY_HALF_LIFE", "14.0"))
PTS_SINGLE_GAME_WEIGHT_CAP = float(os.getenv("NBA_PTS_SINGLE_GAME_WEIGHT_CAP", "3.5"))

TOTAL_HIST_DAYS = int(os.getenv("NBA_TOTAL_HIST_DAYS", "14"))
TOTAL_LINE_BLEND = float(os.getenv("NBA_TOTAL_LINE_BLEND", "0.35"))
TOTAL_REGRESS_WEIGHT = float(os.getenv("NBA_TOTAL_REGRESS_WEIGHT", "0.25"))
TOTAL_SD_FLOOR = float(os.getenv("NBA_TOTAL_SD_FLOOR", "9.0"))
TOTAL_SD_CEIL = float(os.getenv("NBA_TOTAL_SD_CEIL", "30.0"))
TOTAL_MIN_EDGE_VS_BE = float(os.getenv("NBA_TOTAL_MIN_EDGE_VS_BE", "0.015"))
TOTAL_MIN_PTS_EDGE = float(os.getenv("NBA_TOTAL_MIN_PTS_EDGE", "2.5"))
TOTAL_DEFAULT_PRICE = float(os.getenv("NBA_TOTAL_DEFAULT_PRICE", "-110.0"))
TOTAL_DYNAMIC_VAR_MULT = float(os.getenv("NBA_TOTAL_DYNAMIC_VAR_MULT", "0.35"))
TOTAL_PACE_HOME_W = float(os.getenv("NBA_TOTAL_PACE_HOME_W", "0.6"))
TOTAL_PACE_AWAY_W = float(os.getenv("NBA_TOTAL_PACE_AWAY_W", "0.4"))
MODEL_TOTAL_ANCHOR_W = float(os.getenv("NBA_MODEL_TOTAL_ANCHOR_W", "0.70"))
MIN_TOTAL_EDGE_POINTS = float(os.getenv("NBA_MIN_TOTAL_EDGE_POINTS", "4.0"))
TOTAL_SANITY_MAX_DIFF = float(os.getenv("NBA_TOTAL_SANITY_MAX_DIFF", "12.0"))
TOTAL_RECENCY_GAMES = int(os.getenv("NBA_TOTAL_RECENCY_GAMES", "10"))
W_LAST5 = float(os.getenv("NBA_W_LAST5", "0.50"))
W_PREV5 = float(os.getenv("NBA_W_PREV5", "0.30"))
W_SEASON = float(os.getenv("NBA_W_SEASON", "0.20"))
PACE_FACTOR_MIN = float(os.getenv("NBA_PACE_FACTOR_MIN", "0.92"))
PACE_FACTOR_MAX = float(os.getenv("NBA_PACE_FACTOR_MAX", "1.08"))
TOTAL_INJ_ADJ_MAX = float(os.getenv("NBA_TOTAL_INJ_ADJ_MAX", "8.0"))

# ATS (kept simple)
ATS_SD_PTS = float(os.getenv("NBA_ATS_SD_PTS", "13.5"))
ATS_DEFAULT_PRICE = float(os.getenv("NBA_ATS_DEFAULT_PRICE", "-110.0"))

# blowout-aware margin distribution (mixture normal)
MARGIN_SD_BASE = float(os.getenv("NBA_MARGIN_SD_BASE", str(ATS_SD_PTS)))
BLOWOUT_MIX_W = float(os.getenv("NBA_BLOWOUT_MIX_W", "0.18"))
BLOWOUT_SD_MULT = float(os.getenv("NBA_BLOWOUT_SD_MULT", "1.75"))

# Totals clamp helpers (not env-configured; keep stable)
PACE_MULT_MIN = 0.90
PACE_MULT_MAX = 1.12

# Probabilistic stability
UNCERTAINTY_WINDOW = int(os.getenv("NBA_UNCERTAINTY_WINDOW", "120"))
UNCERTAINTY_SHRINK = float(os.getenv("NBA_UNCERTAINTY_SHRINK", "1.4"))
MAX_ABS_MODEL_SPREAD = float(os.getenv("NBA_MAX_ABS_MODEL_SPREAD", "17.0"))


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


def american_to_implied_prob(ml: float) -> float:
    """Convert American moneyline to implied probability (vig still baked in)."""
    try:
        price = float(ml)
    except Exception:
        return float("nan")

    if price == 0:
        return float("nan")
    if price > 0:
        return 100.0 / (price + 100.0)
    return (-price) / ((-price) + 100.0)


def no_vig_pair(p_home: float, p_away: float) -> Tuple[float, float]:
    """Normalize two implied probabilities into a no-vig pair."""
    try:
        hp = float(p_home)
        ap = float(p_away)
    except Exception:
        return (float("nan"), float("nan"))

    if np.isnan(hp) or np.isnan(ap):
        return (float("nan"), float("nan"))

    s = hp + ap
    if s <= 0:
        return (float("nan"), float("nan"))

    return (hp / s, ap / s)


def _build_form_adjustments(stats_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Return Elo-like adjustments from recent team net ratings."""
    if stats_df is None:
        return {}

    required_cols = {"TEAM_NAME", "ORtg_RECENT", "DRtg_RECENT"}
    if not required_cols.issubset(set(stats_df.columns)):
        return {}

    df = stats_df.copy()
    df["net_recent"] = pd.to_numeric(df["ORtg_RECENT"], errors="coerce") - pd.to_numeric(
        df["DRtg_RECENT"], errors="coerce"
    )

    league_net = float(df["net_recent"].mean(skipna=True))
    if np.isnan(league_net):
        league_net = 0.0

    adjs: Dict[str, float] = {}
    for _, row in df.iterrows():
        team = row.get("TEAM_NAME")
        net = row.get("net_recent")
        if team is None or pd.isna(net):
            continue
        adj = (float(net) - float(league_net)) * float(FORM_ELO_PER_NET)
        adj = _clamp(adj, -FORM_ELO_MAX_ABS, FORM_ELO_MAX_ABS)
        adjs[str(team)] = float(adj)

    return adjs


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


def _anchor_model_total(model_total_raw: float, market_total: float, anchor_w: float) -> float:
    if np.isnan(model_total_raw):
        return float("nan")
    if np.isnan(market_total):
        return float(model_total_raw)
    w = float(_clamp(anchor_w, 0.0, 1.0))
    return float(w * market_total + (1.0 - w) * model_total_raw)


def _recency_weighted_team_totals(
    team_tbl: Optional[pd.DataFrame],
    team: str,
    *,
    n_games: int,
    w_last5: float,
    w_prev5: float,
    w_season: float,
) -> dict:
    if team_tbl is None or team_tbl.empty or not team:
        return {"pts_for": float("nan"), "pts_against": float("nan"), "recent_total": float("nan"), "n": 0}

    df = team_tbl[team_tbl["team"] == team].copy()
    if df.empty:
        return {"pts_for": float("nan"), "pts_against": float("nan"), "recent_total": float("nan"), "n": 0}

    df = df.sort_values("date", ascending=False)
    last_n = df.head(int(n_games))
    last5 = last_n.head(5)
    prev5 = last_n.iloc[5:10]

    def _avg(series: pd.Series) -> float:
        if series is None or series.empty:
            return float("nan")
        return float(series.mean())

    last5_for = _avg(last5["pts_for"])
    last5_against = _avg(last5["pts_against"])
    prev5_for = _avg(prev5["pts_for"])
    prev5_against = _avg(prev5["pts_against"])
    season_for = _avg(df["pts_for"])
    season_against = _avg(df["pts_against"])

    weights = []
    comps_for = []
    comps_against = []

    if not np.isnan(last5_for):
        weights.append(float(w_last5))
        comps_for.append(float(last5_for))
        comps_against.append(float(last5_against))
    if not np.isnan(prev5_for):
        weights.append(float(w_prev5))
        comps_for.append(float(prev5_for))
        comps_against.append(float(prev5_against))
    if not np.isnan(season_for):
        weights.append(float(w_season))
        comps_for.append(float(season_for))
        comps_against.append(float(season_against))

    if not weights or sum(weights) <= 0:
        return {"pts_for": float("nan"), "pts_against": float("nan"), "recent_total": float("nan"), "n": int(len(last_n))}

    w_sum = float(sum(weights))
    weights = [w / w_sum for w in weights]

    pts_for = float(sum(w * v for w, v in zip(weights, comps_for)))
    pts_against = float(sum(w * v for w, v in zip(weights, comps_against)))

    recent_total = float("nan")
    if not last_n.empty:
        recent_total = float((last_n["pts_for"] + last_n["pts_against"]).mean())

    return {
        "pts_for": pts_for,
        "pts_against": pts_against,
        "recent_total": recent_total,
        "n": int(len(last_n)),
    }


def _injury_total_adjustment(inj_home: list[dict], inj_away: list[dict]) -> tuple[float, str]:
    def _adj_for_row(row: dict) -> tuple[float, str]:
        pos = str(row.get("pos", "")).upper()
        role = str(row.get("role", "")).lower()
        mult = float(row.get("status_mult", 0.0))
        impact = float(row.get("impact", 0.0))
        base = float(mult * impact)
        if base <= 0:
            return 0.0, ""

        is_guard = pos in {"PG", "SG", "G", "G-F", "F-G"}
        is_wing = pos in {"SF", "SG", "SF-PF", "F"}
        is_big = pos in {"PF", "C", "PF-C"}

        if is_guard and role == "starter":
            pts = -float(_clamp(1.2 * base, 1.0, 3.0))
            return pts, f"ball-handler:{pts:+.1f}"
        if (is_guard or is_wing) and role == "starter":
            pts = -float(_clamp(1.6 * base, 2.0, 6.0))
            return pts, f"scorer:{pts:+.1f}"
        if is_big and role == "starter":
            pts = float(_clamp(1.2 * base, 1.0, 4.0))
            return pts, f"rim-protector:{pts:+.1f}"

        pts = -float(_clamp(0.8 * base, 0.5, 2.0))
        return pts, f"rotation:{pts:+.1f}"

    total_adj = 0.0
    reasons: list[str] = []
    for row in (inj_home or []) + (inj_away or []):
        adj, reason = _adj_for_row(row or {})
        if adj != 0 and reason:
            total_adj += adj
            reasons.append(reason)

    total_adj = float(_clamp(total_adj, -TOTAL_INJ_ADJ_MAX, TOTAL_INJ_ADJ_MAX))
    reason_text = ";".join(reasons[:4]) if reasons else ""
    return total_adj, reason_text


def _injury_confidence_weight(source: str, injury_data: dict) -> tuple[float, str]:
    src = str(source or "").upper()
    if not injury_data:
        return float(INJURY_LOW_CONF_MULT), "LOW"
    if src == "OFFICIAL":
        return 1.0, "HIGH"
    if src == "ESPN":
        return float(INJURY_MED_CONF_MULT), "MEDIUM"
    return float(INJURY_LOW_CONF_MULT), "LOW"


def _total_pick_from_edge(
    edge_points: float,
    *,
    min_edge: float,
    sanity_fail: bool,
    anchored: bool,
) -> tuple[str, str, list[str]]:
    flags: list[str] = ["TOTAL_ANCHORED"] if anchored else []
    if sanity_fail:
        flags.append("TOTAL_SANITY_FAIL_PASS")
        return "NONE", "No total bet (sanity fail)", flags
    if np.isnan(edge_points):
        flags.append("TOTAL_EDGE_TOO_SMALL")
        return "NONE", "No total bet (missing total/model)", flags
    if edge_points >= float(min_edge):
        flags.append("TOTAL_EDGE_OK")
        return "OVER", "Model PICK TOTAL: OVER", flags
    if edge_points <= -float(min_edge):
        flags.append("TOTAL_EDGE_OK")
        return "UNDER", "Model PICK TOTAL: UNDER", flags
    flags.append("TOTAL_EDGE_TOO_SMALL")
    return "NONE", "No total bet (edge too small)", flags


def _exp_weighted_team_stats(team: str, team_tbl: pd.DataFrame, as_of: date) -> tuple[float, float, float, float, int]:
    """
    Return (off_rating, def_rating, pace_home, pace_away, n) using exponential
    recency weights and caps to prevent any single noisy game from dominating.
    """
    if team_tbl is None or team_tbl.empty:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)

    df = team_tbl[team_tbl["team"] == team].copy()
    if df.empty:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)

    def _days_ago(d: date) -> float:
        try:
            return float((as_of - d).days)
        except Exception:
            return float("inf")

    df["days_ago"] = df["date"].apply(lambda d: _days_ago(d if isinstance(d, date) else as_of))
    df["w"] = np.exp(-df["days_ago"] / max(PTS_RECENCY_HALF_LIFE, 1.0))
    df["w"] = df["w"].clip(upper=PTS_SINGLE_GAME_WEIGHT_CAP)

    if df["w"].sum() <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)

    off = float(np.average(df["pts_for"], weights=df["w"]))
    deff = float(np.average(df["pts_against"], weights=df["w"]))

    pace_home = float(np.average(df[df["is_home"]]["pace_proxy"], weights=df[df["is_home"]]["w"])) if not df[df["is_home"]].empty else float("nan")
    pace_away = float(np.average(df[~df["is_home"]]["pace_proxy"], weights=df[~df["is_home"]]["w"])) if not df[~df["is_home"]].empty else float("nan")

    return (
        _clamp(off, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX),
        _clamp(deff, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX),
        pace_home,
        pace_away,
        int(len(df)),
    )


def _ml_recommendation(p_home: float, mkt_home_p: float, *, min_edge: float = 0.02) -> str:
    if np.isnan(p_home) or np.isnan(mkt_home_p):
        return "NONE"
    edge = p_home - mkt_home_p
    if abs(edge) < float(min_edge):
        return "No ML bet (edge/conf too small)"
    if edge > 0:
        return "Model PICK: HOME ML (strong)"
    return "Model PICK: AWAY ML (strong)"


# ----------------------------
# Historical totals (your repo format)
# ----------------------------
def _lookup_hist(team: str, hist: Dict[str, Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not hist:
        return None
    t = canon_team(team) or team
    if t in hist:
        return hist[t]
    tl = t.lower()
    for k, v in hist.items():
        if str(k).lower() == tl:
            return v
    return None


def _team_hist_total_stats(team: str, hist: Dict[str, Dict[str, float]]) -> Tuple[float, float, int]:
    d = _lookup_hist(team, hist)
    if not d:
        return (float("nan"), float("nan"), 0)
    avg = _safe_float(d.get("avg"))
    sd = _safe_float(d.get("sd"))
    n = int(d.get("n") or 0)
    return (float(avg), float(sd), int(n))


# ----------------------------
# BDL scoring table (key totals fix)
# ----------------------------
def _build_team_scoring_table(days_back: int, as_of_date: date) -> pd.DataFrame:
    """
    Columns: team, pts_for, pts_against, date, is_home, pace_proxy.
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
    pace_anchor = float((PTS_LEAGUE_CLAMP_MIN + PTS_LEAGUE_CLAMP_MAX) / 2.0)
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
            game_date_str = g.get("date") or g.get("status") or ""
            try:
                game_date = datetime.fromisoformat(str(game_date_str).replace("Z", "+00:00")).date()
            except Exception:
                game_date = as_of_date - timedelta(days=1)

            # skip unplayed / not final-ish
            if hs == 0 and av == 0 and (g.get("period", 0) or 0) == 0:
                continue

            home = canon_team(home_name) or str(home_name or "")
            away = canon_team(away_name) or str(away_name or "")
            if not home or not away:
                continue

            pace_proxy = float(hs + av) / max(1.0, float(2.0 * pace_anchor))

            rows.append({
                "team": home,
                "pts_for": float(hs),
                "pts_against": float(av),
                "date": game_date,
                "is_home": True,
                "pace_proxy": pace_proxy,
            })
            rows.append({
                "team": away,
                "pts_for": float(av),
                "pts_against": float(hs),
                "date": game_date,
                "is_home": False,
                "pace_proxy": pace_proxy,
            })

        if not cursor:
            break

    if not rows:
        return pd.DataFrame(columns=["team", "pts_for", "pts_against"])

    return pd.DataFrame(rows, columns=["team", "pts_for", "pts_against", "date", "is_home", "pace_proxy"])


def backfill_nba_elo_state(*, default_elo: float = 1500.0) -> EloState:
    """Gracefully load persisted Elo state when live updates fail."""
    try:
        st = EloState.load(ELO_PATH)
    except Exception:
        st = EloState()

    st.default_elo = float(default_elo)
    return st


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

            game_key = f"{ev.get('id','')}|{ev.get('commence_time','')}|{home}|{away}"
            if hasattr(st, "is_processed") and st.is_processed(game_key):
                continue

            score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))

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

            if hasattr(st, "mark_processed"):
                st.mark_processed(game_key)

        except Exception:
            continue

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)

    # fit calibrators if enough samples
    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            ps_arr = np.array(train_ps, dtype=float)
            ys_arr = np.array(train_ys, dtype=float)

            update_uncertainty(
                "nba",
                ps_arr,
                ys_arr,
                window=UNCERTAINTY_WINDOW,
                min_samples=max(30, int(CAL_MIN_GAMES)),
                market="ML",
            )
            update_prob_calibration(
                "nba",
                ps_arr,
                ys_arr,
                window=PROB_CAL_WINDOW,
                min_samples=PROB_CAL_MIN_GAMES,
            )

            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nba calibration] WARNING: calibration fit failed: {e}")

    return st


def _load_calibrators():
    unc = None
    try:
        data = load_uncertainty("nba") or {}
        unc = float(data.get("uncertainty", float("nan")))
    except Exception:
        unc = None
    mcal = None
    try:
        mcal = load_margin_cal(MARGIN_CAL_PATH)
    except Exception:
        mcal = None
    return unc, mcal


def _load_total_calibration() -> tuple[float, float]:
    try:
        with open(TOTAL_CAL_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
            return float(d.get("a", 0.0)), float(d.get("b", 1.0))
    except Exception:
        return (0.0, 1.0)


def _save_total_calibration(a: float, b: float, n: int) -> None:
    try:
        os.makedirs("results", exist_ok=True)
        with open(TOTAL_CAL_PATH, "w", encoding="utf-8") as f:
            json.dump({"a": float(a), "b": float(b), "n": int(n)}, f, indent=2)
    except Exception:
        pass


def run_daily_nba(game_date_str: str, *, odds_dict: dict, stats_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Objective-only predictions.
    """
    game_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()

    try:
        st = update_elo_from_recent_scores(days_from=ELO_TRAIN_DAYS)
    except Exception as e:
        print(f"[nba] WARNING: Elo update failed ({e}); using backfill state")
        st = backfill_nba_elo_state()
    uncertainty, margin_cal = _load_calibrators()

    form_adjs = _build_form_adjustments(stats_df) if stats_df is not None else {}

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
        print(f"[hist_lines] n_teams={len(hist_lines or {})}")
        if (not hist_lines) and (team_tbl is not None) and (not team_tbl.empty):
            try:
                g = team_tbl.copy()
                g["game_total"] = g["pts_for"] + g["pts_against"]
                by_team = g.groupby("team")["game_total"]
                hist_lines = {}
                for team, s in by_team:
                    s = s.dropna()
                    if len(s) >= 3:
                        hist_lines[str(team)] = {
                            "avg": float(s.mean()),
                            "sd": float(s.std(ddof=0)),
                            "n": int(len(s)),
                        }
                print(f"[hist_lines fallback] n_teams={len(hist_lines or {})}")
            except Exception as e:
                print(f"[hist_lines fallback] failed: {e}")
    except Exception as e:
        print(f"[nba totals] WARNING: failed to build historical totals lines: {e}")
        hist_lines = {}

    hist_avgs: list[float] = []
    hist_sds: list[float] = []
    for v in (hist_lines or {}).values():
        try:
            if v.get("avg") is not None:
                hist_avgs.append(float(v.get("avg")))
            if v.get("sd") is not None and not np.isnan(float(v.get("sd"))):
                hist_sds.append(float(v.get("sd")))
        except Exception:
            continue

    hist_league_avg = float(np.mean(hist_avgs)) if hist_avgs else float("nan")
    hist_league_sd = float(np.mean(hist_sds)) if hist_sds else float("nan")

    total_cal_a, total_cal_b = _load_total_calibration()
    total_calibration_samples: list[tuple[float, float]] = []

    if not np.isnan(hist_league_avg):
        if np.isnan(league_avg_total):
            league_avg_total = hist_league_avg
        else:
            league_avg_total = float(0.5 * league_avg_total + 0.5 * hist_league_avg)

    if not np.isnan(hist_league_sd):
        if np.isnan(league_sd_total):
            league_sd_total = hist_league_sd
        else:
            league_sd_total = float(0.5 * league_sd_total + 0.5 * hist_league_sd)

    injury_source = "OFFICIAL"
    try:
        injury_data = _fetch_from_official_nba()
    except Exception as e:
        print(f"[nba injuries] WARNING: official NBA injury page failed ({e}); falling back to ESPN")
        injury_source = "ESPN"
        try:
            injury_data = _fetch_from_espn()
        except Exception as e2:
            print(f"[nba injuries] WARNING: ESPN injuries failed too ({e2}); using empty injuries")
            injury_data = {}
            injury_source = "NONE"

    injury_conf_weight, injury_confidence = _injury_confidence_weight(injury_source, injury_data)

    rows = []

    # IMPORTANT: odds_dict may store teams in the KEY (tuple), not in oi
    for matchup, oi in (odds_dict or {}).items():
        home_raw = (oi or {}).get("home")
        away_raw = (oi or {}).get("away")
        if (home_raw is None or away_raw is None) and isinstance(matchup, (tuple, list)) and len(matchup) == 2:
            home_raw, away_raw = matchup[0], matchup[1]

        home = canon_team(str(home_raw)) if home_raw is not None else None
        away = canon_team(str(away_raw)) if away_raw is not None else None
        if not home or not away:
            continue

        event_id = (oi or {}).get("event_id")
        game_key = build_game_key(event_id, game_date_str, home, away)

        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        p_home_imp = american_to_implied_prob(home_ml)
        p_away_imp = american_to_implied_prob(away_ml)
        mkt_home_p, _ = no_vig_pair(p_home_imp, p_away_imp)

        inj_list_home = build_injury_list_for_team_nba(home, injury_data)
        inj_list_away = build_injury_list_for_team_nba(away, injury_data)
        inj_detail_home = build_injury_detail_list_for_team_nba(home, injury_data)
        inj_detail_away = build_injury_detail_list_for_team_nba(away, injury_data)
        inj_pts_home = (
            _clamp(injury_adjustment_points(inj_list_home), -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)
            * INJ_DAMP
            * injury_conf_weight
        )
        inj_pts_away = (
            _clamp(injury_adjustment_points(inj_list_away), -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)
            * INJ_DAMP
            * injury_conf_weight
        )
        inj_elo_home = float(inj_pts_home) * float(ELO_PER_POINT)
        inj_elo_away = float(inj_pts_away) * float(ELO_PER_POINT)

        inj_total_adj, inj_total_reason = _injury_total_adjustment(inj_detail_home, inj_detail_away)
        if injury_source in {"NONE", "UNKNOWN"}:
            inj_total_adj = 0.0
            inj_total_reason = "INJURY_UNKNOWN"
        else:
            inj_total_adj = float(inj_total_adj) * float(injury_conf_weight)
            if injury_confidence in {"LOW", "MEDIUM"}:
                inj_total_reason = f"{inj_total_reason}|INJ_CONF_{injury_confidence}"

        form_home = float(form_adjs.get(home, 0.0))
        form_away = float(form_adjs.get(away, 0.0))

        eh = float(st.get(home)) + inj_elo_home + form_home
        ea = float(st.get(away)) + inj_elo_away + form_away

        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        p_home = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))

        resid_sd = (
            float(uncertainty) if uncertainty is not None and not math.isnan(uncertainty) else 0.06
        )
        shrink = 1.0 + float(UNCERTAINTY_SHRINK) * float(_clamp(resid_sd, 0.0, 0.50))
        p_home = 0.5 + (p_home - 0.5) / shrink

        edge_home = float(p_home - mkt_home_p) if not (np.isnan(p_home) or np.isnan(mkt_home_p)) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")
        ml_reco = _ml_recommendation(p_home, mkt_home_p, min_edge=MIN_ML_EDGE)

        elo_diff = (eh + HOME_ADV) - ea
        model_spread_home = float(-float(elo_diff) / 16.0)
        if margin_cal is not None:
            try:
                calibrated_margin = float(margin_cal.predict(elo_diff))
                model_spread_home = float(_clamp(-calibrated_margin, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD))
            except Exception:
                model_spread_home = float(_clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD))
        else:
            model_spread_home = float(_clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD))
        mu_margin_home = float(-model_spread_home)

        h_off, h_def, h_pace_home, h_pace_away, h_n_w = _exp_weighted_team_stats(home, team_tbl, as_of)
        a_off, a_def, a_pace_home, a_pace_away, a_n_w = _exp_weighted_team_stats(away, team_tbl, as_of)

        league_anchor = league_avg_total if not np.isnan(league_avg_total) else (2.0 * league_pts if not np.isnan(league_pts) else 224.0)

        recent_home = _recency_weighted_team_totals(
            team_tbl,
            home,
            n_games=TOTAL_RECENCY_GAMES,
            w_last5=W_LAST5,
            w_prev5=W_PREV5,
            w_season=W_SEASON,
        )
        recent_away = _recency_weighted_team_totals(
            team_tbl,
            away,
            n_games=TOTAL_RECENCY_GAMES,
            w_last5=W_LAST5,
            w_prev5=W_PREV5,
            w_season=W_SEASON,
        )

        home_pts_for = recent_home.get("pts_for", float("nan"))
        home_pts_allowed = recent_home.get("pts_against", float("nan"))
        away_pts_for = recent_away.get("pts_for", float("nan"))
        away_pts_allowed = recent_away.get("pts_against", float("nan"))

        home_exp = np.nanmean([home_pts_for, away_pts_allowed])
        away_exp = np.nanmean([away_pts_for, home_pts_allowed])

        weighted_team_offense = np.nanmean([h_off, a_off])
        weighted_opponent_defense = np.nanmean([h_def, a_def])
        fallback_base_total = weighted_team_offense + weighted_opponent_defense

        base_total = home_exp + away_exp if not (np.isnan(home_exp) or np.isnan(away_exp)) else fallback_base_total
        if np.isnan(base_total) and not np.isnan(league_anchor):
            base_total = float(league_anchor)

        model_total_raw = float(base_total + inj_total_adj)

        pace_factor = float("nan")
        pace_reason = "pace=neutral (missing data)"
        if not np.isnan(league_anchor) and league_anchor > 0:
            home_recent_total = recent_home.get("recent_total", float("nan"))
            away_recent_total = recent_away.get("recent_total", float("nan"))
            if not (np.isnan(home_recent_total) or np.isnan(away_recent_total)):
                pace_factor = (home_recent_total + away_recent_total) / float(league_anchor)
                pace_factor = float(_clamp(pace_factor, PACE_FACTOR_MIN, PACE_FACTOR_MAX))
                pace_reason = "pace=recent_totals_vs_league"
            else:
                pace_factor = 1.0
        if np.isnan(pace_factor):
            pace_factor = 1.0

        model_total_raw = float(model_total_raw * pace_factor)

        pace_proxy = _pace_proxy_from_total(model_total_raw, league_anchor) if not np.isnan(league_anchor) else 1.0
        margin_sd = float(MARGIN_SD_BASE) * float(_clamp(0.92 + 0.20 * pace_proxy, 0.85, 1.15))

        win_prob_home = _mix_norm_win_prob(mu_margin_home, margin_sd)
        blowout_prob_abs15 = _mix_norm_tail_prob_abs_ge(15.0, mu_margin_home, margin_sd)
        blowout_prob_abs25 = _mix_norm_tail_prob_abs_ge(25.0, mu_margin_home, margin_sd)

        margin_ci95_low, margin_ci95_high = _normal_ci(mu_margin_home, margin_sd, z=1.96)
        margin_ci80_low, margin_ci80_high = _normal_ci(mu_margin_home, margin_sd, z=1.2816)

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

        if not np.isnan(hist_base):
            w = float(_clamp(TOTAL_LINE_BLEND, 0.20, 0.60))
            model_total_raw = float((1.0 - w) * model_total_raw + w * hist_base)

        league_anchor_total = league_anchor
        if np.isnan(league_anchor_total) and not np.isnan(league_pts):
            league_anchor_total = float(2.0 * league_pts)

        if not np.isnan(model_total_raw) and not np.isnan(league_anchor_total):
            model_total_raw = float((1.0 - TOTAL_REGRESS_WEIGHT) * model_total_raw + TOTAL_REGRESS_WEIGHT * league_anchor_total)

        if not np.isnan(model_total_raw):
            model_total_raw = float(total_cal_a + total_cal_b * model_total_raw)

        if not np.isnan(model_total_raw) and not np.isnan(total_points):
            total_calibration_samples.append((model_total_raw, total_points))

        sd = float("nan")
        if not np.isnan(h_sd) and not np.isnan(a_sd):
            sd = 0.5 * (h_sd + a_sd)
        elif not np.isnan(h_sd):
            sd = h_sd
        elif not np.isnan(a_sd):
            sd = a_sd
        else:
            sd = league_sd_total

        sd = float(_clamp(sd, TOTAL_SD_FLOOR, TOTAL_SD_CEIL))

        model_total_final = _anchor_model_total(model_total_raw, total_points, MODEL_TOTAL_ANCHOR_W)
        anchored = not np.isnan(total_points) and not np.isnan(model_total_raw)

        total_ci95_low, total_ci95_high = _normal_ci(model_total_final, sd, z=1.96)
        total_ci80_low, total_ci80_high = _normal_ci(model_total_final, sd, z=1.2816)

        total_pick_side = "NONE"
        total_pick = "NO BET"
        total_edge_points_raw = float("nan")
        total_edge_points_final = float("nan")
        total_edge_vs_be = float("nan")
        total_reco = "No total bet (missing total/model)"
        total_flags: list[str] = []

        if not np.isnan(model_total_raw) and not np.isnan(total_points):
            total_edge_points_raw = float(model_total_raw - total_points)
        if not np.isnan(model_total_final) and not np.isnan(total_points):
            total_edge_points_final = float(model_total_final - total_points)

        sanity_fail = False
        if not np.isnan(model_total_raw) and not np.isnan(total_points):
            if abs(float(model_total_raw - total_points)) > float(TOTAL_SANITY_MAX_DIFF):
                sanity_fail = True

        if not np.isnan(model_total_final) and not np.isnan(total_points) and sd > 0:
            z = (model_total_final - total_points) / sd
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

            if edge_over >= edge_under:
                total_pick_side = "OVER"
                total_edge_vs_be = float(edge_over)
            else:
                total_pick_side = "UNDER"
                total_edge_vs_be = float(edge_under)

        total_pick_side, total_reco, total_flags = _total_pick_from_edge(
            total_edge_points_final,
            min_edge=MIN_TOTAL_EDGE_POINTS,
            sanity_fail=sanity_fail,
            anchored=anchored,
        )

        if total_pick_side in {"OVER", "UNDER"}:
            total_pick = total_pick_side
        total_reco = total_reco

        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=ATS_DEFAULT_PRICE)
        spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) and not np.isnan(model_spread_home) else float("nan")
        p_home_cover = float(_clamp(_phi((spread_edge_home / ATS_SD_PTS)), 0.001, 0.999)) if not np.isnan(spread_edge_home) else float("nan")

        market_home_delta = (
            float(mkt_home_p - p_home_imp)
            if not (np.isnan(mkt_home_p) or np.isnan(p_home_imp))
            else np.nan
        )

        rows.append(
            {
                "game_key": game_key,
                "event_id": event_id or "",
                "date": game_date_str,
                "home": home,
                "away": away,
                "model_home_prob": float(p_home),
                "model_home_prob_raw": float(p_home),
                "margin_calibrated": bool(margin_cal is not None),
                "market_home_imp": float(p_home_imp) if not np.isnan(p_home_imp) else np.nan,
                "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
                "market_home_delta": float(market_home_delta) if not np.isnan(market_home_delta) else np.nan,
                "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
                "edge_away": float(edge_away) if not np.isnan(edge_away) else np.nan,
                "model_uncertainty": float(resid_sd),
                "ml_recommendation": str(ml_reco),
                "home_ml": float(home_ml) if not np.isnan(home_ml) else np.nan,
                "away_ml": float(away_ml) if not np.isnan(away_ml) else np.nan,
                "inj_points_home": float(inj_pts_home),
                "inj_points_away": float(inj_pts_away),
                "injury_confidence": str(injury_confidence),
                "injury_confidence_weight": float(injury_conf_weight),
                "injury_source": str(injury_source),
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
                "pace_factor": float(pace_factor) if not np.isnan(pace_factor) else np.nan,
                "pace_reason": str(pace_reason),
                "inj_total_adj": float(inj_total_adj),
                "inj_total_reason": str(inj_total_reason),
                "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
                "total_over_price": float(total_over_price),
                "total_under_price": float(total_under_price),
                "model_total_raw": float(model_total_raw) if not np.isnan(model_total_raw) else np.nan,
                "model_total_final": float(model_total_final) if not np.isnan(model_total_final) else np.nan,
                "market_total_used": float(total_points) if not np.isnan(total_points) else np.nan,
                "model_total": float(model_total_final) if not np.isnan(model_total_final) else np.nan,
                "total_sd": float(sd) if not np.isnan(sd) else np.nan,
                "total_ci95_low": float(total_ci95_low),
                "total_ci95_high": float(total_ci95_high),
                "total_ci80_low": float(total_ci80_low),
                "total_ci80_high": float(total_ci80_high),
                "total_edge_points_raw": float(total_edge_points_raw) if not np.isnan(total_edge_points_raw) else np.nan,
                "total_edge_points_final": float(total_edge_points_final) if not np.isnan(total_edge_points_final) else np.nan,
                "total_edge_points": float(total_edge_points_final) if not np.isnan(total_edge_points_final) else np.nan,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_pick_side": str(total_pick_side),
                "total_pick": str(total_pick),
                "total_recommendation": str(total_reco),
                "total_decision_flags": ",".join(total_flags),
                "hist_total_home_avg": float(h_avg) if not np.isnan(h_avg) else np.nan,
                "hist_total_away_avg": float(a_avg) if not np.isnan(a_avg) else np.nan,
                "hist_total_home_n": int(h_n),
                "hist_total_away_n": int(a_n),
            }
        )

    df = pd.DataFrame(rows)

    # Update totals calibration vs. observed market totals when enough samples exist
    if len(total_calibration_samples) >= 5:
        try:
            mt, mk = zip(*total_calibration_samples)
            mt_arr = np.array(mt, dtype=float)
            mk_arr = np.array(mk, dtype=float)
            A = np.column_stack([np.ones_like(mt_arr), mt_arr])
            coef, _, _, _ = np.linalg.lstsq(A, mk_arr, rcond=None)
            _save_total_calibration(float(coef[0]), float(coef[1]), len(mt_arr))
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
    """
    Wrapper used by current_of_sports_betting_algorithm.py
    """
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")
    return run_daily_nba(str(date_in), odds_dict=(odds_dict or {}))
