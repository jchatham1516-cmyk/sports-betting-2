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
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.historical_totals import build_team_historical_total_lines
from sports.nba.injuries import (
    fetch_official_nba_injuries,
    build_injury_list_for_team_nba,
    injury_adjustment_points,
)

# NEW: probability + margin calibrators
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

# Outcome distribution / variance modeling
# We model point margin as a mixture of two normals:
#   - "core" game (typical variance)
#   - "blowout" game (fatter tails)
# This keeps everything objective + allows blowouts without requiring SciPy.
MARGIN_SD_BASE = float(os.getenv("NBA_MARGIN_SD_BASE", str(ATS_SD_PTS)))
BLOWOUT_MIX_W = float(os.getenv("NBA_BLOWOUT_MIX_W", "0.18"))       # weight on blowout component
BLOWOUT_SD_MULT = float(os.getenv("NBA_BLOWOUT_SD_MULT", "1.75"))   # blowout sd multiplier

# Optional: blend in external advanced metrics if you provide a CSV.
# Expected columns: team,srs,pace,shot_quality,per (any subset ok)
ADV_METRICS_CSV = os.getenv("NBA_ADV_METRICS_CSV", "results/nba_advanced_metrics.csv")
ADV_BLEND_W = float(os.getenv("NBA_ADV_BLEND_W", "0.25"))  # how much advanced metrics can shift mean margin

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


def _load_adv_metrics() -> pd.DataFrame:
    """Load optional team-level advanced metrics from a local CSV (no web/sentiment)."""
    try:
        if not ADV_METRICS_CSV or not os.path.exists(ADV_METRICS_CSV):
            return pd.DataFrame()
        df = pd.read_csv(ADV_METRICS_CSV)
        if df is None or df.empty:
            return pd.DataFrame()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "team" not in df.columns:
            return pd.DataFrame()
        df["team"] = df["team"].astype(str).map(lambda x: canon_team(x) or x.strip())
        keep = ["team", "srs", "pace", "shot_quality", "per"]
        out = df[[c for c in keep if c in df.columns]].copy()
        for c in out.columns:
            if c != "team":
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out
    except Exception:
        return pd.DataFrame()


def _pace_proxy_from_totals(team_total: float, league_total: float) -> float:
    """Pace proxy: relative scoring environment (derived from scores only)."""
    try:
        if league_total <= 1e-6 or np.isnan(team_total) or np.isnan(league_total):
            return 1.0
        return float(_clamp(team_total / league_total, 0.85, 1.20))
    except Exception:
        return 1.0


def _blend_margin_with_adv(mu_margin_pts: float, home: str, away: str, adv_df: pd.DataFrame) -> float:
    """
    Blend mean margin with optional advanced metrics.
    SRS acts as a point-margin prior; other metrics provide small tilts.
    """
    try:
        if adv_df is None or adv_df.empty:
            return float(mu_margin_pts)

        h = canon_team(home) or home
        a = canon_team(away) or away

        hrow = adv_df[adv_df["team"] == h]
        arow = adv_df[adv_df["team"] == a]
        if hrow.empty or arow.empty:
            return float(mu_margin_pts)

        def g(row, col):
            return _safe_float(row.iloc[0].get(col), default=np.nan)

        srs_diff = g(hrow, "srs") - g(arow, "srs")
        srs_diff = float(srs_diff) if not np.isnan(srs_diff) else 0.0

        pace_diff = g(hrow, "pace") - g(arow, "pace")
        pace_diff = float(pace_diff) if not np.isnan(pace_diff) else 0.0

        sq_diff = g(hrow, "shot_quality") - g(arow, "shot_quality")
        sq_diff = float(sq_diff) if not np.isnan(sq_diff) else 0.0

        per_diff = g(hrow, "per") - g(arow, "per")
        per_diff = float(per_diff) if not np.isnan(per_diff) else 0.0

        adv_pts = (
            1.00 * srs_diff
            + 0.06 * pace_diff
            + 0.35 * sq_diff
            + 0.04 * per_diff
        )

        w = float(_clamp(ADV_BLEND_W, 0.0, 0.6))
        return float((1.0 - w) * float(mu_margin_pts) + w * (float(mu_margin_pts) + adv_pts))
    except Exception:
        return float(mu_margin_pts)


def _mix_norm_win_prob(mu: float, sd: float) -> float:
    """Mixture-of-normals win prob for margin ~ w*N(mu,(k sd)^2)+(1-w)*N(mu,sd^2)."""
    if sd <= 1e-9 or np.isnan(mu) or np.isnan(sd):
        return float("nan")
    w = float(_clamp(BLOWOUT_MIX_W, 0.0, 0.49))
    k = float(_clamp(BLOWOUT_SD_MULT, 1.0, 3.0))
    z1 = mu / sd
    z2 = mu / (sd * k)
    return float(_clamp((1.0 - w) * _phi(z1) + w * _phi(z2), 0.001, 0.999))


def _mix_norm_tail_prob_abs_ge(thresh: float, mu: float, sd: float) -> float:
    """P(|X|>=thresh) under the same mixture (objective blowout probability)."""
    if sd <= 1e-9 or np.isnan(mu) or np.isnan(sd):
        return float("nan")
    w = float(_clamp(BLOWOUT_MIX_W, 0.0, 0.49))
    k = float(_clamp(BLOWOUT_SD_MULT, 1.0, 3.0))

    def tail(sd0: float) -> float:
        z_hi = (thresh - mu) / sd0
        z_lo = (-thresh - mu) / sd0
        return float((1.0 - _phi(z_hi)) + _phi(z_lo))

    return float(_clamp((1.0 - w) * tail(sd) + w * tail(sd * k), 0.0, 1.0))


def _normal_ci(mu: float, sd: float, z: float = 1.96) -> Tuple[float, float]:
    if np.isnan(mu) or np.isnan(sd) or sd <= 0:
        return (float("nan"), float("nan"))
    return (float(mu - z * sd), float(mu + z * sd))


def _parse_iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _calc_days_off(last_played: Optional[date], game_date: date) -> Optional[int]:
    if last_played is None:
        return None
    return int((game_date - last_played).days)


def _rest_elo(days_off: Optional[int]) -> float:
    if days_off is None:
        return 0.0
    if days_off <= 0:
        return float(SHORT_REST_PENALTY_ELO)
    if days_off == 1:
        return 0.0
    if days_off >= 3:
        return float(NORMAL_REST_BONUS_ELO)
    return 0.0


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


def _phi(z: float) -> float:
    # standard normal CDF
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


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


def _ml_recommendation(p_home: float, mkt_home_p: float, *, min_edge: float = 0.02) -> str:
    if np.isnan(p_home) or np.isnan(mkt_home_p):
        return "NONE"
    edge = p_home - mkt_home_p
    if abs(edge) < float(min_edge):
        return "No ML bet (edge/conf too small)"
    if edge > 0:
        return "Model PICK: HOME ML (strong)"
    return "Model PICK: AWAY ML (strong)"


def _cover_prob_from_edge(spread_edge_home: float, *, sd_pts: float = 13.5) -> float:
    """
    If spread_edge_home = market_spread - model_spread_home
      positive means model likes HOME more than market (market giving home more points)
    We approximate cover probability with Normal(0, sd_pts) on ATS error.
    """
    if np.isnan(spread_edge_home) or sd_pts <= 0:
        return float("nan")
    z = float(spread_edge_home) / float(sd_pts)
    return float(_clamp(_phi(z), 0.001, 0.999))


def _ats_pick_and_edge(p_home_cover: float, spread_price: float) -> Tuple[str, float, float, float]:
    if np.isnan(p_home_cover):
        return ("NONE", float("nan"), float("nan"), float("nan"))
    be = _breakeven_prob_from_american(spread_price)
    if np.isnan(be):
        be = 0.5238
    p_away_cover = 1.0 - p_home_cover
    edge_home = p_home_cover - be
    edge_away = p_away_cover - be
    if edge_home >= edge_away:
        return ("HOME", float(p_home_cover), float(edge_home), float(be))
    return ("AWAY", float(p_away_cover), float(edge_away), float(be))


def _ats_strength_label(edge_vs_be: float) -> str:
    if np.isnan(edge_vs_be):
        return ""
    if edge_vs_be >= 0.06:
        return "strong"
    if edge_vs_be >= 0.03:
        return "medium"
    if edge_vs_be >= 0.02:
        return "lean"
    return "weak"


def _ats_reco(side: str, strength: str) -> str:
    if side == "NONE":
        return "No ATS bet (missing spread/model)"
    if not strength:
        strength = "lean"
    return f"Model PICK ATS: {side} ({strength})"


def _total_pick_and_edge(p_over: float, over_price: float, under_price: float) -> Tuple[str, float, float, float]:
    if np.isnan(p_over):
        return ("NONE", float("nan"), float("nan"), float("nan"))
    be_over = _breakeven_prob_from_american(over_price)
    be_under = _breakeven_prob_from_american(under_price)
    if np.isnan(be_over):
        be_over = 0.5238
    if np.isnan(be_under):
        be_under = 0.5238
    p_under = 1.0 - p_over
    edge_over = p_over - be_over
    edge_under = p_under - be_under
    if edge_over >= edge_under:
        return ("OVER", float(p_over), float(edge_over), float(be_over))
    return ("UNDER", float(p_under), float(edge_under), float(be_under))


def _total_reco(side: str, edge_vs_be: float) -> str:
    if side == "NONE":
        return "No total bet (missing total/model)"
    if np.isnan(edge_vs_be):
        return "No total bet (missing edge)"
    if edge_vs_be >= TOTAL_MIN_EDGE_VS_BE:
        return f"Model PICK TOTAL: {side}"
    return "No total bet (edge too small)"


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
        "start_date": start_date,
        "end_date": end_date,
        "seasons[]": season_year,
        "per_page": 100,
    }

    rows = []
    cursor = None
    for _ in range(20):  # hard cap
        if cursor:
            params["cursor"] = cursor

        games_json = bdl_get("games", params=params, api_key=api_key)
        games = games_json.get("data", []) if isinstance(games_json, dict) else []
        meta = games_json.get("meta", {}) if isinstance(games_json, dict) else {}
        cursor = meta.get("next_cursor")

        for g in games:
            try:
                if g.get("status") != "Final":
                    continue
                hs = g.get("home_team_score")
                av = g.get("visitor_team_score")
                if hs is None or av is None:
                    continue
                hs = float(hs)
                av = float(av)
            except Exception:
                continue

            home_name = None
            away_name = None
            try:
                home_name = g.get("home_team", {}).get("full_name")
                away_name = g.get("visitor_team", {}).get("full_name")
            except Exception:
                pass

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

        if sub.empty:
            return (None, None)

        pf = sub["pts_for"].mean()
        pa = sub["pts_against"].mean()
        if pf is None or pa is None or np.isnan(pf) or np.isnan(pa):
            return (None, None)
        return (float(pf), float(pa))

    h_pf, h_pa = _team_means(home)
    a_pf, a_pa = _team_means(away)

    # Offense strength ~ points for vs league; defense strength ~ points allowed vs league
    def _strength(pf: Optional[float], pa: Optional[float]) -> Tuple[float, float]:
        off = 1.0
        deff = 1.0
        if pf is not None and not np.isnan(pf):
            off = float(pf) / float(league_pts)
        if pa is not None and not np.isnan(pa):
            deff = float(pa) / float(league_pts)
        return (off, deff)

    h_off, h_def = _strength(h_pf, h_pa)
    a_off, a_def = _strength(a_pf, a_pa)

    # expected points per team: blend offense vs opponent defense
    exp_home = float(league_pts) * (0.55 * h_off + 0.45 * a_def)
    exp_away = float(league_pts) * (0.55 * a_off + 0.45 * h_def)

    # regression toward league to avoid extremes when sample small
    exp_home = float((1.0 - PTS_REGRESS) * exp_home + PTS_REGRESS * float(league_pts))
    exp_away = float((1.0 - PTS_REGRESS) * exp_away + PTS_REGRESS * float(league_pts))

    exp_home = _clamp(exp_home, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX)
    exp_away = _clamp(exp_away, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX)

    return (float(exp_home), float(exp_away), float(exp_home + exp_away))


def _team_means(team_tbl: pd.DataFrame, team: str) -> Tuple[float, float, float]:
    if team_tbl is None or team_tbl.empty:
        return (float("nan"), float("nan"), float("nan"))
    t = canon_team(team) or team
    sub = team_tbl[team_tbl["team"] == t]
    if sub.empty:
        return (float("nan"), float("nan"), float("nan"))
    pf = float(sub["pts_for"].mean())
    pa = float(sub["pts_against"].mean())
    net = pf - pa
    return (pf, pa, net)


def _build_last_game_date_map(games: pd.DataFrame) -> Dict[str, date]:
    last_played: Dict[str, date] = {}
    if games is None or games.empty:
        return last_played

    for _, g in games.iterrows():
        try:
            d = g.get("date")
            if isinstance(d, str):
                # accept YYYY-MM-DD or ISO with time
                d = d.split("T")[0]
                dt = _parse_iso_date(d)
            elif isinstance(d, datetime):
                dt = d.date()
            else:
                continue
        except Exception:
            continue

        ht = canon_team(str(g.get("home_team", ""))) or str(g.get("home_team", ""))
        at = canon_team(str(g.get("visitor_team", ""))) or str(g.get("visitor_team", ""))

        for t in (ht, at):
            if not t:
                continue
            prev = last_played.get(t)
            if prev is None or dt > prev:
                last_played[t] = dt

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
            out[str(team)] = {"elo": float(elo), "mult": float(mult)}
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

    for ev in events:
        try:
            if not ev.get("completed"):
                continue
            home = canon_team(ev.get("home_team", ""))
            away = canon_team(ev.get("away_team", ""))
            hs = ev.get("home_score")
            as_ = ev.get("away_score")
            if home is None or away is None:
                continue
            if hs is None or as_ is None:
                continue

            eh = st.get(home)
            ea = st.get(away)
            p_home = elo_win_prob(eh + HOME_ADV, ea)
            y = 1.0 if float(hs) > float(as_) else 0.0

            train_ps.append(float(p_home))
            train_ys.append(float(y))

            st.set(home, elo_update(eh + HOME_ADV, ea, y, k=ELO_K))
            st.set(away, elo_update(ea, eh + HOME_ADV, 1.0 - y, k=ELO_K))
        except Exception:
            continue

    # Fit/save Platt if enough games
    if len(train_ps) >= CAL_MIN_GAMES:
        try:
            cal = fit_platt(np.array(train_ps, dtype=float), np.array(train_ys, dtype=float))
            save_platt(PLATT_PATH, cal)
        except Exception:
            pass

        # Fit/save margin cal using last-N predictions if present
        try:
            # Using historical predictions CSVs, not here
            pass
        except Exception:
            pass

    st.save(ELO_PATH)
    return st


def load_nba_calibrator():
    platt = load_platt(PLATT_PATH)
    margin = load_margin_cal(MARGIN_CAL_PATH)
    return platt, margin


def run_daily_nba(game_date_str: str, odds_dict: dict) -> pd.DataFrame:
    """
    Objective-only model:
    - Elo (strength) + injuries + rest/fatigue + recent form
    - Expected margin distribution (mixture normals) => win prob + blowout tail probs
    - Totals model from recent scoring + historical market totals lines
    - Outputs include confidence intervals/ranges for margin and total
    """
    game_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()

    # Update Elo first (keeps your pipeline behavior)
    st = update_elo_from_recent_scores(days_from=ELO_TRAIN_DAYS)

    platt, margin_cal = load_nba_calibrator()

    # Scoring table for totals/pace proxy
    as_of = game_date - timedelta(days=1)
    team_tbl = _build_team_scoring_table(PTS_LOOKBACK_DAYS, as_of)
    adv_df = _load_adv_metrics()

    # League average points/team from scoring table
    league_pts = float("nan")
    league_avg_total = float("nan")
    league_sd_total = float("nan")
    if team_tbl is not None and not team_tbl.empty:
        # league_pts = avg points for per team-game
        league_pts = float(team_tbl["pts_for"].mean())
        league_avg_total = float((team_tbl["pts_for"] + team_tbl["pts_against"]).mean())
        league_sd_total = float((team_tbl["pts_for"] + team_tbl["pts_against"]).std(ddof=0))

    # Historical market totals lines (optional signal)
    hist_lines = build_team_historical_total_lines(
        sport_key="basketball_nba",
        days_back=TOTAL_HIST_DAYS,
        sample_step_days=1,
        max_requests=60,
        max_event_odds_calls=35,
    )
    home_in = hist_lines.get("home", {})
    away_in = hist_lines.get("away", {})

    # Injuries (objective)
    try:
        injury_data = fetch_official_nba_injuries()
    except Exception:
        injury_data = []

    # Build a best-effort last-played map from the scoring table window (approx)
    # If you later add an explicit schedule table, plug it in here.
    last_played = {}  # type: Dict[str, date]

    rows = []

    # Iterate games from odds_dict (already normalized upstream)
    for key, oi in (odds_dict or {}).items():
        try:
            home = canon_team(str(oi.get("home"))) or str(oi.get("home"))
            away = canon_team(str(oi.get("away"))) or str(oi.get("away"))
            if not home or not away:
                continue
        except Exception:
            continue

        # Elo base
        eh = st.get(home)
        ea = st.get(away)

        # Injuries => points => Elo shift
        inj_list_home = build_injury_list_for_team_nba(home, injury_data)
        inj_list_away = build_injury_list_for_team_nba(away, injury_data)

        inj_pts_home = _clamp(injury_adjustment_points(inj_list_home), -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)
        inj_pts_away = _clamp(injury_adjustment_points(inj_list_away), -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        inj_pts_home *= float(INJ_DAMP)
        inj_pts_away *= float(INJ_DAMP)

        inj_elo_home = float(inj_pts_home) * float(ELO_PER_POINT)
        inj_elo_away = float(inj_pts_away) * float(ELO_PER_POINT)

        # Rest/fatigue
        h_last = last_played.get(home)
        a_last = last_played.get(away)
        h_days = _calc_days_off(h_last, game_date)
        a_days = _calc_days_off(a_last, game_date)

        h_rest_elo = _rest_elo(h_days)
        a_rest_elo = _rest_elo(a_days)

        # Recent form (objective)
        form = _recent_form_adjustments()
        form_h = form.get(home, {})
        form_a = form.get(away, {})
        form_elo_home = float(form_h.get("elo", 1500.0)) - 1500.0
        form_elo_away = float(form_a.get("elo", 1500.0)) - 1500.0
        form_elo_home = float(_clamp(form_elo_home, -FORM_ELO_CLAMP, FORM_ELO_CLAMP))
        form_elo_away = float(_clamp(form_elo_away, -FORM_ELO_CLAMP, FORM_ELO_CLAMP))

        # Effective Elo
        eh_eff = float(eh) + float(inj_elo_home) + float(h_rest_elo) + float(form_elo_home)
        ea_eff = float(ea) + float(inj_elo_away) + float(a_rest_elo) + float(form_elo_away)

        # Market ML -> no-vig
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        mkt_home_p = float("nan")
        mkt_away_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, mkt_away_p = _no_vig_probs(home_ml, away_ml)

        # Model win prob from Elo
        p_home_raw = elo_win_prob(eh_eff + HOME_ADV, ea_eff)
        # Apply Platt calibration if available
        if platt is not None:
            try:
                p_home = float(platt.predict_proba(p_home_raw))
            except Exception:
                p_home = float(p_home_raw)
        else:
            p_home = float(p_home_raw)

        # Compress extremes a bit (objective regularization)
        p_home = float(_clamp(0.5 + (p_home - 0.5) * BASE_COMPRESS, 0.01, 0.99))

        # Missing Elo handling: soften toward neutral + tiny market blend
        if (eh == 1500.0 and ea == 1500.0) and not np.isnan(mkt_home_p):
            neutralized = 0.5 + (p_home - 0.5) * float(_clamp(MISSING_ELO_SHRINK, 0.0, 1.0))
            if not np.isnan(mkt_home_p):
                w = float(_clamp(MISSING_ELO_MARKET_BLEND, 0.0, 0.85))
                p_home = float(_clamp((1.0 - w) * neutralized + w * float(mkt_home_p), 0.01, 0.99))
            else:
                p_home = neutralized

        # Spread (market convention: negative = home favored)
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

        # TOTALS (model_total now varies; also fixed gating bug that prevented totals picks)
        home_avg, home_sd = _team_line_avg_sd(home, home_in)
        away_avg, away_sd = _team_line_avg_sd(away, away_in)

        exp_home, exp_away, exp_total = _expected_points_total(home, away, league_pts, team_tbl)
        # ----- Objective distribution over score differentials (blowout-aware) -----
        # model_spread_home uses market convention (negative = home favored).
        mu_margin_home = float(-model_spread_home) if not np.isnan(model_spread_home) else float("nan")

        # Pace proxy affects variance slightly (higher pace -> more variance).
        league_anchor_total_tmp = league_avg_total
        if np.isnan(league_anchor_total_tmp):
            league_anchor_total_tmp = float(2.0 * league_pts) if league_pts > 0 else float("nan")
        pace_proxy = _pace_proxy_from_totals(float(exp_total), float(league_anchor_total_tmp)) if not np.isnan(exp_total) else 1.0

        margin_sd = float(MARGIN_SD_BASE) * float(_clamp(0.92 + 0.20 * pace_proxy, 0.85, 1.15))

        # Win prob + blowout probabilities (|margin| >= 15 / 25)
        win_prob_home = _mix_norm_win_prob(mu_margin_home, margin_sd)
        blowout_prob_abs15 = _mix_norm_tail_prob_abs_ge(15.0, mu_margin_home, margin_sd)
        blowout_prob_abs25 = _mix_norm_tail_prob_abs_ge(25.0, mu_margin_home, margin_sd)

        # 95% and 80% ranges for margin (points)
        margin_ci95_low, margin_ci95_high = _normal_ci(mu_margin_home, margin_sd, z=1.96)
        margin_ci80_low, margin_ci80_high = _normal_ci(mu_margin_home, margin_sd, z=1.2816)
        win_prob_ci95_low = _mix_norm_win_prob(mu_margin_home - 1.96 * margin_sd, margin_sd)
        win_prob_ci95_high = _mix_norm_win_prob(mu_margin_home + 1.96 * margin_sd, margin_sd)

        # Optional: blend margin with external advanced metrics (SRS/Pace/ShotQuality/PER)
        mu_margin_home = _blend_margin_with_adv(mu_margin_home, home, away, adv_df)
        win_prob_home = _mix_norm_win_prob(mu_margin_home, margin_sd)

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

        total_ci95_low, total_ci95_high = _normal_ci(model_total, sd, z=1.96)
        total_ci80_low, total_ci80_high = _normal_ci(model_total, sd, z=1.2816)

        total_side = "NONE"
        total_edge_goals = float("nan")
        total_edge_vs_be = float("nan")
        total_recommendation = "No total bet (missing total/model)"

        if not np.isnan(model_total) and not np.isnan(total_points) and sd > 0:
            z = (model_total - total_points) / sd
            p_over = float(_clamp(_phi(z), 0.001, 0.999))
            p_under = 1.0 - p_over

            be_over = _breakeven_prob_from_american(total_over_price)
            be_under = _breakeven_prob_from_american(total_under_price)
            edge_over = p_over - be_over
            edge_under = p_under - be_under

            total_edge_goals = float(model_total - total_points)
            if edge_over >= edge_under:
                total_side = "OVER"
                total_edge_vs_be = float(edge_over)
            else:
                total_side = "UNDER"
                total_edge_vs_be = float(edge_under)

            if abs(total_edge_goals) >= TOTAL_MIN_PTS_EDGE and total_edge_vs_be >= TOTAL_MIN_EDGE_VS_BE:
                total_recommendation = f"Model PICK TOTAL: {total_side}"
            else:
                total_recommendation = "No total bet (edge too small)"

        rows.append(
            {
                "date": game_date_str,
                "home": home,
                "away": away,
                "model_home_prob": float(p_home),
                "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
                "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
                "edge_away": float(-edge_home) if not np.isnan(edge_home) else np.nan,
                "ml_recommendation": ml_pick,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "rest_days_home": float(h_days) if h_days is not None else np.nan,
                "rest_days_away": float(a_days) if a_days is not None else np.nan,
                "form_elo_home": float(form_elo_home),
                "form_elo_away": float(form_elo_away),
                "inj_points_home": float(inj_pts_home),
                "inj_points_away": float(inj_pts_away),
                "elo_diff": float(elo_diff),
                "model_spread_home": float(model_spread_home) if not np.isnan(model_spread_home) else np.nan,
                "model_margin_home": float(mu_margin_home) if not np.isnan(mu_margin_home) else np.nan,
                "margin_sd": float(margin_sd) if not np.isnan(margin_sd) else np.nan,
                "margin_ci95_low": float(margin_ci95_low) if not np.isnan(margin_ci95_low) else np.nan,
                "margin_ci95_high": float(margin_ci95_high) if not np.isnan(margin_ci95_high) else np.nan,
                "margin_ci80_low": float(margin_ci80_low) if not np.isnan(margin_ci80_low) else np.nan,
                "margin_ci80_high": float(margin_ci80_high) if not np.isnan(margin_ci80_high) else np.nan,
                "win_prob_home": float(win_prob_home) if not np.isnan(win_prob_home) else np.nan,
                "win_prob_ci95_low": float(win_prob_ci95_low) if not np.isnan(win_prob_ci95_low) else np.nan,
                "win_prob_ci95_high": float(win_prob_ci95_high) if not np.isnan(win_prob_ci95_high) else np.nan,
                "blowout_prob_abs15": float(blowout_prob_abs15) if not np.isnan(blowout_prob_abs15) else np.nan,
                "blowout_prob_abs25": float(blowout_prob_abs25) if not np.isnan(blowout_prob_abs25) else np.nan,
                "home_spread": float(home_spread) if not np.isnan(home_spread) else np.nan,
                "spread_price": float(spread_price),
                "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
                "p_home_cover": float(p_home_cover) if not np.isnan(p_home_cover) else np.nan,
                "ats_pick_side": str(ats_side),
                "ats_p_win": float(ats_p_win) if not np.isnan(ats_p_win) else np.nan,
                "ats_be": float(ats_be) if not np.isnan(ats_be) else np.nan,
                "ats_edge_vs_be": float(ats_edge_vs_be) if not np.isnan(ats_edge_vs_be) else np.nan,
                "spread_recommendation": str(spread_reco),
                "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
                "total_over_price": float(total_over_price),
                "total_under_price": float(total_under_price),
                "model_total": float(model_total) if not np.isnan(model_total) else np.nan,
                "total_sd": float(sd) if not np.isnan(sd) else np.nan,
                "total_ci95_low": float(total_ci95_low) if not np.isnan(total_ci95_low) else np.nan,
                "total_ci95_high": float(total_ci95_high) if not np.isnan(total_ci95_high) else np.nan,
                "total_ci80_low": float(total_ci80_low) if not np.isnan(total_ci80_low) else np.nan,
                "total_ci80_high": float(total_ci80_high) if not np.isnan(total_ci80_high) else np.nan,
                "total_edge_goals": float(total_edge_goals) if not np.isnan(total_edge_goals) else np.nan,
                "total_edge_points": float(total_edge_goals) if not np.isnan(total_edge_goals) else np.nan,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_pick_side": total_side,
                "total_recommendation": str(total_recommendation),
            }
        )

    return pd.DataFrame(rows)


def _team_line_avg_sd(team: str, d: dict) -> Tuple[float, float]:
    t = canon_team(team) or team
    if not d or t not in d:
        return (float("nan"), float("nan"))
    xs = d.get(t) or []
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return (float("nan"), float("nan"))
    return (float(np.mean(xs)), float(np.std(xs)))


def _margin_model_spread_from_elo_diff(elo_diff: float) -> float:
    """
    Convert Elo diff -> model spread (market convention: negative means home favored).
    If you have a saved margin calibration, you can plug it in here later.
    """
    try:
        cal = load_margin_cal(MARGIN_CAL_PATH)
        if cal is not None:
            return float(cal.predict(float(elo_diff)))
    except Exception:
        pass

    # Fallback linear mapping
    # Typical NBA: ~30 Elo ~= 1 point
    return float(-elo_diff / 30.0)


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
