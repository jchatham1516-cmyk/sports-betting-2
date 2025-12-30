# sports/nhl/model.py
from __future__ import annotations

import math
import os
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.historical_totals import build_team_historical_total_lines

from sports.common.prob_calibration import load as load_platt, save as save_platt, fit_platt
from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin

ELO_PATH = "results/elo_state_nhl.json"
PLATT_PATH = "results/prob_cal_nhl.json"
MARGIN_CAL_PATH = "results/margin_cal_nhl.json"

HOME_ADV = float(os.getenv("NHL_HOME_ADV", "45.0"))
ELO_K = float(os.getenv("NHL_ELO_K", "18.0"))

BASE_COMPRESS = float(os.getenv("NHL_BASE_COMPRESS", "0.78"))
MIN_ML_EDGE = float(os.getenv("NHL_MIN_ML_EDGE", "0.02"))

# If Elo training fails, allow market->elo inference (prevents constant 0.520 outputs)
FALLBACK_USE_MARKET_IF_EMPTY = os.getenv("NHL_FALLBACK_USE_MARKET_IF_EMPTY", "1") == "1"
MARKET_FALLBACK_BLEND = float(os.getenv("NHL_MARKET_FALLBACK_BLEND", "0.35"))  # 0..0.85

CAL_MIN_GAMES = int(os.getenv("NHL_CAL_MIN_GAMES", "120"))

TOTAL_DEFAULT_PRICE = float(os.getenv("NHL_TOTAL_DEFAULT_PRICE", "-110.0"))
TOTAL_HIST_DAYS = int(os.getenv("NHL_TOTAL_HIST_DAYS", "21"))
TOTAL_REGRESS_WEIGHT = float(os.getenv("NHL_TOTAL_REGRESS_WEIGHT", "0.40"))
TOTAL_SD_FLOOR = float(os.getenv("NHL_TOTAL_SD_FLOOR", "0.55"))
TOTAL_SD_CEIL = float(os.getenv("NHL_TOTAL_SD_CEIL", "1.35"))
TOTAL_MIN_EDGE_VS_BE = float(os.getenv("NHL_TOTAL_MIN_EDGE_VS_BE", "0.02"))
TOTAL_MIN_GOALS_EDGE = float(os.getenv("NHL_TOTAL_MIN_GOALS_EDGE", "0.35"))
TOTAL_LINE_BLEND = float(os.getenv("NHL_TOTAL_LINE_BLEND", "0.35"))

# Recent scoring model
PTS_LOOKBACK_DAYS = int(os.getenv("NHL_PTS_LOOKBACK_DAYS", "45"))
PTS_MIN_GAMES = int(os.getenv("NHL_PTS_MIN_GAMES", "2"))
PTS_REGRESS = float(os.getenv("NHL_PTS_REGRESS", "0.35"))
PTS_LEAGUE_CLAMP_MIN = float(os.getenv("NHL_PTS_LEAGUE_CLAMP_MIN", "2.4"))
PTS_LEAGUE_CLAMP_MAX = float(os.getenv("NHL_PTS_LEAGUE_CLAMP_MAX", "3.6"))

STRICT_SANITY = os.getenv("NHL_STRICT_SANITY", "0") == "1"


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


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ml_recommendation(model_p: float, market_p: float, min_edge: float = MIN_ML_EDGE) -> str:
    if np.isnan(model_p) or np.isnan(market_p):
        return "No ML bet (missing market prob)"
    edge = model_p - market_p
    if edge >= min_edge:
        return "Model PICK: HOME ML (strong)" if edge >= 0.06 else "Model lean: HOME ML"
    if edge <= -min_edge:
        return "Model PICK: AWAY ML (strong)" if edge <= -0.06 else "Model lean: AWAY ML"
    return "No ML bet (edge too small)"


def _odds_scores_fallback(sport_key: str, days_from: int) -> list[dict]:
    """
    Direct Odds API scores endpoint fallback:
      GET /v4/sports/{sport_key}/scores?daysFrom=...
    """
    key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API") or ""
    if not key:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    try:
        r = requests.get(url, params={"apiKey": key, "daysFrom": int(days_from)}, timeout=20)
        if r.status_code != 200:
            return []
        return r.json() or []
    except Exception:
        return []


def update_elo_from_recent_scores(days_from: int = 120) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nhl"]

    train_days = int(max(30, int(days_from or 120)))
    events = []
    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=train_days) or []
    except Exception:
        events = []

    # If scores source returns nothing, fallback to Odds API scores endpoint
    if not events:
        print("[nhl] WARNING: fetch_recent_scores returned 0 events; using Odds API /scores fallback.")
        events = _odds_scores_fallback(sport_key=sport_key, days_from=train_days)

    train_ps: list[float] = []
    train_ys: list[float] = []
    train_xs: list[float] = []
    train_margins: list[float] = []

    processed_any = False

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
        processed_any = True

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)

    if not processed_any:
        print("[nhl] WARNING: Elo update processed 0 games. Ratings may stay empty/constant.")

    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            cal = fit_platt(np.array(train_ps, dtype=float), np.array(train_ys, dtype=float))
            save_platt(PLATT_PATH, cal)

            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nhl calibration] WARNING: calibration fit failed: {e}")

    return st


def _build_team_scoring_table(days_back: int) -> pd.DataFrame:
    sport_key = SPORT_TO_ODDS_KEY["nhl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=int(days_back))

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


def _expected_points_total(home: str, away: str, league_pts: float, team_tbl: pd.DataFrame) -> Tuple[float, float, float]:
    if team_tbl is None or team_tbl.empty or league_pts <= 1e-6:
        return (league_pts, league_pts, 2.0 * league_pts)

    def _team_means(t: str) -> Tuple[Optional[float], Optional[float]]:
        sub = team_tbl[team_tbl["team"] == t]
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


def run_daily_nhl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=120)

    # Historical totals lines
    sport_key = SPORT_TO_ODDS_KEY.get("nhl")
    team_total_lines: Dict[str, Dict[str, float]] = {}
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
    league_sd_total = float(np.mean(league_sds)) if league_sds else 0.95

    team_tbl = _build_team_scoring_table(days_back=PTS_LOOKBACK_DAYS)
    if team_tbl is None or team_tbl.empty:
        team_tbl = _build_team_scoring_table(days_back=120)

    league_pts = 3.0
    try:
        if not np.isnan(league_avg_total) and league_avg_total > 1.0:
            league_pts = float(league_avg_total / 2.0)
        elif team_tbl is not None and not team_tbl.empty:
            league_pts = float(team_tbl["pts_for"].mean())
        league_pts = _clamp(league_pts, PTS_LEAGUE_CLAMP_MIN, PTS_LEAGUE_CLAMP_MAX)
    except Exception:
        league_pts = 3.0

    def _team_line_avg_sd(team_canon: str, team_raw: str) -> Tuple[float, float]:
        for k in [team_canon, team_raw, (team_raw or "").strip(), (team_canon or "").strip()]:
            if not k:
                continue
            v = (team_total_lines or {}).get(k)
            if isinstance(v, dict) and v.get("avg") is not None:
                return (_safe_float(v.get("avg")), _safe_float(v.get("sd"), default=np.nan))
        return (float("nan"), float("nan"))

    ratings_map = getattr(st, "ratings", {}) or {}
    if not ratings_map:
        print("[nhl] WARNING: Elo ratings map is empty after update. Expect constant probs unless fallback enabled.")

    rows: list[dict] = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        # Market prob
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)

        # Elo-based prob (if we actually have ratings)
        eh = st.get(home)
        ea = st.get(away)
        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        p_home = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))

        # If ratings empty (or both default), blend toward market but do NOT copy it 1:1
        if FALLBACK_USE_MARKET_IF_EMPTY and (not ratings_map or (eh == st.default_elo and ea == st.default_elo)):
            if not np.isnan(mkt_home_p):
                w = float(_clamp(MARKET_FALLBACK_BLEND, 0.0, 0.85))
                p_home = float(_clamp((1.0 - w) * p_home + w * float(mkt_home_p), 0.01, 0.99))

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        ml_pick = _ml_recommendation(p_home, mkt_home_p)

        # Totals
        total_points = _safe_float((oi or {}).get("total_points"))
        over_price = _safe_float((oi or {}).get("over_price"), default=TOTAL_DEFAULT_PRICE)
        under_price = _safe_float((oi or {}).get("under_price"), default=TOTAL_DEFAULT_PRICE)

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

        total_side = "NONE"
        total_edge_goals = float("nan")
        total_edge_vs_be = float("nan")
        total_recommendation = "No total bet (missing total/model)"

        if not np.isnan(model_total) and not np.isnan(total_points) and sd > 0:
            z = (model_total - total_points) / sd
            p_over = float(_clamp(_phi(z), 0.001, 0.999))
            p_under = 1.0 - p_over
            be_over = _breakeven_prob_from_american(over_price)
            be_under = _breakeven_prob_from_american(under_price)
            edge_over = p_over - be_over
            edge_under = p_under - be_under

            total_edge_goals = float(model_total - total_points)
            if edge_over >= edge_under:
                total_side = "OVER"
                total_edge_vs_be = float(edge_over)
            else:
                total_side = "UNDER"
                total_edge_vs_be = float(edge_under)

            if abs(total_edge_goals) >= TOTAL_MIN_GOALS_EDGE and total_edge_vs_be >= TOTAL_MIN_EDGE_VS_BE:
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
                "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
                "total_over_price": float(over_price),
                "total_under_price": float(under_price),
                "model_total": float(model_total) if not np.isnan(model_total) else np.nan,
                "total_edge_goals": float(total_edge_goals) if not np.isnan(total_edge_goals) else np.nan,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_pick_side": total_side,
                "total_recommendation": str(total_recommendation),
            }
        )

    # Warn if near-constant probs
    if len(rows) >= 5:
        probs = [round(r["model_home_prob"], 3) for r in rows if not np.isnan(r.get("model_home_prob", np.nan))]
        if len(set(probs)) <= 2:
            msg = "Model produced near-constant probabilities — check scores feed / team mapping."
            if STRICT_SANITY:
                raise RuntimeError(msg)
            print(f"[NHL WARNING] {msg}")

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
    return run_daily_nhl(str(date_in), odds_dict=(odds_dict or {}))
