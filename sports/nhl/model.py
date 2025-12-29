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
HOME_ADV = float(os.getenv("NHL_HOME_ADV", "45.0"))
ELO_K = float(os.getenv("NHL_ELO_K", "18.0"))

# IMPORTANT: if this is too small, Elo never separates and probs look identical
ELO_TRAIN_DAYS = int(os.getenv("NHL_ELO_TRAIN_DAYS", "180"))

# Elo -> goals (fallback until margin calibrator is trained)
ELO_PER_GOAL = float(os.getenv("NHL_ELO_PER_GOAL", "55.0"))
MAX_ABS_MODEL_SPREAD = float(os.getenv("NHL_MAX_ABS_MODEL_SPREAD", "2.5"))

# Prob compression
BASE_COMPRESS = float(os.getenv("NHL_BASE_COMPRESS", "0.78"))
MIN_ML_EDGE = float(os.getenv("NHL_MIN_ML_EDGE", "0.02"))

# Calibration
CAL_MIN_GAMES = int(os.getenv("NHL_CAL_MIN_GAMES", "120"))

# Rest effects (Elo points)
SHORT_REST_PENALTY_ELO = float(os.getenv("NHL_SHORT_REST_PENALTY_ELO", "-10.0"))
NORMAL_REST_BONUS_ELO = float(os.getenv("NHL_NORMAL_REST_BONUS_ELO", "0.0"))

# Recent form (margin = goals_for - goals_against)
FORM_LOOKBACK_DAYS = int(os.getenv("NHL_FORM_LOOKBACK_DAYS", "35"))
FORM_MIN_GAMES = int(os.getenv("NHL_FORM_MIN_GAMES", "2"))
FORM_ELO_PER_GOAL = float(os.getenv("NHL_FORM_ELO_PER_GOAL", "7.0"))
FORM_ELO_CLAMP = float(os.getenv("NHL_FORM_ELO_CLAMP", "35.0"))

# Missing-Elo handling: shrink toward 0.5 if a team is missing in Elo state
MISSING_ELO_SHRINK = float(os.getenv("NHL_MISSING_ELO_SHRINK", "0.40"))

# Totals (historical market totals lines)
TOTAL_DEFAULT_PRICE = float(os.getenv("NHL_TOTAL_DEFAULT_PRICE", "-110.0"))
TOTAL_HIST_DAYS = int(os.getenv("NHL_TOTAL_HIST_DAYS", "21"))
TOTAL_REGRESS_WEIGHT = float(os.getenv("NHL_TOTAL_REGRESS_WEIGHT", "0.40"))
TOTAL_SD_FLOOR = float(os.getenv("NHL_TOTAL_SD_FLOOR", "0.55"))
TOTAL_SD_CEIL = float(os.getenv("NHL_TOTAL_SD_CEIL", "1.35"))
TOTAL_MIN_EDGE_VS_BE = float(os.getenv("NHL_TOTAL_MIN_EDGE_VS_BE", "0.02"))
TOTAL_MIN_GOALS_EDGE = float(os.getenv("NHL_TOTAL_MIN_GOALS_EDGE", "0.35"))

# Sanity behavior: warn by default, optionally raise
STRICT_SANITY = os.getenv("NHL_STRICT_SANITY", "0") == "1"


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
    # back-to-back / 0 days rest
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


def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nhl"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=int(days_back))

    last_played: Dict[str, date] = {}
    for ev in events:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        if not home_raw or not away_raw:
            continue

        home = canon_team(home_raw)
        away = canon_team(away_raw)
        if not home or not away:
            continue

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
        elo_adj = _clamp(avg_margin * FORM_ELO_PER_GOAL, -FORM_ELO_CLAMP, FORM_ELO_CLAMP)
        out[team] = {"avg_margin": float(avg_margin), "games": int(games), "elo_adj": float(elo_adj)}
    return out


def update_elo_from_recent_scores(*, days_from: int = None, force_full_rebuild: bool = False) -> EloState:
    """
    Updates Elo ratings from recent completed NHL games.

    IMPORTANT: if days_from is too small, Elo stays near-default and daily probs look identical.
    """
    # Full rebuild = wipe processed + ratings
    if force_full_rebuild:
        st = EloState()
    else:
        st = EloState.load(ELO_PATH)

    sport_key = SPORT_TO_ODDS_KEY["nhl"]
    train_days = int(days_from) if days_from is not None else int(ELO_TRAIN_DAYS)
    train_days = int(max(30, train_days))

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

        # If not rebuilding, avoid double-processing games
        game_key = f"{ev.get('id','')}|{ev.get('commence_time','')}|{home}|{away}"
        if (not force_full_rebuild) and st.is_processed(game_key):
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

    # optional calibration
    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            cal = fit_platt(np.array(train_ps, dtype=float), np.array(train_ys, dtype=float))
            save_platt(PLATT_PATH, cal)

            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nhl calibration] WARNING: calibration fit failed: {e}")

    return st


def run_daily_nhl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    # If you want to force rebuild from runner without changing runner code:
    # set env NHL_FORCE_FULL_REBUILD=1 once, run, then unset.
    force = os.getenv("NHL_FORCE_FULL_REBUILD", "0") == "1"
    st = update_elo_from_recent_scores(days_from=ELO_TRAIN_DAYS, force_full_rebuild=force)

    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    last_played = _build_last_game_date_map(days_back=21)
    form_map = _recent_form_adjustments(days_back=FORM_LOOKBACK_DAYS)

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
                return (_safe_float(v.get("avg")), _safe_float(v.get("sd"), default=np.nan))
        return (float("nan"), float("nan"))

    def _form_adj(team: str) -> float:
        try:
            return float((form_map.get(team) or {}).get("elo_adj", 0.0))
        except Exception:
            return 0.0

    rows: list[dict] = []
    default_ct = 0

    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)
        if not home or not away:
            continue

        eh = st.get(home)
        ea = st.get(away)

        if eh == st.default_elo or ea == st.default_elo:
            default_ct += 1
            msg = f"[NHL WARNING] Default Elo used: home={home_in}->{home} eh={eh}, away={away_in}->{away} ea={ea}"
            if STRICT_SANITY:
                raise RuntimeError(msg)
            print(msg)

        # Rest
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Form
        form_diff = float(_form_adj(home) - _form_adj(away))

        # Effective Elo
        eh_eff = float(eh) + float(rest_adj) + 0.5 * float(form_diff)
        ea_eff = float(ea) - 0.5 * float(form_diff)

        # Model win prob
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_home = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))

        # Market ML
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)

        # Missing-Elo shrink (prevents crazy default-based biases, but DOES NOT copy market)
        home_missing = home not in (st.ratings or {})
        away_missing = away not in (st.ratings or {})
        if home_missing or away_missing:
            p_home = float(_clamp(0.5 + float(MISSING_ELO_SHRINK) * (p_home - 0.5), 0.01, 0.99))

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        ml_pick = _ml_recommendation(p_home, mkt_home_p)

        # Totals
        total_points = _safe_float((oi or {}).get("total_points"))
        over_price = _safe_float((oi or {}).get("over_price"), default=TOTAL_DEFAULT_PRICE)
        under_price = _safe_float((oi or {}).get("under_price"), default=TOTAL_DEFAULT_PRICE)

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
                "ml_recommendation": ml_pick,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "rest_days_home": np.nan if home_days_off is None else float(home_days_off),
                "rest_days_away": np.nan if away_days_off is None else float(away_days_off),
                "form_elo_home": float(_form_adj(home)),
                "form_elo_away": float(_form_adj(away)),
                # totals
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

    # Sanity: constant probability detector + too-many-default warning
    if len(rows) >= 5:
        probs = [round(r["model_home_prob"], 3) for r in rows if not np.isnan(r.get("model_home_prob", np.nan))]
        if len(set(probs)) <= 2:
            msg = "Model produced near-constant probabilities — likely default Elo / short training window / mapping issue."
            if STRICT_SANITY:
                raise RuntimeError(msg)
            print(f"[NHL WARNING] {msg}")

    if len(rows) > 0 and default_ct / max(1, len(rows)) >= 0.50:
        print(
            f"[NHL WARNING] {default_ct}/{len(rows)} games used default Elo. "
            f"Try NHL_FORCE_FULL_REBUILD=1 once, or increase NHL_ELO_TRAIN_DAYS."
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
    return run_daily_nhl(str(date_in), odds_dict=(odds_dict or {}))
