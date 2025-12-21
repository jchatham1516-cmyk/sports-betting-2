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

# Elo -> points (spread-ish)
ELO_PER_POINT = 40.0

# Injury scaling (reduced to avoid saturation)
MAX_ABS_INJ_ELO_ADJ = 45.0
MAX_ABS_INJ_POINTS = 6.0
INJ_ELO_PER_POINT = 6.0
QB_EXTRA_ELO = 10.0

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

# ATS gating (prevent ATS for every game)
ATS_MIN_EDGE_VS_BE = 0.03   # must beat breakeven by 3% to bet ATS
ATS_MIN_PTS_EDGE = 2.0      # must have at least 2.0 pts edge to bet ATS

# Plausibility gate:
# If the market line is large but your model line is near 0, it's probably an artifact
ATS_BIG_LINE = 7.0          # market spread magnitude considered "big"
ATS_TINY_MODEL = 2.0        # model spread magnitude considered "tiny"
ATS_BIGLINE_FORCE_PASS = True

# Optional: limit ATS picks per day
MAX_ATS_PLAYS_PER_DAY = 3   # set None to disable


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
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
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


def _recent_form_adjustments(days_back: int = FORM_LOOKBACK_DAYS) -> Dict[str, Dict[str, float]]:
    """
    Builds per-team recent form adjustments based on average scoring margin.
    Returns:
      { team: {"avg_margin": float, "games": int, "elo_adj": float } }
    """
    sport_key = SPORT_TO_ODDS_KEY.get("nfl")
    if not sport_key:
        return {}

    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_back), 45))
    except Exception:
        return {}

    margins = defaultdict(list)  # team -> list[(date, margin)]

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
            hs = score_map.get(home_raw) or score_map.get(home)
            aw = score_map.get(away_raw) or score_map.get(away)
            hs = float(hs)
            aw = float(aw)
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

        out[team] = {
            "avg_margin": float(avg_margin),
            "games": int(games),
            "elo_adj": float(elo_adj),
        }

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


# ---------- ATS helpers ----------
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
# Data builders
# ----------------------------
def _build_last_game_date_map(days_back: int = 21) -> Dict[str, date]:
    sport_key = SPORT_TO_ODDS_KEY["nfl"]
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
    sport_key = SPORT_TO_ODDS_KEY["nfl"]

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
            hs = score_map.get(home_raw) or score_map.get(home)
            aw = score_map.get(away_raw) or score_map.get(away)
            hs = float(hs)
            aw = float(aw)
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
def run_daily_nfl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=14)

    # Parse date
    try:
        target_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    except Exception:
        target_date = None

    # Load injuries once
    try:
        injuries_map = fetch_espn_nfl_injuries()
    except Exception as e:
        print(f"[nfl injuries] WARNING: failed to load ESPN injuries: {e}")
        injuries_map = {}

    # Rest map once
    last_played = _build_last_game_date_map(days_back=21)

    # Recent form once
    form_map = _recent_form_adjustments(days_back=FORM_LOOKBACK_DAYS)

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)

        eh = st.get(home)
        ea = st.get(away)

        # Rest days
        home_days_off = _calc_days_off(target_date, last_played.get(home))
        away_days_off = _calc_days_off(target_date, last_played.get(away))
        rest_adj = _rest_elo(home_days_off) - _rest_elo(away_days_off)

        # Injuries
        home_inj = build_injury_list_for_team_nfl(home, injuries_map)
        away_inj = build_injury_list_for_team_nfl(away, injuries_map)

        # Positive => HOME advantage (away more hurt)
        inj_pts_raw = float(injury_adjustment_points(home_inj, away_inj))
        inj_pts = _clamp(inj_pts_raw, -MAX_ABS_INJ_POINTS, MAX_ABS_INJ_POINTS)

        # QB cost (use impact already scaled in injuries.py)
        def qb_cost(lst) -> float:
            s = 0.0
            for (player, role, mult, impact) in (lst or []):
                try:
                    player_s = str(player).lower()
                    qb_like = (
                        (str(player_s).find(" qb") >= 0)
                        or ("quarterback" in player_s)
                        or (float(impact) >= 6.0)
                    )
                    if not qb_like:
                        continue
                    rw = 1.0 if role == "starter" else 0.55
                    s += rw * float(mult) * float(impact)
                except Exception:
                    continue
            return float(s)

        qb_home = qb_cost(home_inj)
        qb_away = qb_cost(away_inj)
        qb_diff = float(qb_away - qb_home)

        inj_elo_adj = float(inj_pts) * float(INJ_ELO_PER_POINT)
        qb_elo_adj = float(qb_diff) * float(QB_EXTRA_ELO)
        inj_total_elo = _clamp(inj_elo_adj + qb_elo_adj, -MAX_ABS_INJ_ELO_ADJ, MAX_ABS_INJ_ELO_ADJ)

        # Recent form (scoring margin -> Elo tweak)
        form_home = float((form_map.get(home) or {}).get("elo_adj", 0.0))
        form_away = float((form_map.get(away) or {}).get("elo_adj", 0.0))
        form_diff = float(form_home - form_away)

        # Symmetric injury + form application
        eh_eff = eh + rest_adj + 0.5 * inj_total_elo + 0.5 * form_diff
        ea_eff = ea - 0.5 * inj_total_elo - 0.5 * form_diff

        # Win prob + compression
        p_raw = float(elo_win_prob(eh_eff, ea_eff, home_adv=HOME_ADV))
        p_home = _clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99)

        # Spread-ish output
        elo_diff = (eh_eff - ea_eff) + HOME_ADV
        model_spread_home = _clamp(-(elo_diff / ELO_PER_POINT), -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        # Market
        home_ml = _safe_float((oi or {}).get("home_ml"))
        away_ml = _safe_float((oi or {}).get("away_ml"))
        home_spread = _safe_float((oi or {}).get("home_spread"))
        spread_price = _safe_float((oi or {}).get("spread_price"), default=ATS_DEFAULT_PRICE)

        # Market no-vig prob + ML edges
        mkt_home_p = float("nan")
        if not np.isnan(home_ml) and not np.isnan(away_ml):
            mkt_home_p, _ = _no_vig_probs(home_ml, away_ml)

        edge_home = float(p_home - mkt_home_p) if not np.isnan(mkt_home_p) else float("nan")
        edge_away = float(-edge_home) if not np.isnan(edge_home) else float("nan")

        ml_pick = _ml_recommendation(float(p_home), float(mkt_home_p), min_edge=MIN_ML_EDGE)
        value_tier = _pick_value_tier(abs(edge_home)) if not np.isnan(edge_home) else "UNKNOWN"

        # ATS calculations (spread edge in "home spread" sign convention)
        spread_edge_home = float(home_spread - model_spread_home) if not np.isnan(home_spread) else float("nan")
        p_home_cover = _cover_prob_from_edge(spread_edge_home, sd_pts=ATS_SD_PTS)
        ats_side, ats_p_win, ats_edge_vs_be, ats_be = _ats_pick_and_edge(p_home_cover, spread_price=spread_price)

        # --- ATS gating to avoid "ATS every game" ---
        ats_pass_reason = ""
        ats_allowed = True

        if np.isnan(home_spread) or np.isnan(model_spread_home):
            ats_allowed = False
            ats_pass_reason = "missing spread"
        else:
            # Plausibility gate
            if ATS_BIGLINE_FORCE_PASS and abs(home_spread) >= ATS_BIG_LINE and abs(model_spread_home) <= ATS_TINY_MODEL:
                ats_allowed = False
                ats_pass_reason = "big market line but tiny model line"

            # Minimum edge gates (vs breakeven)
            if ats_allowed and (np.isnan(ats_edge_vs_be) or ats_edge_vs_be < ATS_MIN_EDGE_VS_BE):
                ats_allowed = False
                ats_pass_reason = f"ats_edge_vs_be<{ATS_MIN_EDGE_VS_BE:.3f}"

            # Minimum points edge gate
            if ats_allowed and (np.isnan(spread_edge_home) or abs(spread_edge_home) < ATS_MIN_PTS_EDGE):
                ats_allowed = False
                ats_pass_reason = f"|spread_edge|<{ATS_MIN_PTS_EDGE:.1f}"

        if not ats_allowed:
            ats_strength = "pass"
            spread_reco = "No ATS bet (gated)"
        else:
            ats_strength = _ats_strength_label(ats_edge_vs_be)
            spread_reco = _ats_reco(ats_side, ats_strength)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,

            # Model
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),

            # Market (processed)
            "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
            "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
            "edge_away": float(edge_away) if not np.isnan(edge_home) else np.nan,

            # ATS
            "spread_edge_home": float(spread_edge_home) if not np.isnan(spread_edge_home) else np.nan,
            "ats_home_cover_prob": float(p_home_cover) if not np.isnan(p_home_cover) else np.nan,
            "ats_pick_side": ats_side,
            "ats_pick_prob": float(ats_p_win) if not np.isnan(ats_p_win) else np.nan,
            "ats_breakeven_prob": float(ats_be) if not np.isnan(ats_be) else np.nan,
            "ats_edge_vs_be": float(ats_edge_vs_be) if not np.isnan(ats_edge_vs_be) else np.nan,
            "ats_strength": ats_strength,
            "ats_pass_reason": ats_pass_reason,

            "ml_recommendation": ml_pick,
            "spread_recommendation": spread_reco,
            "value_tier": value_tier,

            # Debug
            "elo_diff": float(elo_diff),
            "inj_points_raw": float(inj_pts_raw),
            "inj_points": float(inj_pts),
            "inj_elo_total": float(inj_total_elo),
            "qb_diff": float(qb_diff),
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
    if MAX_ATS_PLAYS_PER_DAY is not None and not df.empty:
        elig = df["spread_recommendation"].astype(str).str.contains("Model PICK ATS:", na=False)
        df["ats_rank_score"] = np.where(elig, df["ats_edge_vs_be"].astype(float), -999.0)

        top_idx = df.sort_values("ats_rank_score", ascending=False).head(MAX_ATS_PLAYS_PER_DAY).index
        keep = set(top_idx.tolist())

        for i in df.index:
            if elig.loc[i] and i not in keep:
                df.loc[i, "spread_recommendation"] = "No ATS bet (top-N filter)"
                df.loc[i, "ats_strength"] = "pass"
                df.loc[i, "ats_pass_reason"] = "top-N filter"

        df.drop(columns=["ats_rank_score"], inplace=True, errors="ignore")

    return df


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
    return run_daily_nfl(str(date_in), odds_dict=(odds_dict or {}))
