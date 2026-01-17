# sports/nhl/model.py
from __future__ import annotations

import math
import os
import re
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

from sports.common.margin_calibration import load as load_margin_cal, save as save_margin_cal, fit as fit_margin
from sports.nhl.goalies import GoalieInfo, get_starting_goalies
from sports.nhl.goalie_ratings import current_season_label, get_goalie_rating_with_meta
from sports.nhl.results_source import fetch_nhl_completed_games

ELO_PATH = "results/elo_state_nhl.json"
MARGIN_CAL_PATH = "results/margin_cal_nhl.json"

HOME_ADV = float(os.getenv("NHL_HOME_ADV", "45.0"))
ELO_K = float(os.getenv("NHL_ELO_K", "18.0"))

BASE_COMPRESS = float(os.getenv("NHL_BASE_COMPRESS", "0.78"))
SAFE_SHRINK = float(os.getenv("NHL_SAFE_SHRINK", "0.75"))
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
MODEL_TOTAL_ANCHOR_W = float(os.getenv("NHL_MODEL_TOTAL_ANCHOR_W", "0.70"))
TOTAL_SANITY_MAX_DIFF = float(os.getenv("NHL_TOTAL_SANITY_MAX_DIFF", "12.0"))

# Recent scoring model
PTS_LOOKBACK_DAYS = int(os.getenv("NHL_PTS_LOOKBACK_DAYS", "45"))
PTS_MIN_GAMES = int(os.getenv("NHL_PTS_MIN_GAMES", "2"))
PTS_REGRESS = float(os.getenv("NHL_PTS_REGRESS", "0.35"))
PTS_LEAGUE_CLAMP_MIN = float(os.getenv("NHL_PTS_LEAGUE_CLAMP_MIN", "2.4"))
PTS_LEAGUE_CLAMP_MAX = float(os.getenv("NHL_PTS_LEAGUE_CLAMP_MAX", "3.6"))
TOTAL_RECENT_GAMES = int(os.getenv("NHL_TOTAL_RECENT_GAMES", "10"))
TOTAL_MODEL_MIN = float(os.getenv("NHL_TOTAL_MODEL_MIN", "4.0"))
TOTAL_MODEL_MAX = float(os.getenv("NHL_TOTAL_MODEL_MAX", "8.5"))

STRICT_SANITY = os.getenv("NHL_STRICT_SANITY", "0") == "1"
NHL_LEAGUE_AVG_GOALIE_RATING = float(os.getenv("NHL_LEAGUE_AVG_GOALIE_RATING", "0.0"))
NHL_GOALIE_WEIGHT = float(os.getenv("NHL_GOALIE_WEIGHT", "0.45"))
NHL_GOALIE_MAX_PROB_SHIFT = float(os.getenv("NHL_GOALIE_MAX_PROB_SHIFT", "0.06"))
NHL_GOALIE_UNKNOWN_PENALTY = float(os.getenv("NHL_GOALIE_UNKNOWN_PENALTY", "0.01"))
GOALIE_STRENGTH_WEIGHT = float(os.getenv("NHL_GOALIE_STRENGTH_WEIGHT", "0.012"))
GOALIE_MAX_SHIFT = float(os.getenv("NHL_GOALIE_MAX_SHIFT", "0.06"))


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float("nan")


def _shrink_prob_toward_half(p_raw: float, shrink: float) -> float:
    if not np.isfinite(p_raw):
        return float("nan")
    s = float(max(0.0, min(1.0, shrink)))
    return float(0.5 + s * (float(p_raw) - 0.5))


def _anchor_model_total(model_total_raw: float, market_total: float, anchor_w: float) -> float:
    if np.isnan(model_total_raw):
        return float("nan")
    if np.isnan(market_total):
        return float(model_total_raw)
    w = float(_clamp(anchor_w, 0.0, 1.0))
    return float(w * market_total + (1.0 - w) * model_total_raw)


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


def _goalie_date_keys(game_date_str: str) -> list[str]:
    formats = ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y")
    parsed: Optional[date] = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(str(game_date_str), fmt).date()
            break
        except Exception:
            continue
    if parsed is None:
        try:
            parsed_dt = datetime.fromisoformat(str(game_date_str).replace("Z", "+00:00"))
            parsed = parsed_dt.date()
        except Exception:
            parsed = None
    if parsed is None:
        return [str(game_date_str)]
    return [
        parsed.strftime("%Y-%m-%d"),
        parsed.strftime("%m-%d-%Y"),
        parsed.strftime("%m/%d/%Y"),
    ]


def _goalie_key_variants(team_key: Optional[str]) -> list[str]:
    if not team_key:
        return []
    raw = " ".join(str(team_key).strip().split())
    if not raw:
        return []
    seen: set[str] = set()
    variants: list[str] = []

    def _add(value: Optional[str]) -> None:
        if not value:
            return
        cleaned = " ".join(str(value).strip().split())
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        variants.append(cleaned)

    _add(raw)
    _add(raw.upper())
    raw_no_punct = re.sub(r"[’'`\\.]", "", raw)
    _add(raw_no_punct)
    _add(raw_no_punct.upper())
    raw_canon = canon_team(raw)
    _add(raw_canon)
    if raw_no_punct:
        _add(canon_team(raw_no_punct))

    simplified = raw_no_punct.strip()
    if simplified:
        upper_simplified = simplified.upper()
        if upper_simplified.startswith("NY "):
            _add(f"New York {simplified[3:].strip()}")
        if upper_simplified.startswith("NEW YORK "):
            _add(f"NY {simplified[9:].strip()}")
        if upper_simplified.startswith("N Y "):
            _add(f"New York {simplified[4:].strip()}")
    return variants


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
    events: list[dict] = []
    try:
        events = fetch_recent_scores(sport_key=sport_key, days_from=train_days) or []
    except Exception:
        events = []

    train_ps: list[float] = []
    train_ys: list[float] = []
    train_xs: list[float] = []
    train_margins: list[float] = []

    processed_any = False
    processed_count = 0
    parsed_scores_count = 0
    skip_counters = {
        "missing_home_away": 0,
        "missing_scores": 0,
        "canon_team_failure": 0,
        "score_parse_failure": 0,
        "already_processed": 0,
    }

    def _process_events(events_in: list[dict], *, source: str) -> None:
        nonlocal processed_any, processed_count, parsed_scores_count
        for ev in events_in:
            home_raw = ev.get("home_team")
            away_raw = ev.get("away_team")
            if not home_raw or not away_raw:
                skip_counters["missing_home_away"] += 1
                continue

            home = canon_team(home_raw)
            away = canon_team(away_raw)
            if not home or not away:
                skip_counters["canon_team_failure"] += 1
                continue

            game_date = ev.get("date") or ev.get("commence_time") or ev.get("completed_at")
            game_key = f"{ev.get('id') or ev.get('game_id') or ''}|{game_date or ''}|{home}|{away}"
            if st.is_processed(game_key):
                skip_counters["already_processed"] += 1
                continue

            scores = ev.get("scores")
            hs = None
            aw = None
            if scores:
                score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
                try:
                    hs = float(score_map.get(home_raw) or score_map.get(home))
                    aw = float(score_map.get(away_raw) or score_map.get(away))
                except Exception:
                    hs = None
                    aw = None
            else:
                try:
                    hs = float(ev.get("home_score"))
                    aw = float(ev.get("away_score"))
                except Exception:
                    hs = None
                    aw = None

            if hs is None or aw is None:
                if scores:
                    skip_counters["score_parse_failure"] += 1
                else:
                    skip_counters["missing_scores"] += 1
                continue

            parsed_scores_count += 1

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
            processed_count += 1

        print(
            f"[nhl elo] source={source} events={len(events_in)} processed={processed_count} parsed_scores={parsed_scores_count}"
        )

    if not events:
        print("[nhl] WARNING: fetch_recent_scores returned 0 events; using NHL schedule fallback.")
        events = []
    _process_events(events, source="odds_recent_scores")

    fallback_events_count = 0
    if processed_count < 200 or not processed_any:
        print(
            "[nhl] WARNING: too few usable scores from fetch_recent_scores; "
            "falling back to NHL schedule results."
        )
        fallback_events = fetch_nhl_completed_games(train_days)
        fallback_events_count = len(fallback_events)
        _process_events(fallback_events, source="nhl_schedule_results")

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)

    if not processed_any:
        print("[nhl] WARNING: Elo update processed 0 games. Ratings may stay empty/constant.")
    team_count = len(getattr(st, "ratings", {}) or {})
    print(
        "[nhl elo] summary "
        f"events_fetched={len(events)} fallback_events={fallback_events_count} "
        f"processed={processed_count} skips={skip_counters} team_count={team_count}"
    )
    print(f"[nhl elo] team_count={team_count}")
    if team_count == 0:
        print(f"[nhl elo] ELO EMPTY skips={skip_counters}")
        if os.getenv("NHL_STRICT_SANITY") == "1" and processed_any:
            raise RuntimeError(
                f"EloState persistence bug: processed={processed_count} but ratings empty. skips={skip_counters}"
            )
    st._nhl_elo_debug = {
        "events_fetched": len(events),
        "fallback_events": fallback_events_count,
        "processed": processed_count,
        "parsed_scores": parsed_scores_count,
        "skip_counters": dict(skip_counters),
        "teams": team_count,
    }

    try:
        if len(train_ps) >= CAL_MIN_GAMES:
            mcal = fit_margin(np.array(train_xs, dtype=float), np.array(train_margins, dtype=float))
            save_margin_cal(MARGIN_CAL_PATH, mcal)
    except Exception as e:
        print(f"[nhl calibration] WARNING: margin calibration fit failed: {e}")

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

        game_date = _parse_iso_date(ev.get("commence_time") or ev.get("completed_at") or ev.get("date"))
        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = float(score_map.get(home_raw) or score_map.get(home))
            aw = float(score_map.get(away_raw) or score_map.get(away))
        except Exception:
            continue

        rows.append({"team": home, "opp": away, "pts_for": hs, "pts_against": aw, "date": game_date})
        rows.append({"team": away, "opp": home, "pts_for": aw, "pts_against": hs, "date": game_date})

    return pd.DataFrame(rows)


def _team_recent_avgs(
    team: str,
    team_tbl: pd.DataFrame,
    n_games: int,
) -> Tuple[Optional[float], Optional[float], int]:
    if team_tbl is None or team_tbl.empty:
        return (None, None, 0)
    sub = team_tbl[team_tbl["team"] == team].copy()
    if sub.empty:
        return (None, None, 0)
    if "date" in sub.columns:
        sub = sub.sort_values("date", ascending=False)
    recent = sub.head(int(n_games)) if n_games > 0 else sub
    if recent.empty:
        return (None, None, 0)
    return (
        float(recent["pts_for"].mean()),
        float(recent["pts_against"].mean()),
        int(len(recent)),
    )


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


def _goalie_status_weight(status: str) -> float:
    status = str(status or "").upper()
    if status == "CONFIRMED":
        return 1.0
    if status == "PROJECTED":
        return 0.6
    return 0.35


def _goalie_game_status(home_name: Optional[str], away_name: Optional[str]) -> str:
    if home_name and away_name:
        return "OK"
    if home_name or away_name:
        return "PARTIAL"
    return "UNKNOWN"


def _sanitize_goalie_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = " ".join(str(name).strip().split())
    if not cleaned:
        return None
    if cleaned.upper() in {"TBD", "UNKNOWN", "TBA", "N/A"}:
        return None
    return cleaned


def _compute_goalie_adjustment(
    *,
    goalie_home_name: Optional[str],
    goalie_away_name: Optional[str],
    goalie_home_status: str,
    goalie_away_status: str,
    season_label: str,
) -> tuple[float, float, float, float, float, str, str, float, bool, bool]:
    goalie_home_rating = NHL_LEAGUE_AVG_GOALIE_RATING
    goalie_away_rating = NHL_LEAGUE_AVG_GOALIE_RATING
    goalie_rating_diff = 0.0
    goalie_prob_shift = 0.0
    goalie_adj = 0.0
    goalie_status = _goalie_game_status(goalie_home_name, goalie_away_name)
    goalie_reason = "goalie_status_unknown"

    home_found = False
    away_found = False
    status_w_home = _goalie_status_weight(goalie_home_status)
    status_w_away = _goalie_status_weight(goalie_away_status)
    if goalie_home_name and goalie_away_name:
        conf_w = float(status_w_home * status_w_away)
    else:
        conf_w = 0.5

    if goalie_home_name:
        goalie_home_rating, home_found = get_goalie_rating_with_meta(goalie_home_name, season_label)
    if goalie_away_name:
        goalie_away_rating, away_found = get_goalie_rating_with_meta(goalie_away_name, season_label)

    if not goalie_home_name and not goalie_away_name:
        return (
            0.0,
            goalie_home_rating,
            goalie_away_rating,
            goalie_rating_diff,
            goalie_prob_shift,
            goalie_status,
            "goalie_missing",
            conf_w,
            home_found,
            away_found,
        )

    if goalie_home_name and goalie_away_name:
        goalie_rating_diff = float(goalie_home_rating) - float(goalie_away_rating)
        max_shift = float(max(0.0, GOALIE_MAX_SHIFT))
        raw_shift = float(_clamp(goalie_rating_diff * GOALIE_STRENGTH_WEIGHT, -max_shift, max_shift))
        scaled_shift = float(raw_shift * conf_w * NHL_GOALIE_WEIGHT)
        if home_found and away_found:
            goalie_reason = "goalie_found"
        elif home_found or away_found:
            penalty = float(NHL_GOALIE_UNKNOWN_PENALTY * conf_w * NHL_GOALIE_WEIGHT)
            if home_found and not away_found:
                scaled_shift = float(_clamp(scaled_shift + penalty, -max_shift, max_shift))
            elif away_found and not home_found:
                scaled_shift = float(_clamp(scaled_shift - penalty, -max_shift, max_shift))
            goalie_reason = "partial_found"
        else:
            raw_shift = 0.0
            scaled_shift = 0.0
            goalie_reason = "goalie_rating_fallback"
        goalie_prob_shift = float(scaled_shift)
        goalie_adj = float(abs(scaled_shift))
        if os.getenv("NHL_DEBUG_GOALIES") == "1":
            print(
                "[nhl goalies] adjustment "
                f"home_goalie={goalie_home_name} away_goalie={goalie_away_name} "
                f"home_found={home_found} away_found={away_found} "
                f"raw_shift={raw_shift:.4f} scaled_shift={scaled_shift:.4f} "
                f"conf_w={conf_w:.3f} weight={NHL_GOALIE_WEIGHT:.3f} "
                f"strength_weight={GOALIE_STRENGTH_WEIGHT:.4f}"
            )
        return (
            goalie_adj,
            goalie_home_rating,
            goalie_away_rating,
            goalie_rating_diff,
            goalie_prob_shift,
            goalie_status,
            goalie_reason,
            conf_w,
            home_found,
            away_found,
        )

    goalie_reason = "goalie_missing_opponent"
    goalie_rating_diff = float(goalie_home_rating) - float(goalie_away_rating)
    return (
        goalie_adj,
        goalie_home_rating,
        goalie_away_rating,
        goalie_rating_diff,
        goalie_prob_shift,
        goalie_status,
        goalie_reason,
        conf_w,
        home_found,
        away_found,
    )


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

    league_avg_ga = float("nan")
    if team_tbl is not None and not team_tbl.empty:
        try:
            league_avg_ga = float(team_tbl["pts_against"].mean())
        except Exception:
            league_avg_ga = float("nan")

    # If historical lines were empty, anchor league totals to the scoring table so totals model still runs
    if np.isnan(league_avg_total) and not np.isnan(league_pts):
        try:
            league_avg_total = float(2.0 * league_pts)
        except Exception:
            pass

    if np.isnan(league_sd_total) and team_tbl is not None and not team_tbl.empty:
        try:
            league_sd_total = float((team_tbl["pts_for"] + team_tbl["pts_against"]).std(ddof=0))
        except Exception:
            pass

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

    try:
        dt = datetime.strptime(game_date_str, "%m/%d/%Y").date()
        date_key = dt.strftime("%Y-%m-%d")
    except Exception:
        date_key = str(game_date_str)
    season_label = current_season_label()
    try:
        season_label = current_season_label()
        dt = datetime.strptime(game_date_str, "%m/%d/%Y").date()
        season_label = str(dt.year - 1 if dt.month < 7 else dt.year)
    except Exception:
        pass

    debug_goalies = os.getenv("NHL_DEBUG_GOALIES") == "1"
    goalies_map_raw: dict = {}
    date_keys_tried = _goalie_date_keys(game_date_str)
    date_key_sizes: list[tuple[str, int]] = []
    for dk in date_keys_tried:
        try:
            candidate = get_starting_goalies(dk) or {}
            date_key_sizes.append((dk, len(candidate)))
            if candidate:
                goalies_map_raw = candidate
                break
        except Exception as exc:
            date_key_sizes.append((dk, 0))
            if debug_goalies:
                print(f"[nhl goalies] get_starting_goalies({dk}) failed: {exc}")
            continue

    if debug_goalies:
        print(
            "[nhl goalies] "
            f"date_keys_tried={date_keys_tried} raw_size={len(goalies_map_raw)} "
            f"date_key_sizes={date_key_sizes}"
        )
    if not goalies_map_raw:
        print(
            "[nhl goalies] WARNING: no starting goalies returned for any date format; "
            "check goalie source availability."
        )
    goalies_map_norm: dict[str, GoalieInfo] = {}
    for key, info in (goalies_map_raw or {}).items():
        if not key and not info:
            continue
        raw_key = " ".join(str(key).strip().split()) if key else ""
        if raw_key:
            for variant in {raw_key, canon_team(raw_key)}:
                if variant and variant not in goalies_map_norm:
                    goalies_map_norm[variant] = info
        for variant in _goalie_key_variants(raw_key):
            if variant not in goalies_map_norm:
                goalies_map_norm[variant] = info
        for variant in _goalie_key_variants(getattr(info, "team", None)):
            if variant not in goalies_map_norm:
                goalies_map_norm[variant] = info
        for variant in _goalie_key_variants(getattr(info, "original_team", None)):
            if variant not in goalies_map_norm:
                goalies_map_norm[variant] = info
    if debug_goalies:
        print(
            "[nhl goalies] "
            f"normalized_size={len(goalies_map_norm)} raw_keys_sample={list(goalies_map_raw)[:5]}"
        )

    def _get_goalie(team_canon: str, team_raw: str) -> tuple[GoalieInfo, list[str]]:
        keys_tried: list[str] = []
        seen: set[str] = set()
        for team_key in [team_canon, team_raw, canon_team(team_raw), canon_team(team_canon)]:
            for k in _goalie_key_variants(team_key):
                if not k or k in seen:
                    continue
                seen.add(k)
                keys_tried.append(k)
                if k in goalies_map_norm:
                    return goalies_map_norm[k], keys_tried
        return GoalieInfo(team=team_canon, goalie_name=None, status="UNKNOWN", source=""), keys_tried

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
        goalie_home_info, goalie_home_keys = _get_goalie(home, home_in)
        goalie_away_info, goalie_away_keys = _get_goalie(away, away_in)

        goalie_home_name = _sanitize_goalie_name(goalie_home_info.goalie_name)
        goalie_away_name = _sanitize_goalie_name(goalie_away_info.goalie_name)
        goalie_home_status = goalie_home_info.status or "UNKNOWN"
        goalie_away_status = goalie_away_info.status or "UNKNOWN"
        if not goalie_home_name:
            goalie_home_status = "UNKNOWN"
        if not goalie_away_name:
            goalie_away_status = "UNKNOWN"

        (
            goalie_adj,
            goalie_home_rating,
            goalie_away_rating,
            goalie_rating_diff,
            goalie_prob_shift,
            goalie_status,
            goalie_reason,
            goalie_confidence_weight,
            goalie_home_found,
            goalie_away_found,
        ) = _compute_goalie_adjustment(
            goalie_home_name=goalie_home_name,
            goalie_away_name=goalie_away_name,
            goalie_home_status=goalie_home_status,
            goalie_away_status=goalie_away_status,
            season_label=season_label,
        )
        if debug_goalies:
            print(
                "[nhl goalies] matchup "
                f"{away} @ {home} home_goalie={goalie_home_name} ({goalie_home_status}) "
                f"away_goalie={goalie_away_name} ({goalie_away_status}) "
                f"home_found={goalie_home_found} away_found={goalie_away_found} "
                f"home_source={goalie_home_info.source or 'unknown'} "
                f"away_source={goalie_away_info.source or 'unknown'} "
                f"home_keys={goalie_home_keys} away_keys={goalie_away_keys} "
                f"home_rating={goalie_home_rating:.3f} away_rating={goalie_away_rating:.3f} "
                f"rating_diff={goalie_rating_diff:.3f} conf_w={goalie_confidence_weight:.2f} "
                f"prob_shift={goalie_prob_shift:.4f} adj={goalie_adj:.4f}"
            )

        p_raw = float(elo_win_prob(eh, ea, home_adv=HOME_ADV))
        p_raw = float(_clamp(p_raw + goalie_prob_shift, 0.01, 0.99))
        p_home = float(_clamp(0.5 + BASE_COMPRESS * (p_raw - 0.5), 0.01, 0.99))
        p_home = float(_clamp(_shrink_prob_toward_half(p_home, SAFE_SHRINK), 0.01, 0.99))

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

        home_gf_avg, home_ga_avg, home_games = _team_recent_avgs(home, team_tbl, TOTAL_RECENT_GAMES)
        away_gf_avg, away_ga_avg, away_games = _team_recent_avgs(away, team_tbl, TOTAL_RECENT_GAMES)

        model_total = float("nan")
        home_exp = float("nan")
        away_exp = float("nan")
        missing_totals = (
            home_gf_avg is None
            or home_ga_avg is None
            or away_gf_avg is None
            or away_ga_avg is None
            or np.isnan(league_avg_ga)
            or league_avg_ga <= 0
        )

        if missing_totals:
            if np.isnan(league_avg_total):
                league_avg_total = float(2.0 * league_pts) if league_pts > 0 else float("nan")
            model_total = float(league_avg_total)
            print(f"[NHL TOTALS] FALLBACK used for {away} @ {home} (missing stats)")
        else:
            home_exp = float(home_gf_avg) * (float(away_ga_avg) / float(league_avg_ga))
            away_exp = float(away_gf_avg) * (float(home_ga_avg) / float(league_avg_ga))
            model_total = float(home_exp + away_exp)
            model_total = _clamp(model_total, TOTAL_MODEL_MIN, TOTAL_MODEL_MAX)
        def _fmt_total(v: Optional[float]) -> str:
            try:
                return f"{float(v):.2f}"
            except Exception:
                return "nan"

        print(
            "[NHL TOTALS] "
            f"{away} @ {home} home_gf_avg={_fmt_total(home_gf_avg)} home_ga_avg={_fmt_total(home_ga_avg)} "
            f"away_gf_avg={_fmt_total(away_gf_avg)} away_ga_avg={_fmt_total(away_ga_avg)} "
            f"model_total={_fmt_total(model_total)}"
        )

        model_total_raw = float(model_total) if not np.isnan(model_total) else float("nan")
        model_total_final = _anchor_model_total(model_total_raw, total_points, MODEL_TOTAL_ANCHOR_W)
        anchored = not np.isnan(model_total_raw) and not np.isnan(total_points)

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
        total_pick = "NO BET"
        total_edge_goals_raw = float("nan")
        total_edge_goals_final = float("nan")
        total_edge_vs_be = float("nan")
        total_recommendation = "No total bet (missing total/model)"
        total_flags: list[str] = ["TOTAL_ANCHORED"] if anchored else []

        if not np.isnan(model_total_raw) and not np.isnan(total_points):
            total_edge_goals_raw = float(model_total_raw - total_points)
        if not np.isnan(model_total_final) and not np.isnan(total_points):
            total_edge_goals_final = float(model_total_final - total_points)

        sanity_fail = False
        if not np.isnan(model_total_raw) and not np.isnan(total_points):
            if abs(float(model_total_raw - total_points)) > float(TOTAL_SANITY_MAX_DIFF):
                sanity_fail = True

        if not np.isnan(model_total_final) and not np.isnan(total_points) and sd > 0:
            z = (model_total_final - total_points) / sd
            p_over = float(_clamp(_phi(z), 0.001, 0.999))
            p_under = 1.0 - p_over
            be_over = _breakeven_prob_from_american(over_price)
            be_under = _breakeven_prob_from_american(under_price)
            edge_over = p_over - be_over
            edge_under = p_under - be_under

            if edge_over >= edge_under:
                total_side = "OVER"
                total_edge_vs_be = float(edge_over)
            else:
                total_side = "UNDER"
                total_edge_vs_be = float(edge_under)

        if sanity_fail:
            total_flags.append("TOTAL_SANITY_FAIL_PASS")
            total_recommendation = "No total bet (sanity fail)"
        elif np.isnan(total_edge_goals_final):
            total_flags.append("TOTAL_EDGE_TOO_SMALL")
            total_recommendation = "No total bet (missing total/model)"
        elif total_edge_goals_final >= TOTAL_MIN_GOALS_EDGE:
            total_flags.append("TOTAL_EDGE_OK")
            total_side = "OVER"
            total_pick = "OVER"
            total_recommendation = "Model PICK TOTAL: OVER"
        elif total_edge_goals_final <= -TOTAL_MIN_GOALS_EDGE:
            total_flags.append("TOTAL_EDGE_OK")
            total_side = "UNDER"
            total_pick = "UNDER"
            total_recommendation = "Model PICK TOTAL: UNDER"
        else:
            total_flags.append("TOTAL_EDGE_TOO_SMALL")
            total_recommendation = "No total bet (edge too small)"

        rows.append(
            {
                "date": game_date_str,
                "home": home,
                "away": away,
                "model_home_prob": float(p_home),
                "model_home_prob_raw": float(p_raw),
                "goalie_adj": float(goalie_adj),
                "goalie_status": goalie_status,
                "goalie_home_status": goalie_home_status,
                "goalie_away_status": goalie_away_status,
                "goalie_home_name": goalie_home_name,
                "goalie_away_name": goalie_away_name,
                "goalie_home_found": bool(goalie_home_found),
                "goalie_away_found": bool(goalie_away_found),
                "goalie_confidence_weight": float(goalie_confidence_weight),
                "goalie_home_rating": float(goalie_home_rating),
                "goalie_away_rating": float(goalie_away_rating),
                "goalie_rating_diff": float(goalie_rating_diff),
                "goalie_prob_shift": float(goalie_prob_shift),
                "goalie_reason": goalie_reason,
                "goalie_lookup_home_keys": ",".join(goalie_home_keys),
                "goalie_lookup_away_keys": ",".join(goalie_away_keys),
                "market_home_prob": float(mkt_home_p) if not np.isnan(mkt_home_p) else np.nan,
                "edge_home": float(edge_home) if not np.isnan(edge_home) else np.nan,
                "edge_away": float(-edge_home) if not np.isnan(edge_home) else np.nan,
                "ml_recommendation": ml_pick,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "total_points": float(total_points) if not np.isnan(total_points) else np.nan,
                "total_over_price": float(over_price),
                "total_under_price": float(under_price),
                "model_total_raw": float(model_total_raw) if not np.isnan(model_total_raw) else np.nan,
                "model_total_final": float(model_total_final) if not np.isnan(model_total_final) else np.nan,
                "market_total_used": float(total_points) if not np.isnan(total_points) else np.nan,
                "model_total": float(model_total_final) if not np.isnan(model_total_final) else np.nan,
                "total_edge_goals_raw": float(total_edge_goals_raw) if not np.isnan(total_edge_goals_raw) else np.nan,
                "total_edge_goals_final": float(total_edge_goals_final) if not np.isnan(total_edge_goals_final) else np.nan,
                "total_edge_goals": float(total_edge_goals_final) if not np.isnan(total_edge_goals_final) else np.nan,
                "total_edge_vs_be": float(total_edge_vs_be) if not np.isnan(total_edge_vs_be) else np.nan,
                "total_pick_side": total_side,
                "total_pick": total_pick,
                "total_recommendation": str(total_recommendation),
                "total_decision_flags": ",".join(total_flags),
            }
        )

    # Warn if near-constant probs
    if len(rows) >= 5:
        raw_probs = [
            float(r["model_home_prob_raw"])
            for r in rows
            if not np.isnan(r.get("model_home_prob_raw", np.nan))
        ]
        if len(raw_probs) >= 5:
            std_raw = float(np.std(raw_probs))
            if std_raw < 0.01:
                debug = getattr(st, "_nhl_elo_debug", {}) or {}
                msg = (
                    "Model produced near-constant raw probabilities — check scores feed / team mapping. "
                    f"std_raw={std_raw:.4f} teams_in_ratings={len(getattr(st, 'ratings', {}) or {})} "
                    f"processed_games={debug.get('processed')} skips={debug.get('skip_counters')}"
                )
                if STRICT_SANITY:
                    raise RuntimeError(msg)
                print(f"[NHL WARNING] {msg}")

    df = pd.DataFrame(rows)
    _run_goalie_regression_check(df)
    return df


def _run_goalie_regression_check(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    if os.getenv("NHL_GOALIE_REGRESSION_TEST", "0") != "1":
        # Default: keep production runs alive if goalie sources are missing.
        return
    home_names = df.get("goalie_home_name")
    away_names = df.get("goalie_away_name")
    if home_names is None or away_names is None:
        return
    home_found = home_names.fillna("").astype(str).str.strip().ne("")
    away_found = away_names.fillna("").astype(str).str.strip().ne("")
    any_both_named = bool((home_found & away_found).any())
    if not any_both_named:
        return
    goalie_adj = df.get("goalie_adj")
    if goalie_adj is None:
        raise RuntimeError("goalie_adj column missing during regression check")
    goalie_home_found = df.get("goalie_home_found")
    goalie_away_found = df.get("goalie_away_found")
    if goalie_home_found is not None and goalie_away_found is not None:
        both_found_series = (
            goalie_home_found.fillna(False).astype(bool)
            & goalie_away_found.fillna(False).astype(bool)
        )
        games_with_both_found = int(both_found_series.sum())
    else:
        games_with_both_found = 0
    any_non_zero = bool(goalie_adj.fillna(0.0).astype(float).ne(0.0).any())
    games_with_both_names = int((home_found & away_found).sum())
    any_non_zero_adj = bool(any_non_zero)
    if not any_non_zero_adj and games_with_both_found > 0:
        msg = (
            "goalie_adj was zero for all games; "
            f"games_with_both_names={games_with_both_names} "
            f"games_with_both_found={games_with_both_found} "
            f"any_non_zero_adj={any_non_zero_adj}"
        )
        if STRICT_SANITY:
            raise RuntimeError(msg)
        print(f"[NHL WARNING] {msg}")


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
