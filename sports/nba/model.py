# sports/nba/model.py
from __future__ import annotations

import os
from datetime import datetime, date
import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.calibration import load_nba_calibrator, update_and_save_nba_calibration
from sports.nba.bdl_client import bdl_get, get_bdl_api_key, season_start_year_for_date

ELO_PATH = "results/elo_state_nba.json"
BACKFILL_MARKER_PATH = "results/.elo_state_nba_backfill_date.txt"

# ---- Tunables ----
HOME_ADV = 65.0           # Elo home advantage (tunable)
ELO_K = 20.0              # Elo update aggressiveness (tunable)

# Convert elo_diff -> spread uses calibration; still sanity-clamp output:
MAX_ABS_MODEL_SPREAD = 15.0

# Injury impact mapping (inj_points -> elo points)
INJ_ELO_PER_POINT = 18.0  # tune this
FORM_ELO_PER_NET = 3.0    # elo points per 1 pt net rating above league avg (tunable)
FORM_ELO_CLAMP = 50.0     # clamp for form-based adjustment

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(np.nan)


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


# -----------------------------
# Injuries (auto-detected)
# -----------------------------
def _load_nba_injuries():
    """
    Expected in sports/nba/injuries.py:
      - fetch_official_nba_injuries()
      - build_injury_list_for_team_nba(team, injuries_map)
      - injury_adjustment_points(home_inj, away_inj)
    """
    try:
        from sports.nba.injuries import (
            fetch_official_nba_injuries,
            build_injury_list_for_team_nba,
            injury_adjustment_points,
        )
        return fetch_official_nba_injuries, build_injury_list_for_team_nba, injury_adjustment_points
    except Exception as e:
        print(f"[nba injuries] NOTE: injuries module not available: {e}")
        return None, None, None


# -----------------------------
# Elo update from recent scores
# -----------------------------
def _parse_game_datetime(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None


def _fetch_finished_games_from_bdl(*, season_year: int, api_key: str) -> list[dict]:
    params = {"seasons[]": season_year, "per_page": 100}
    cursor = None
    games: list[dict] = []

    while True:
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        games_json = bdl_get("games", params=params, api_key=api_key)
        data = games_json.get("data", []) or []
        meta = games_json.get("meta", {}) or {}
        cursor = meta.get("next_cursor")

        for g in data:
            home_team = (g.get("home_team") or {}).get("full_name")
            away_team = (g.get("visitor_team") or {}).get("full_name")
            if not home_team or not away_team:
                continue

            home_score = _safe_float(g.get("home_team_score"), 0.0)
            away_score = _safe_float(g.get("visitor_team_score"), 0.0)
            if home_score == 0.0 and away_score == 0.0 and (g.get("period", 0) == 0):
                continue  # not completed

            dt = _parse_game_datetime(g.get("date"))
            date_key = dt.date().isoformat() if dt else str(g.get("date") or "")
            sort_ts = dt.timestamp() if dt else 0.0

            home = canon_team(home_team)
            away = canon_team(away_team)
            if not home or not away:
                continue

            games.append({
                "game_key": f"{date_key}|{home}|{away}",
                "home": home,
                "away": away,
                "home_score": float(home_score),
                "away_score": float(away_score),
                "sort_ts": sort_ts,
            })

        if not cursor:
            break

    return games


def _fetch_finished_games_from_odds_api(days_from: int = 3) -> list[dict]:
    sport_key = SPORT_TO_ODDS_KEY["nba"]
    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 3))

    games: list[dict] = []
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
            hs = score_map.get(home_raw)
            aw = score_map.get(away_raw)
            if hs is None:
                hs = score_map.get(home)
            if aw is None:
                aw = score_map.get(away)

            hs = float(hs)
            aw = float(aw)
        except Exception:
            continue

        dt = _parse_game_datetime(ev.get("commence_time"))
        date_key = dt.date().isoformat() if dt else str(ev.get("commence_time") or "")
        sort_ts = dt.timestamp() if dt else 0.0

        games.append({
            "game_key": f"{date_key}|{home}|{away}",
            "home": home,
            "away": away,
            "home_score": float(hs),
            "away_score": float(aw),
            "sort_ts": sort_ts,
        })
    return games


def backfill_nba_elo_state(
    *,
    season_year: int | None = None,
    bdl_api_key: str | None = None,
    force_full_rebuild: bool = False,
) -> EloState:
    today_str = datetime.utcnow().date().isoformat()
    if not force_full_rebuild and os.path.exists(ELO_PATH) and os.path.exists(BACKFILL_MARKER_PATH):
        try:
            with open(BACKFILL_MARKER_PATH, "r", encoding="utf-8") as f:
                last_run = f.read().strip()
            if last_run == today_str:
                print("[nba backfill] Backfill already executed today; using existing Elo state.")
                return EloState.load(ELO_PATH)
        except Exception:
            pass

    if season_year is None:
        season_year = season_start_year_for_date(date.today())

    if bdl_api_key is None:
        try:
            bdl_api_key = get_bdl_api_key()
        except Exception as e:
            print(f"[nba backfill] WARNING: no BallDontLie API key available: {e}")
            bdl_api_key = None

    games: list[dict] = []
    if bdl_api_key:
        try:
            games.extend(_fetch_finished_games_from_bdl(season_year=season_year, api_key=bdl_api_key))
        except Exception as e:
            print(f"[nba backfill] WARNING: failed to load season games from BallDontLie: {e}")

    try:
        games.extend(_fetch_finished_games_from_odds_api(days_from=3))
    except Exception as e:
        print(f"[nba backfill] WARNING: failed to load recent scores from Odds API: {e}")

    if not games:
        print("[nba backfill] No finished games fetched; returning existing Elo state.")
        return EloState.load(ELO_PATH)

    deduped = []
    seen_keys = set()
    for g in games:
        key = g.get("game_key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(g)

    deduped.sort(key=lambda x: x.get("sort_ts", 0.0))

    st = EloState(ratings={}, processed_games={})
    applied = 0
    for g in deduped:
        home = g.get("home")
        away = g.get("away")
        hs = g.get("home_score")
        aw = g.get("away_score")
        game_key = g.get("game_key")

        if not home or not away or hs is None or aw is None or not game_key:
            continue
        if st.is_processed(game_key):
            continue

        eh = st.get(home)
        ea = st.get(away)
        nh, na = elo_update(eh, ea, float(hs), float(aw), k=ELO_K, home_adv=HOME_ADV)
        st.set(home, nh)
        st.set(away, na)
        st.mark_processed(game_key)
        applied += 1

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)
    try:
        with open(BACKFILL_MARKER_PATH, "w", encoding="utf-8") as f:
            f.write(today_str)
    except Exception:
        pass

    print(f"[nba backfill] Applied {applied} finished games out of {len(deduped)} fetched for season {season_year}.")
    return st


def update_elo_from_recent_scores(days_from: int = 3, st: EloState | None = None) -> EloState:
    if st is None:
        st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nba"]

    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 3))

    for ev in events:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        scores = ev.get("scores")
        if not home_raw or not away_raw or not scores:
            continue

        home = canon_team(home_raw)
        away = canon_team(away_raw)

       dt = _parse_game_datetime(ev.get("commence_time"))
        date_key = dt.date().isoformat() if dt else str(ev.get("commence_time") or "")
        game_key = f"{date_key}|{home}|{away}"
        if st.is_processed(game_key):
            continue

        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = score_map.get(home_raw)
            aw = score_map.get(away_raw)
            if hs is None:
                hs = score_map.get(home)
            if aw is None:
                aw = score_map.get(away)

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


def _injuries_debug_summary(injuries_map: dict) -> None:
    """Print a small summary so we can confirm injuries are actually loading."""
    try:
        if not injuries_map:
            print("[nba injuries] loaded: 0 teams")
            return
        teams = list(injuries_map.keys())
        total_rows = 0
        for k in teams[:10]:
            total_rows += len(injuries_map.get(k, []) or [])
        print(f"[nba injuries] loaded teams: {len(teams)} | sample teams: {teams[:5]}")
        # total rows across ALL teams (cheap)
        tot = sum(len(v or []) for v in injuries_map.values())
        print(f"[nba injuries] total rows: {tot}")
    except Exception:
        pass


def _build_team_injuries(build_list_fn, team: str, injuries_map: dict):
    """
    Robust team lookup helper:
    - Try canonical team key
    - Try raw key
    - Try partial nickname fallback (handled in injuries.py too, but redundancy helps)
    """
    if build_list_fn is None or not injuries_map:
        return []
    try:
        return build_list_fn(team, injuries_map)
    except Exception:
        # If build_list itself throws, don't kill the run
        return []

def _build_form_adjustments(stats_df: pd.DataFrame | None) -> dict[str, float]:
    """
    Build per-team Elo bump based on recent net rating relative to league average.
    Uses recency-weighted ORtg/DRtg when available (via bdl_client).
    """
    if stats_df is None or len(stats_df) == 0:
        return {}

    try:
        team_col = None
        for cand in ["TEAM_NAME", "team", "TEAM"]:
            if cand in stats_df.columns:
                team_col = cand
                break
        if team_col is None:
            return {}

        off_col = "ORtg_RECENT" if "ORtg_RECENT" in stats_df.columns else "ORtg"
        def_col = "DRtg_RECENT" if "DRtg_RECENT" in stats_df.columns else "DRtg"
        if off_col not in stats_df.columns or def_col not in stats_df.columns:
            return {}

        df = stats_df[[team_col, off_col, def_col]].copy()
        df["net_recent"] = df[off_col].apply(_safe_float) - df[def_col].apply(_safe_float)
        nets = df["net_recent"].dropna().astype(float)
        if len(nets) == 0:
            return {}

        league_avg_net = float(nets.mean())
        adjs = {}
        for _, row in df.iterrows():
            team = canon_team(row.get(team_col))
            net_recent = _safe_float(row.get("net_recent"))
            if not team or net_recent is None or np.isnan(net_recent):
                continue

            centered = net_recent - league_avg_net
            elo_adj = centered * FORM_ELO_PER_NET
            elo_adj = _clamp(elo_adj, -FORM_ELO_CLAMP, FORM_ELO_CLAMP)
            adjs[team] = elo_adj
        return adjs
    except Exception:
        return {}

# -----------------------------
# Daily run
# -----------------------------
def run_daily_nba(
    game_date_str: str,
    *,
    odds_dict: dict,
    stats_df: pd.DataFrame | None = None,
    spreads_dict: dict | None = None,
    force_full_rebuild: bool = False,
    bdl_api_key: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    st = backfill_nba_elo_state(force_full_rebuild=force_full_rebuild, bdl_api_key=bdl_api_key)
    st = update_elo_from_recent_scores(days_from=3, st=st)
    form_adjustments = _build_form_adjustments(stats_df)
    # Calibration
    cal = load_nba_calibrator()
    cal = update_and_save_nba_calibration()

    # Injuries
    fetch_inj, build_list, inj_points_fn = _load_nba_injuries()
    injuries_map = {}
    if fetch_inj is not None:
        try:
            injuries_map = fetch_inj() or {}
        except Exception as e:
            print(f"[nba injuries] WARNING: failed to load injuries: {e}")
            injuries_map = {}

    _injuries_debug_summary(injuries_map)

    if not odds_dict:
        return pd.DataFrame(columns=[
            "date", "home", "away",
            "model_home_prob", "model_spread_home", "elo_diff", "inj_points",
            "home_ml", "away_ml", "home_spread",
        ])

    rows = []
    for (home_in, away_in), oi in (odds_dict or {}).items():
        home = canon_team(home_in)
        away = canon_team(away_in)

        eh = st.get(home)
        ea = st.get(away)
        home_form = float(form_adjustments.get(home, 0.0))
        away_form = float(form_adjustments.get(away, 0.0))

        inj_pts = 0.0
        inj_elo_adj = 0.0
 
        if inj_points_fn is not None and build_list is not None and injuries_map:
            try:
                home_inj = _build_team_injuries(build_list, home, injuries_map)
                away_inj = _build_team_injuries(build_list, away, injuries_map)

                inj_pts = float(inj_points_fn(home_inj, away_inj))  # + means away more hurt
                inj_pts = _clamp(inj_pts, -8.0, 8.0)                # stability clamp
                inj_elo_adj = float(inj_pts) * INJ_ELO_PER_POINT
            except Exception as e:
                # Important: keep run alive
                print(f"[nba injuries] WARNING: injury calc failed for {home} vs {away}: {e}")
                inj_pts = 0.0
                inj_elo_adj = 0.0

        eh_total = eh + inj_elo_adj + home_form
        ea_total = ea + away_form

        p_home = elo_win_prob(eh_total, ea_total, home_adv=HOME_ADV)

        elo_diff = (eh_total - ea_total) + HOME_ADV
        model_spread_home = cal.predict_spread(elo_diff)
        model_spread_home = _clamp(model_spread_home, -MAX_ABS_MODEL_SPREAD, MAX_ABS_MODEL_SPREAD)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "elo_diff": float(elo_diff),
            "inj_points": float(inj_pts),
            "home_ml": _safe_float((oi or {}).get("home_ml")),
            "away_ml": _safe_float((oi or {}).get("away_ml")),
            "home_spread": _safe_float((oi or {}).get("home_spread")),
        })

    return pd.DataFrame(rows)

def run_daily_probs_for_date(
    game_date_str: str = None,
    *,
    game_date: str = None,
    odds_dict: dict = None,
    spreads_dict: dict = None,   # older callers pass this
         stats_df: pd.DataFrame | None = None,
    force_full_rebuild: bool = False,
    api_key: str | None = None,
    **kwargs,                    # swallow any future extra args safely
) -> pd.DataFrame:
    """
    Backwards-compatible alias for older code paths.
    """
    date_in = game_date if game_date is not None else game_date_str
    if date_in is None:
        raise ValueError("Must provide game_date or game_date_str")

    return run_daily_nba(
        str(date_in),
        odds_dict=(odds_dict or {}),
        stats_df=stats_df,
        spreads_dict=spreads_dict,
        force_full_rebuild=force_full_rebuild,
        bdl_api_key=api_key,
        **kwargs,
    )
