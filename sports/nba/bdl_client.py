import os
import time
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests


BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"
RECENT_GAMES_WINDOW = 10


def get_bdl_api_key() -> str:
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError("BALLDONTLIE_API_KEY environment variable is not set.")
    return api_key


def bdl_get(path, params=None, api_key=None, max_retries=5):
    if api_key is None:
        api_key = get_bdl_api_key()

    url = BALLDONTLIE_BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Authorization": api_key}
    params = dict(params or {})

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after is not None else 15
                print(f"[bdl_get] Rate limited (429) on {path}, attempt {attempt}/{max_retries}. Sleeping {wait}s...")
                time.sleep(wait)
                last_exc = RuntimeError("Rate limited by BallDontLie")
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout as e:
            last_exc = e
            print(f"[bdl_get] Timeout calling {path} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"[bdl_get] HTTP error calling {path} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(5)

    raise RuntimeError(f"Failed to GET {path} from BallDontLie after {max_retries} attempts") from last_exc


def season_start_year_for_date(d: date) -> int:
    # NBA season starts ~Oct; before Aug belongs to prior season year label for BDL
    return d.year - 1 if d.month < 8 else d.year


def fetch_games_for_date(game_date_str: str, api_key: str) -> pd.DataFrame:
    dt = datetime.strptime(game_date_str, "%m/%d/%Y").date()
    iso_date = dt.strftime("%Y-%m-%d")
    params = {"dates[]": iso_date, "per_page": 100}
    games_json = bdl_get("games", params=params, api_key=api_key)
    games = games_json.get("data", [])

    rows = []
    for g in games:
        home_team = g["home_team"]
        away_team = g["visitor_team"]
        rows.append({
            "GAME_ID": g["id"],
            "HOME_TEAM_NAME": home_team["full_name"],
            "AWAY_TEAM_NAME": away_team["full_name"],
            "HOME_TEAM_ID": home_team["id"],
            "AWAY_TEAM_ID": away_team["id"],
            "GAME_DATE": g.get("date"),
            "HOME_SCORE": g.get("home_team_score"),
            "AWAY_SCORE": g.get("visitor_team_score"),
        })
    return pd.DataFrame(rows)


def get_team_last_game_date(team_id: int, game_date_obj: date, season_year: int, api_key: str) -> Optional[date]:
    iso_date = game_date_obj.strftime("%Y-%m-%d")
    params = {"seasons[]": season_year, "team_ids[]": team_id, "end_date": iso_date, "per_page": 100}
    cursor = None
    last_date = None

    while True:
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        games_json = bdl_get("games", params=params, api_key=api_key)
        games = games_json.get("data", [])
        meta = games_json.get("meta", {}) or {}
        cursor = meta.get("next_cursor")

        for g in games:
            g_date_str = g.get("date")
            if not g_date_str:
                continue
            g_date = datetime.fromisoformat(g_date_str.replace("Z", "+00:00")).date()
            if g_date >= game_date_obj:
                continue

            home_score = g.get("home_team_score", 0) or 0
            away_score = g.get("visitor_team_score", 0) or 0
            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue

            if (last_date is None) or (g_date > last_date):
                last_date = g_date

        if not cursor:
            break

    return last_date


def fetch_team_ratings_bdl(season_year: int, end_date_iso: str, api_key: str) -> pd.DataFrame:
    teams_json = bdl_get("teams", params={}, api_key=api_key)
    teams_data = teams_json.get("data", [])

    agg = {}
    games_by_team = {}
    for t in teams_data:
        tid = t["id"]
        agg[tid] = {
            "TEAM_NAME": t["full_name"],
            "gp": 0,
            "pts_for": 0,
            "pts_against": 0,
            "wins": 0,
            "losses": 0,
        }
        games_by_team[tid] = []

    params = {"seasons[]": season_year, "end_date": end_date_iso, "per_page": 100}
    cursor = None

    while True:
        if cursor is not None:
            params["cursor"] = cursor
        else:
            params.pop("cursor", None)

        games_json = bdl_get("games", params=params, api_key=api_key)
        games = games_json.get("data", [])
        meta = games_json.get("meta", {}) or {}
        cursor = meta.get("next_cursor")

        for g in games:
            home_team = g["home_team"]
            away_team = g["visitor_team"]

            home_id = home_team["id"]
            away_id = away_team["id"]
            home_score = g.get("home_team_score", 0) or 0
            away_score = g.get("visitor_team_score", 0) or 0

            if home_score == 0 and away_score == 0 and g.get("period", 0) == 0:
                continue

            g_date_str = g.get("date")
            try:
                g_date = datetime.fromisoformat(g_date_str.replace("Z", "+00:00")).date()
            except Exception:
                g_date = None

            if home_id in agg:
                agg[home_id]["gp"] += 1
                agg[home_id]["pts_for"] += home_score
                agg[home_id]["pts_against"] += away_score
                if g_date:
                    games_by_team[home_id].append({"date": g_date, "pts_for": home_score, "pts_against": away_score})

            if away_id in agg:
                agg[away_id]["gp"] += 1
                agg[away_id]["pts_for"] += away_score
                agg[away_id]["pts_against"] += home_score
                if g_date:
                    games_by_team[away_id].append({"date": g_date, "pts_for": away_score, "pts_against": home_score})

            if home_score > away_score:
                if home_id in agg:
                    agg[home_id]["wins"] += 1
                if away_id in agg:
                    agg[away_id]["losses"] += 1
            elif away_score > home_score:
                if away_id in agg:
                    agg[away_id]["wins"] += 1
                if home_id in agg:
                    agg[home_id]["losses"] += 1

        if not cursor:
            break

    rows = []
    for tid, rec in agg.items():
        gp = rec["gp"]
        wins = rec["wins"]
        losses = rec["losses"]
        w_pct = wins / gp if gp > 0 else 0.0
        or_p = rec["pts_for"] / gp if gp > 0 else 0.0
        dr_p = rec["pts_against"] / gp if gp > 0 else 0.0

        total_pts = rec["pts_for"] + rec["pts_against"]
        pace = total_pts / gp if gp > 0 else 0.0

        poss = max(total_pts, 1)
        off_eff = rec["pts_for"] / poss
        def_eff = rec["pts_against"] / poss

        recent_games = sorted(games_by_team[tid], key=lambda x: x["date"])[-RECENT_GAMES_WINDOW:]
        gp_recent = len(recent_games)

        if gp_recent > 0:
            pts_for_recent = sum(g["pts_for"] for g in recent_games)
            pts_against_recent = sum(g["pts_against"] for g in recent_games)
            total_pts_recent = pts_for_recent + pts_against_recent

            or_p_recent = pts_for_recent / gp_recent
            dr_p_recent = pts_against_recent / gp_recent
            pace_recent = total_pts_recent / gp_recent
            poss_recent = max(total_pts_recent, 1)
            off_eff_recent = pts_for_recent / poss_recent
            def_eff_recent = pts_against_recent / poss_recent
        else:
            or_p_recent, dr_p_recent, pace_recent = or_p, dr_p, pace
            off_eff_recent, def_eff_recent = off_eff, def_eff

        rows.append({
            "TEAM_ID": tid,
            "TEAM_NAME": rec["TEAM_NAME"],
            "GP": gp,
            "W": wins,
            "L": losses,
            "W_PCT": w_pct,
            "ORtg": or_p,
            "DRtg": dr_p,
            "PACE": pace,
            "OFF_EFF": off_eff,
            "DEF_EFF": def_eff,
            "ORtg_RECENT": or_p_recent,
            "DRtg_RECENT": dr_p_recent,
            "PACE_RECENT": pace_recent,
            "OFF_EFF_RECENT": off_eff_recent,
            "DEF_EFF_RECENT": def_eff_recent,
        })

    return pd.DataFrame(rows)

