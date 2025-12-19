# sports/nfl/model.py
from __future__ import annotations

import os
import pandas as pd

from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.nfl.injuries import fetch_espn_nfl_injuries, build_injury_list_for_team_nfl, injury_adjustment_points
from sports.common.teams import canon_team

ELO_PATH = "results/elo_state_nfl.json"
inj_pts = max(min(inj_pts, 8.0), -8.0)

def update_elo_from_recent_scores(days_from: int = 3) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nfl"]

    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 3))
home = canon_team(home)
away = canon_team(away)

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        scores = ev.get("scores")
        if not home or not away or not scores:
            continue

        game_key = f"{ev.get('id','')}|{ev.get('commence_time','')}|{home}|{away}"
        if st.is_processed(game_key):
            continue

        score_map = {s.get("name"): s.get("score") for s in scores if s.get("name")}
        try:
            hs = float(score_map.get(home))
            aw = float(score_map.get(away))
        except Exception:
            continue

        eh = st.get(home)
        ea = st.get(away)
        nh, na = elo_update(eh, ea, hs, aw, k=20.0, home_adv=55.0)
        st.set(home, nh)
        st.set(away, na)
        st.mark_processed(game_key)

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)
    return st

def run_daily_nfl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=3)

    injuries_map = {}
    try:
        injuries_map = fetch_espn_nfl_injuries()
    except Exception as e:
        print(f"[nfl injuries] WARNING: failed to load ESPN injuries: {e}")

    HOME_ADV = 55.0
    ELO_PER_POINT = 25.0

    # Tunable: how many Elo points per “injury point”
    INJ_ELO_PER_POINT = 18.0

    rows = []
    for (home, away), oi in (odds_dict or {}).items():
        eh = st.get(home)
        ea = st.get(away)

        home_inj = build_injury_list_for_team_nfl(home, injuries_map)
        away_inj = build_injury_list_for_team_nfl(away, injuries_map)
        inj_pts = injury_adjustment_points(home_inj, away_inj)  # + means away more hurt

        inj_elo_adj = inj_pts * INJ_ELO_PER_POINT

        p_home = elo_win_prob(eh + inj_elo_adj, ea, home_adv=HOME_ADV)

        elo_diff = ((eh + inj_elo_adj) - ea) + HOME_ADV
        model_spread_home = -(elo_diff / ELO_PER_POINT)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "inj_points": float(inj_pts),   # optional debug
            "home_ml": oi.get("home_ml"),
            "away_ml": oi.get("away_ml"),
            "home_spread": oi.get("home_spread"),
        })

    return pd.DataFrame(rows)
