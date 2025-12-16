# sports/nhl/model.py
from __future__ import annotations

import os
import pandas as pd

from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.nhl.injuries import (
    fetch_espn_nhl_injuries,
    build_injury_list_for_team_nhl,
    injury_adjustment_points,
)

ELO_PATH = "results/elo_state_nhl.json"


def update_elo_from_recent_scores(days_from: int = 3) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nhl"]

    events = fetch_recent_scores(sport_key=sport_key, days_from=min(int(days_from), 3))

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
        nh, na = elo_update(eh, ea, hs, aw, k=18.0, home_adv=45.0)
        st.set(home, nh)
        st.set(away, na)
        st.mark_processed(game_key)

    os.makedirs("results", exist_ok=True)
    st.save(ELO_PATH)
    return st


def run_daily_nhl(game_date_str: str, *, odds_dict: dict) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=3)

    HOME_ADV = 45.0
    ELO_PER_GOAL = 30.0  # simple spread-ish mapping (tunable)
    INJ_ELO_PER_POINT = 22.0  # tunable for NHL

    # Load injuries once (do NOT do this per game)
    injuries_map = {}
    try:
        injuries_map = fetch_espn_nhl_injuries()
    except Exception as e:
        print(f"[nhl injuries] WARNING: failed to load ESPN injuries: {e}")

    rows = []
    for (home, away), oi in (odds_dict or {}).items():
        eh = st.get(home)
        ea = st.get(away)

        # Injuries for this matchup
        home_inj = build_injury_list_for_team_nhl(home, injuries_map)
        away_inj = build_injury_list_for_team_nhl(away, injuries_map)

        # +inj_pts means away is more hurt => helps home
        inj_pts = injury_adjustment_points(home_inj, away_inj)
        inj_elo_adj = inj_pts * INJ_ELO_PER_POINT

        p_home = elo_win_prob(eh + inj_elo_adj, ea, home_adv=HOME_ADV)

        elo_diff = ((eh + inj_elo_adj) - ea) + HOME_ADV
        model_spread_home = -(elo_diff / ELO_PER_GOAL)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "model_spread_home": float(model_spread_home),
            "inj_points": float(inj_pts),
            "home_ml": oi.get("home_ml"),
            "away_ml": oi.get("away_ml"),
            "home_spread": oi.get("home_spread"),
        })

    return pd.DataFrame(rows)
