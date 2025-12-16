# sports/nhl/model.py
from __future__ import annotations

import pandas as pd

from sports.common.elo import EloState, elo_win_prob, elo_update
from sports.common.scores_sources import fetch_recent_scores
from sports.common.odds_sources import fetch_odds_for_date_from_odds_api, SPORT_TO_ODDS_KEY


ELO_PATH = "results/elo_state_nhl.json"


def update_elo_from_recent_scores(days_from: int = 7) -> EloState:
    st = EloState.load(ELO_PATH)
    sport_key = SPORT_TO_ODDS_KEY["nhl"]

    events = fetch_recent_scores(sport_key=sport_key, days_from=days_from)

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

    st.save(ELO_PATH)
    return st


def run_daily_nhl(game_date_str: str) -> pd.DataFrame:
    st = update_elo_from_recent_scores(days_from=7)

    odds_dict, spreads_dict = fetch_odds_for_date_from_odds_api(
        game_date_str,
        sport_key=SPORT_TO_ODDS_KEY["nhl"],
        debug=True,
    )

    rows = []
    for (home, away), oi in odds_dict.items():
        p_home = elo_win_prob(st.get(home), st.get(away), home_adv=45.0)

        rows.append({
            "date": game_date_str,
            "home": home,
            "away": away,
            "model_home_prob": float(p_home),
            "home_ml": oi.get("home_ml"),
            "away_ml": oi.get("away_ml"),
            "home_spread": oi.get("home_spread"),
        })

    return pd.DataFrame(rows)

