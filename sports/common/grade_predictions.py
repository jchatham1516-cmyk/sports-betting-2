import pandas as pd
import requests
from datetime import datetime

SPORT_ENDPOINTS = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
}

def fetch_espn_results(sport, date_str):
    date_fmt = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    url = f"{SPORT_ENDPOINTS[sport]}?dates={date_fmt}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def extract_games(json_data):
    games = []
    for ev in json_data.get("events", []):
        comp = ev["competitions"][0]
        if not comp["status"]["type"]["completed"]:
            continue
        home = away = None
        for c in comp["competitors"]:
            team = c["team"]["displayName"]
            score = float(c["score"])
            if c["homeAway"] == "home":
                home = (team, score)
            else:
                away = (team, score)
        if home and away:
            games.append({
                "home": home[0],
                "away": away[0],
                "home_score": home[1],
                "away_score": away[1],
            })
    return games

def grade_row(row, games):
    rec = str(row["primary_recommendation"]).upper()
    if "NO BET" in rec:
        return None

    for g in games:
        if g["home"] == row["home"] and g["away"] == row["away"]:
            home_win = g["home_score"] > g["away_score"]
            away_win = g["away_score"] > g["home_score"]

            if "HOME ML" in rec:
                return home_win
            if "AWAY ML" in rec:
                return away_win

            # ATS (spread)
            if "ATS" in rec:
                spread = float(row["home_spread"])
                adj_home = g["home_score"] + spread
                if "HOME" in rec:
                    return adj_home > g["away_score"]
                if "AWAY" in rec:
                    return adj_home < g["away_score"]
    return None

def grade_file(csv_path, sport):
    df = pd.read_csv(csv_path)
    results = fetch_espn_results(sport, df.iloc[0]["date"])
    games = extract_games(results)

    df["graded"] = df.apply(lambda r: grade_row(r, games), axis=1)
    df["graded"] = df["graded"].map({True: "WIN", False: "LOSS", None: "SKIP"})

    out = csv_path.replace(".csv", "_graded.csv")
    df.to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    grade_file("predictions_nba_2026-01-02.csv", "nba")
