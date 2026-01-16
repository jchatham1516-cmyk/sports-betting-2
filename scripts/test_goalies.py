from __future__ import annotations

from datetime import date

from sports.nhl.goalies import get_starting_goalies
from sports.nhl.model import run_daily_probs_for_date


def main() -> None:
    date_key = date.today().isoformat()
    goalies = get_starting_goalies(date_key)
    with_names = {team: info for team, info in goalies.items() if info.goalie_name}
    print(f"[test_goalies] date_key={date_key} parsed_goalies={len(goalies)} with_names={len(with_names)}")
    for team, info in list(with_names.items())[:5]:
        print(f"[test_goalies] sample team={team} goalie={info.goalie_name} status={info.status} source={info.source}")

    teams = list(with_names.keys())
    if len(teams) < 2:
        print("[test_goalies] ERROR: need at least two teams with goalie names to build odds stub.")
        return

    home, away = teams[0], teams[1]
    odds_dict = {
        (home, away): {
            "home_ml": -110,
            "away_ml": -110,
        }
    }

    df = run_daily_probs_for_date(date_key, odds_dict=odds_dict)
    if df.empty:
        print("[test_goalies] ERROR: run_daily_probs_for_date returned empty dataframe.")
        return

    row = df.iloc[0]
    print(
        "[test_goalies] model output "
        f"home={row.get('home_team')} away={row.get('away_team')} "
        f"goalie_home_name={row.get('goalie_home_name')} goalie_away_name={row.get('goalie_away_name')} "
        f"goalie_adj={row.get('goalie_adj')}")

    non_zero = df["goalie_adj"].abs().gt(0).any()
    print(f"[test_goalies] goalie_adj_non_zero={non_zero}")


if __name__ == "__main__":
    main()
