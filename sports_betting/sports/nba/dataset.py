from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

HISTORICAL_PATH = Path("sports_betting/data/historical/nba_historical.csv")
PROCESSED_DIR = Path("sports_betting/data/processed")

NBA_HISTORICAL_COLUMNS = [
    "date",
    "season",
    "game_id",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
    "home_cover",
    "over_hit",
    "margin",
    "total_points",
    "closing_moneyline_home",
    "closing_moneyline_away",
    "closing_spread_home",
    "closing_total",
    "market_prob_home",
    "market_prob_away",
    "elo_home",
    "elo_away",
    "elo_diff",
    "rest_days_home",
    "rest_days_away",
    "rest_diff",
    "back_to_back_home",
    "back_to_back_away",
    "three_in_four_home",
    "three_in_four_away",
    "travel_distance_home",
    "travel_distance_away",
    "timezone_shift_home",
    "timezone_shift_away",
    "road_trip_length_home",
    "road_trip_length_away",
    "travel_fatigue_diff",
    "injury_impact_home",
    "injury_impact_away",
    "injury_impact_diff",
    "starter_out_count_home",
    "starter_out_count_away",
    "star_player_out_home",
    "star_player_out_away",
    "off_rating_home",
    "off_rating_away",
    "def_rating_home",
    "def_rating_away",
    "net_rating_home",
    "net_rating_away",
    "net_rating_diff",
    "pace_home",
    "pace_away",
    "pace_diff",
    "true_shooting_home",
    "true_shooting_away",
    "true_shooting_diff",
    "turnover_rate_home",
    "turnover_rate_away",
    "turnover_rate_diff",
    "rebound_rate_home",
    "rebound_rate_away",
    "rebound_rate_diff",
    "last5_net_rating_home",
    "last5_net_rating_away",
    "last5_net_rating_diff",
    "last10_net_rating_home",
    "last10_net_rating_away",
    "last10_net_rating_diff",
]


def _empty_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=NBA_HISTORICAL_COLUMNS)


def _extract_closing_markets(markets: list[dict[str, Any]], home_team: str, away_team: str) -> dict[str, float | None]:
    closing = {
        "closing_moneyline_home": None,
        "closing_moneyline_away": None,
        "closing_spread_home": None,
        "closing_total": None,
    }
    for market in markets:
        mkt = market.get("market")
        side = market.get("side", "")
        if mkt == "moneyline" and side == home_team:
            closing["closing_moneyline_home"] = market.get("odds")
        elif mkt == "moneyline" and side == away_team:
            closing["closing_moneyline_away"] = market.get("odds")
        elif mkt == "spread" and side == home_team:
            closing["closing_spread_home"] = market.get("line")
        elif mkt == "total" and str(side).lower() == "over":
            closing["closing_total"] = market.get("line")
    return closing


def _american_to_prob(odds: float | int | None) -> float | None:
    if odds is None:
        return None
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def _from_processed_snapshots() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(PROCESSED_DIR.glob("nba_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for game in payload.get("games", []):
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            ratings = game.get("meta", {}).get("team_ratings", {})
            home_rt = ratings.get(home_team, {}) if isinstance(ratings, dict) else {}
            away_rt = ratings.get(away_team, {}) if isinstance(ratings, dict) else {}

            markets = _extract_closing_markets(game.get("markets", []), home_team, away_team)
            market_prob_home = _american_to_prob(markets["closing_moneyline_home"])
            market_prob_away = _american_to_prob(markets["closing_moneyline_away"])

            row = {
                "date": str(game.get("date", ""))[:10],
                "season": str(game.get("date", ""))[:4],
                "game_id": game.get("event_id"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": None,
                "away_score": None,
                "home_win": None,
                "home_cover": None,
                "over_hit": None,
                "margin": None,
                "total_points": None,
                "market_prob_home": market_prob_home,
                "market_prob_away": market_prob_away,
                "elo_home": home_rt.get("elo", home_rt.get("power_rating")),
                "elo_away": away_rt.get("elo", away_rt.get("power_rating")),
                "rest_days_home": home_rt.get("rest_days"),
                "rest_days_away": away_rt.get("rest_days"),
                "back_to_back_home": home_rt.get("back_to_back"),
                "back_to_back_away": away_rt.get("back_to_back"),
                "three_in_four_home": home_rt.get("three_in_four"),
                "three_in_four_away": away_rt.get("three_in_four"),
                "travel_distance_home": home_rt.get("travel_distance"),
                "travel_distance_away": away_rt.get("travel_distance"),
                "timezone_shift_home": home_rt.get("timezone_shift"),
                "timezone_shift_away": away_rt.get("timezone_shift"),
                "road_trip_length_home": home_rt.get("road_trip_length"),
                "road_trip_length_away": away_rt.get("road_trip_length"),
                "injury_impact_home": home_rt.get("injury_impact"),
                "injury_impact_away": away_rt.get("injury_impact"),
                "starter_out_count_home": home_rt.get("starter_out_count"),
                "starter_out_count_away": away_rt.get("starter_out_count"),
                "star_player_out_home": home_rt.get("star_player_out"),
                "star_player_out_away": away_rt.get("star_player_out"),
                "off_rating_home": home_rt.get("off_rating"),
                "off_rating_away": away_rt.get("off_rating"),
                "def_rating_home": home_rt.get("def_rating"),
                "def_rating_away": away_rt.get("def_rating"),
                "net_rating_home": home_rt.get("net_rating"),
                "net_rating_away": away_rt.get("net_rating"),
                "pace_home": home_rt.get("pace"),
                "pace_away": away_rt.get("pace"),
                "true_shooting_home": home_rt.get("true_shooting"),
                "true_shooting_away": away_rt.get("true_shooting"),
                "turnover_rate_home": home_rt.get("turnover_rate"),
                "turnover_rate_away": away_rt.get("turnover_rate"),
                "rebound_rate_home": home_rt.get("rebound_rate"),
                "rebound_rate_away": away_rt.get("rebound_rate"),
                "last5_net_rating_home": home_rt.get("last5_net_rating"),
                "last5_net_rating_away": away_rt.get("last5_net_rating"),
                "last10_net_rating_home": home_rt.get("last10_net_rating"),
                "last10_net_rating_away": away_rt.get("last10_net_rating"),
                **markets,
            }
            rows.append(row)

    if not rows:
        return _empty_dataset()
    return pd.DataFrame(rows)


def _compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    numeric_cols = [c for c in frame.columns if c.endswith("_home") or c.endswith("_away") or c.endswith("_diff")]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    diffs = [
        ("elo_home", "elo_away", "elo_diff"),
        ("rest_days_home", "rest_days_away", "rest_diff"),
        ("injury_impact_home", "injury_impact_away", "injury_impact_diff"),
        ("net_rating_home", "net_rating_away", "net_rating_diff"),
        ("pace_home", "pace_away", "pace_diff"),
        ("true_shooting_home", "true_shooting_away", "true_shooting_diff"),
        ("turnover_rate_home", "turnover_rate_away", "turnover_rate_diff"),
        ("rebound_rate_home", "rebound_rate_away", "rebound_rate_diff"),
        ("last5_net_rating_home", "last5_net_rating_away", "last5_net_rating_diff"),
        ("last10_net_rating_home", "last10_net_rating_away", "last10_net_rating_diff"),
    ]
    for home_col, away_col, diff_col in diffs:
        if diff_col not in frame.columns:
            frame[diff_col] = None
        frame[diff_col] = frame[home_col] - frame[away_col]

    frame["travel_fatigue_diff"] = (
        frame["travel_distance_home"].fillna(0) / 1000.0
        + frame["timezone_shift_home"].abs().fillna(0)
        + frame["road_trip_length_home"].fillna(0) * 0.35
        + frame["back_to_back_home"].fillna(0)
        + frame["three_in_four_home"].fillna(0) * 0.6
        - (frame["travel_distance_away"].fillna(0) / 1000.0)
        - frame["timezone_shift_away"].abs().fillna(0)
        - frame["road_trip_length_away"].fillna(0) * 0.35
        - frame["back_to_back_away"].fillna(0)
        - frame["three_in_four_away"].fillna(0) * 0.6
    )

    frame["margin"] = frame["home_score"] - frame["away_score"]
    frame["total_points"] = frame["home_score"] + frame["away_score"]
    frame["home_win"] = frame["margin"].apply(lambda v: None if pd.isna(v) else int(v > 0))
    frame["home_cover"] = (frame["margin"] + frame["closing_spread_home"]).apply(
        lambda v: None if pd.isna(v) else int(v > 0)
    )
    frame["over_hit"] = (frame["total_points"] - frame["closing_total"]).apply(lambda v: None if pd.isna(v) else int(v > 0))

    return frame


def build_nba_historical_dataset(output_path: Path | str = HISTORICAL_PATH) -> pd.DataFrame:
    """Build and persist the NBA historical dataset used for model training."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(output_path) if output_path.exists() else _empty_dataset()
    incremental_df = _from_processed_snapshots()
    merged = pd.concat([base_df, incremental_df], ignore_index=True)

    if merged.empty:
        merged = _empty_dataset()

    for col in NBA_HISTORICAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    merged = _compute_derived_fields(merged)
    merged = merged[NBA_HISTORICAL_COLUMNS].drop_duplicates(subset=["game_id"], keep="last")
    merged = merged.sort_values(["date", "game_id"], kind="mergesort").reset_index(drop=True)
    merged.to_csv(output_path, index=False)
    return merged
