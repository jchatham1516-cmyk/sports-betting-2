from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

NBA_FEATURE_COLUMNS = [
    "elo_diff",
    "rest_diff",
    "travel_fatigue_diff",
    "injury_impact_diff",
    "net_rating_diff",
    "pace_diff",
    "last5_net_rating_diff",
    "last10_net_rating_diff",
    "market_prob_home",
]

_NUMERIC_LIMITS = {
    "elo_diff": (-400.0, 400.0),
    "rest_diff": (-5.0, 5.0),
    "travel_fatigue_diff": (-10.0, 10.0),
    "injury_impact_diff": (-15.0, 15.0),
    "net_rating_diff": (-40.0, 40.0),
    "pace_diff": (-20.0, 20.0),
    "last5_net_rating_diff": (-50.0, 50.0),
    "last10_net_rating_diff": (-40.0, 40.0),
    "market_prob_home": (0.01, 0.99),
}


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clip_column(df: pd.DataFrame, column: str) -> pd.Series:
    lower, upper = _NUMERIC_LIMITS[column]
    return df[column].clip(lower=lower, upper=upper)


def build_nba_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic NBA model features for both training and daily inference.

    The function only uses pre-game inputs and applies conservative cleaning:
    missing handling, clipping, and numeric coercion.
    """

    features = df.copy()

    # Base diffs can be provided directly or derived from home/away columns.
    pair_diffs = [
        ("elo_home", "elo_away", "elo_diff"),
        ("rest_days_home", "rest_days_away", "rest_diff"),
        ("injury_impact_home", "injury_impact_away", "injury_impact_diff"),
        ("net_rating_home", "net_rating_away", "net_rating_diff"),
        ("pace_home", "pace_away", "pace_diff"),
        ("last5_net_rating_home", "last5_net_rating_away", "last5_net_rating_diff"),
        ("last10_net_rating_home", "last10_net_rating_away", "last10_net_rating_diff"),
    ]

    for home_col, away_col, diff_col in pair_diffs:
        if diff_col not in features.columns and home_col in features.columns and away_col in features.columns:
            features[diff_col] = pd.to_numeric(features[home_col], errors="coerce") - pd.to_numeric(
                features[away_col], errors="coerce"
            )

    if "travel_fatigue_diff" not in features.columns:
        home_fatigue = (
            pd.to_numeric(features.get("travel_distance_home", 0.0), errors="coerce").fillna(0.0) / 1000.0
            + pd.to_numeric(features.get("timezone_shift_home", 0.0), errors="coerce").abs().fillna(0.0)
            + pd.to_numeric(features.get("road_trip_length_home", 0.0), errors="coerce").fillna(0.0) * 0.35
            + pd.to_numeric(features.get("back_to_back_home", 0.0), errors="coerce").fillna(0.0) * 1.0
            + pd.to_numeric(features.get("three_in_four_home", 0.0), errors="coerce").fillna(0.0) * 0.6
        )
        away_fatigue = (
            pd.to_numeric(features.get("travel_distance_away", 0.0), errors="coerce").fillna(0.0) / 1000.0
            + pd.to_numeric(features.get("timezone_shift_away", 0.0), errors="coerce").abs().fillna(0.0)
            + pd.to_numeric(features.get("road_trip_length_away", 0.0), errors="coerce").fillna(0.0) * 0.35
            + pd.to_numeric(features.get("back_to_back_away", 0.0), errors="coerce").fillna(0.0) * 1.0
            + pd.to_numeric(features.get("three_in_four_away", 0.0), errors="coerce").fillna(0.0) * 0.6
        )
        features["travel_fatigue_diff"] = home_fatigue - away_fatigue

    features = _ensure_numeric(features, NBA_FEATURE_COLUMNS)

    for col in NBA_FEATURE_COLUMNS:
        features[col] = _clip_column(features, col)
        fill_value = 0.5 if col == "market_prob_home" else 0.0
        features[col] = features[col].fillna(fill_value)

    return features
