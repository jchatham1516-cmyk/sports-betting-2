from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from sports_betting.sports.nba.features import NBA_FEATURE_COLUMNS, build_nba_features


@dataclass
class TrainedNBAModels:
    moneyline: CalibratedClassifierCV
    spread: CalibratedClassifierCV
    totals: CalibratedClassifierCV
    metrics: dict[str, dict[str, float]]
    feature_importance: dict[str, dict[str, float]]


TARGETS = {
    "moneyline": "home_win",
    "spread": "home_cover",
    "totals": "over_hit",
}


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["date", "game_id"], kind="mergesort").reset_index(drop=True)
    if "season" in ordered.columns:
        seasons = ordered["season"].dropna().unique().tolist()
        if len(seasons) >= 2:
            cutoff = seasons[-1]
            train_df = ordered[ordered["season"] != cutoff]
            valid_df = ordered[ordered["season"] == cutoff]
            if len(train_df) > 0 and len(valid_df) > 0:
                return train_df, valid_df

    split_idx = int(len(ordered) * 0.8)
    split_idx = max(1, min(split_idx, len(ordered) - 1))
    return ordered.iloc[:split_idx], ordered.iloc[split_idx:]


def _safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    if y_true.nunique(dropna=True) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _train_single_market(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
    random_state: int,
) -> tuple[CalibratedClassifierCV, dict[str, float], dict[str, float]]:
    x_train = train_df[NBA_FEATURE_COLUMNS]
    y_train = train_df[target].astype(int)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=350,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=random_state,
    )
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv=3)
    calibrated.fit(x_train, y_train)

    x_valid = valid_df[NBA_FEATURE_COLUMNS]
    y_valid = valid_df[target].astype(int)
    valid_prob = pd.Series(calibrated.predict_proba(x_valid)[:, 1], index=valid_df.index)

    metrics = {
        "auc": _safe_auc(y_valid, valid_prob),
        "log_loss": float(log_loss(y_valid, valid_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_valid, valid_prob)),
        "sample_size_train": float(len(train_df)),
        "sample_size_valid": float(len(valid_df)),
    }

    # HistGradientBoosting does not expose split importances directly; use permutation proxy by correlation.
    importances = {
        col: float(abs(train_df[col].corr(train_df[target]))) if train_df[col].notna().any() else 0.0
        for col in NBA_FEATURE_COLUMNS
    }

    return calibrated, metrics, importances


def train_nba_models(df: pd.DataFrame, random_state: int = 42) -> TrainedNBAModels:
    """Train calibrated NBA models using time-based validation splits."""

    features_df = build_nba_features(df)
    required_cols = {"date", "game_id", *TARGETS.values()}
    missing_required = required_cols.difference(features_df.columns)
    if missing_required:
        raise ValueError(f"Missing required training columns: {sorted(missing_required)}")

    clean = features_df.dropna(subset=list(TARGETS.values())).copy()
    for target_col in TARGETS.values():
        clean[target_col] = clean[target_col].astype(int)

    train_df, valid_df = _time_split(clean)

    trained: dict[str, CalibratedClassifierCV] = {}
    metrics: dict[str, dict[str, float]] = {}
    importance: dict[str, dict[str, float]] = {}
    for i, (market, target) in enumerate(TARGETS.items()):
        trained_model, market_metrics, market_importance = _train_single_market(
            train_df=train_df,
            valid_df=valid_df,
            target=target,
            random_state=random_state + i,
        )
        trained[market] = trained_model
        metrics[market] = market_metrics
        importance[market] = market_importance

    return TrainedNBAModels(
        moneyline=trained["moneyline"],
        spread=trained["spread"],
        totals=trained["totals"],
        metrics=metrics,
        feature_importance=importance,
    )


def save_trained_nba_models(models: TrainedNBAModels, model_dir: Path | str) -> None:
    import joblib

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.moneyline, model_path / "nba_moneyline_model.pkl")
    joblib.dump(models.spread, model_path / "nba_spread_model.pkl")
    joblib.dump(models.totals, model_path / "nba_totals_model.pkl")


def load_trained_nba_models(model_dir: Path | str) -> dict[str, Any] | None:
    import joblib

    model_path = Path(model_dir)
    artifacts = {
        "moneyline": model_path / "nba_moneyline_model.pkl",
        "spread": model_path / "nba_spread_model.pkl",
        "totals": model_path / "nba_totals_model.pkl",
    }
    if not all(path.exists() for path in artifacts.values()):
        return None

    return {name: joblib.load(path) for name, path in artifacts.items()}
