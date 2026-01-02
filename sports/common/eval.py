"""Evaluation helpers for model sanity checks and calibration reporting."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.teams import canon_team


def _find_score_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort detection of home/away score columns in results frames."""
    score_candidates = [
        "home_score",
        "home_team_score",
        "score_home",
        "home_pts",
    ]
    opp_candidates = [
        "away_score",
        "away_team_score",
        "score_away",
        "away_pts",
    ]

    home_col = next((c for c in score_candidates if c in df.columns), None)
    away_col = next((c for c in opp_candidates if c in df.columns), None)
    return home_col, away_col


def brier_score(y_true: Iterable[float], p: Iterable[float]) -> float:
    y_arr = np.array(list(y_true), dtype=float)
    p_arr = np.array(list(p), dtype=float)
    mask = ~np.isnan(y_arr) & ~np.isnan(p_arr)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y_arr[mask] - p_arr[mask]) ** 2))


def calibration_table(y_true: Iterable[float], p: Iterable[float], step: float = 0.05) -> pd.DataFrame:
    y_arr = np.array(list(y_true), dtype=float)
    p_arr = np.array(list(p), dtype=float)
    mask = ~np.isnan(y_arr) & ~np.isnan(p_arr)
    y_arr = y_arr[mask]
    p_arr = p_arr[mask]

    buckets: List[Dict[str, float]] = []
    edges = np.arange(0.0, 1.0 + step, step)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            in_bucket = (p_arr >= lo) & (p_arr <= hi)
        else:
            in_bucket = (p_arr >= lo) & (p_arr < hi)
        if in_bucket.any():
            bucket_ps = p_arr[in_bucket]
            bucket_y = y_arr[in_bucket]
            buckets.append(
                {
                    "bucket": f"{lo:.2f}-{hi:.2f}",
                    "bucket_count": int(len(bucket_ps)),
                    "avg_pred_prob": float(np.mean(bucket_ps)),
                    "actual_win_rate": float(np.mean(bucket_y)),
                    "diff": float(np.mean(bucket_ps) - np.mean(bucket_y)),
                }
            )
        else:
            buckets.append(
                {
                    "bucket": f"{lo:.2f}-{hi:.2f}",
                    "bucket_count": 0,
                    "avg_pred_prob": float("nan"),
                    "actual_win_rate": float("nan"),
                    "diff": float("nan"),
                }
            )
    return pd.DataFrame(buckets)


def summarize_edges(df: pd.DataFrame) -> Dict[str, object]:
    """Compute edge sanity stats from a predictions DataFrame."""
    edges = None
    if "model_home_prob" in df.columns and "market_home_prob" in df.columns:
        edges = df["model_home_prob"].astype(float) - df["market_home_prob"].astype(float)
    elif "edge_home" in df.columns:
        edges = df["edge_home"].astype(float)

    if edges is None:
        return {
            "mean_abs_edge": float("nan"),
            "median_abs_edge": float("nan"),
            "p90_abs_edge": float("nan"),
            "max_abs_edge": float("nan"),
            "over_0_15_rate": float("nan"),
            "over_0_25_rate": float("nan"),
            "warnings": ["No edge columns found (expected model_home_prob vs market_home_prob)."],
            "abs_edges": pd.Series(dtype=float),
        }

    abs_edges = edges.abs()
    clean_edges = abs_edges.dropna()
    mean_abs_edge = float(clean_edges.mean()) if not clean_edges.empty else float("nan")
    median_abs_edge = float(clean_edges.median()) if not clean_edges.empty else float("nan")
    p90_abs_edge = float(np.nanpercentile(clean_edges, 90)) if not clean_edges.empty else float("nan")
    max_abs_edge = float(clean_edges.max()) if not clean_edges.empty else float("nan")
    over_015 = float((clean_edges > 0.15).mean()) if not clean_edges.empty else float("nan")
    over_025 = float((clean_edges > 0.25).mean()) if not clean_edges.empty else float("nan")

    warnings: List[str] = []
    if not math.isnan(over_015) and over_015 > 0.05:
        warnings.append(f"Warning: {over_015*100:.1f}% of games have |edge|>0.15.")
    if not math.isnan(over_025) and over_025 > 0:
        warnings.append("Warning: at least one game has |edge|>0.25.")

    return {
        "mean_abs_edge": mean_abs_edge,
        "median_abs_edge": median_abs_edge,
        "p90_abs_edge": p90_abs_edge,
        "max_abs_edge": max_abs_edge,
        "over_0_15_rate": over_015,
        "over_0_25_rate": over_025,
        "warnings": warnings,
        "abs_edges": abs_edges,
    }


def _prepare_results(df_results: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df_results is None or df_results.empty:
        return None
    if "home" not in df_results.columns or "away" not in df_results.columns:
        return None

    home_col, away_col = _find_score_columns(df_results)
    if home_col is None or away_col is None:
        return None

    df_res = df_results.copy()
    df_res["home_key"] = df_res["home"].apply(canon_team)
    df_res["away_key"] = df_res["away"].apply(canon_team)
    df_res["actual_home_win"] = df_res.apply(
        lambda r: float(r[home_col] > r[away_col]) if not (pd.isna(r[home_col]) or pd.isna(r[away_col])) else float("nan"),
        axis=1,
    )
    return df_res[["home_key", "away_key", "actual_home_win"]]


def evaluate_predictions(
    df_preds: pd.DataFrame,
    df_results: Optional[pd.DataFrame] = None,
    *,
    sport: str = "nba",
    run_date_str: Optional[str] = None,
) -> pd.DataFrame:
    """Print evaluation summary for a set of predictions.

    Returns a single-row DataFrame suitable for saving to CSV.
    """
    if df_preds is None:
        df_preds = pd.DataFrame()
    preds = df_preds.copy()
    preds["home_key"] = preds.get("home", pd.Series(dtype=str)).apply(canon_team)
    preds["away_key"] = preds.get("away", pd.Series(dtype=str)).apply(canon_team)

    edge_summary = summarize_edges(preds)

    print("\n[eval] Market vs. model edge sanity check:")
    print(
        f" mean |edge|={edge_summary['mean_abs_edge']:.4f}"
        f" | median |edge|={edge_summary['median_abs_edge']:.4f}"
        f" | p90 |edge|={edge_summary['p90_abs_edge']:.4f}"
        f" | max |edge|={edge_summary['max_abs_edge']:.4f}"
    )
    print(
        f" share(|edge|>0.15)={edge_summary['over_0_15_rate']:.3f}"
        f" | share(|edge|>0.25)={edge_summary['over_0_25_rate']:.3f}"
    )
    for w in edge_summary.get("warnings", []):
        print(f"  -> {w}")

    brier = float("nan")
    calib_df: Optional[pd.DataFrame] = None

    aligned_results = _prepare_results(df_results) if df_results is not None else None
    if aligned_results is not None and not aligned_results.empty:
        joined = preds.merge(
            aligned_results,
            how="inner",
            on=["home_key", "away_key"],
            suffixes=("", "_res"),
        )
        if "actual_home_win" in joined.columns and "model_home_prob" in joined.columns:
            brier = brier_score(joined["actual_home_win"], joined["model_home_prob"])
            calib_df = calibration_table(joined["actual_home_win"], joined["model_home_prob"])

    if calib_df is not None:
        print("\n[eval] Brier score (home win): {:.5f}".format(brier))
        print("[eval] Calibration buckets (model_home_prob vs actual home wins):")
        print(calib_df.to_string(index=False))
    else:
        print("\n[eval] No results to compute Brier/calibration yet. Skipping accuracy table.")

    n_games = len(preds)
    out_row = pd.DataFrame(
        [
            {
                "date": run_date_str or "",
                "sport": sport,
                "n_games": int(n_games),
                "brier": brier,
                "mean_abs_edge": edge_summary.get("mean_abs_edge", float("nan")),
                "p90_abs_edge": edge_summary.get("p90_abs_edge", float("nan")),
                "max_abs_edge": edge_summary.get("max_abs_edge", float("nan")),
                "over_0.15_rate": edge_summary.get("over_0_15_rate", float("nan")),
                "over_0.25_rate": edge_summary.get("over_0_25_rate", float("nan")),
            }
        ]
    )
    return out_row
