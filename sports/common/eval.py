"""Evaluation helpers for model sanity checks and calibration reporting."""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.odds_sources import (
    SPORT_TO_ODDS_KEY,
    _parse_spread_from_bookmakers,
    _parse_total_from_bookmakers,
)
from sports.common.scores_sources import fetch_scores_history_by_day
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


def _to_iso_date_str(date_in) -> str:
    if date_in is None:
        return ""

    try:
        dt = pd.to_datetime(date_in, errors="coerce")
        if pd.notna(dt):
            return dt.date().isoformat()
    except Exception:
        pass

    try:
        dt = datetime.strptime(str(date_in), "%m/%d/%Y")
        return dt.date().isoformat()
    except Exception:
        return str(date_in)


def build_game_key(event_id, date_str: str, home: str, away: str) -> str:
    """Stable key for matching predictions to scores."""

    if event_id is not None:
        event_id_str = str(event_id).strip()
        if event_id_str:
            return event_id_str

    date_iso = _to_iso_date_str(date_str)
    return f"{date_iso}|{canon_team(home)}|{canon_team(away)}"


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
    ml_hit_rate = float("nan")
    ml_hit_rate_hi_edge = float("nan")
    ml_high_edge_count = 0

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

            ml_mask = pd.Series([True] * len(joined))
            if "primary_market" in joined.columns:
                ml_mask = joined["primary_market"].astype(str).str.upper() == "ML"

            if "primary_side" in joined.columns:
                pick_home = joined["primary_side"].astype(str).str.upper() == "HOME"
            else:
                pick_home = joined["model_home_prob"].astype(float) >= 0.5

            actual_home_win = joined["actual_home_win"].astype(float)
            hit = (pick_home & (actual_home_win == 1.0)) | (~pick_home & (actual_home_win == 0.0))
            if ml_mask.any():
                ml_hit_rate = float(hit[ml_mask].mean())

            edge = None
            for col in ("edge_prob_final", "edge_prob_cal", "edge_prob_raw"):
                if col in joined.columns:
                    edge = pd.to_numeric(joined[col], errors="coerce")
                    break
            if edge is None and "market_home_prob" in joined.columns:
                edge = pd.to_numeric(joined["model_home_prob"], errors="coerce") - pd.to_numeric(
                    joined["market_home_prob"], errors="coerce"
                )
            if edge is not None:
                hi_edge_mask = ml_mask & edge.abs().ge(0.05)
                ml_high_edge_count = int(hi_edge_mask.sum())
                if ml_high_edge_count > 0:
                    ml_hit_rate_hi_edge = float(hit[hi_edge_mask].mean())

    if calib_df is not None:
        print("\n[eval] Brier score (home win): {:.5f}".format(brier))
        print("[eval] Calibration buckets (model_home_prob vs actual home wins):")
        print(calib_df.to_string(index=False))
        if np.isfinite(ml_hit_rate):
            print(f"[eval] ML hit rate: {ml_hit_rate:.3f}")
        if ml_high_edge_count > 0 and np.isfinite(ml_hit_rate_hi_edge):
            print(
                f"[eval] ML hit rate (abs edge >= 0.05): {ml_hit_rate_hi_edge:.3f} "
                f"(n={ml_high_edge_count})"
            )
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


def _load_recent_predictions(
    sport: str,
    *,
    preds_dir: str,
    today: date,
    days_back: int,
) -> pd.DataFrame:
    files: List[Path] = []
    pattern = re.compile(rf"predictions_{sport}_(\d{{2}}-\d{{2}}-\d{{4}})\.csv")
    for path in Path(preds_dir).glob(f"predictions_{sport}_*.csv"):
        match = pattern.search(path.name)
        game_date = None
        if match:
            try:
                game_date = datetime.strptime(match.group(1), "%m-%d-%Y").date()
            except Exception:
                game_date = None
        if game_date is None:
            files.append(path)
            continue
        if (today - game_date).days <= days_back:
            files.append(path)

    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        df = df.copy()
        df["__source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    if "game_key" not in out.columns:
        out["game_key"] = out.apply(
            lambda r: build_game_key(
                r.get("event_id"),
                r.get("date"),
                r.get("home"),
                r.get("away"),
            ),
            axis=1,
        )
    else:
        out["game_key"] = out["game_key"].fillna(
            out.apply(
                lambda r: build_game_key(
                    r.get("event_id"),
                    r.get("date"),
                    r.get("home"),
                    r.get("away"),
                ),
                axis=1,
            )
        )

    return out


def _scores_events_to_df(events: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ev in events or []:
        home = canon_team(ev.get("home_team")) if ev.get("home_team") else None
        away = canon_team(ev.get("away_team")) if ev.get("away_team") else None
        if not home or not away:
            continue

        commence = ev.get("commence_time") or ""
        event_id = ev.get("id")
        game_date_iso = _to_iso_date_str(commence)

        scores = ev.get("scores") or []
        home_score = np.nan
        away_score = np.nan
        for s in scores:
            try:
                name = canon_team(s.get("name"))
                sc = float(s.get("score"))
            except Exception:
                continue
            if name == home:
                home_score = sc
            elif name == away:
                away_score = sc

        if np.isnan(home_score) or np.isnan(away_score):
            continue

        bookmakers = ev.get("bookmakers") or []
        closing_spread, closing_spread_price = _parse_spread_from_bookmakers(bookmakers, ev.get("home_team"))
        closing_total, closing_over_price, closing_under_price = _parse_total_from_bookmakers(bookmakers)

        rows.append(
            {
                "game_key": build_game_key(event_id, game_date_iso, home, away),
                "event_id": event_id or "",
                "home": home,
                "away": away,
                "home_score": home_score,
                "away_score": away_score,
                "score_date": game_date_iso,
                "commence_time": commence,
                "closing_home_spread": float(closing_spread) if closing_spread is not None else float("nan"),
                "closing_spread_price": closing_spread_price if closing_spread_price is not None else np.nan,
                "closing_total_points": float(closing_total) if closing_total is not None else float("nan"),
                "closing_over_price": closing_over_price if closing_over_price is not None else np.nan,
                "closing_under_price": closing_under_price if closing_under_price is not None else np.nan,
            }
        )

    return pd.DataFrame(rows)


def update_eval_history_with_scores(
    *,
    sport: str,
    preds_dir: str,
    out_path: str,
    days_back: int = 30,
) -> pd.DataFrame:
    today = datetime.utcnow().date()
    preds = _load_recent_predictions(sport, preds_dir=preds_dir, today=today, days_back=days_back)
    if preds.empty:
        print("[eval history] No recent predictions found; skipping rolling eval update.")
        return pd.DataFrame()

    sport_key = SPORT_TO_ODDS_KEY.get(sport)
    if not sport_key:
        print(f"[eval history] Unsupported sport for rolling eval: {sport}")
        return pd.DataFrame()

    events = fetch_scores_history_by_day(sport_key, as_of_date=today, days_back=days_back)
    scores_df = _scores_events_to_df(events)
    if scores_df.empty:
        print("[eval history] No completed scores fetched; skipping.")
        return pd.DataFrame()

    preds = preds.drop_duplicates(subset=["game_key"], keep="last")
    merged = preds.merge(scores_df, on="game_key", how="inner", suffixes=("", "_score"))
    if merged.empty:
        print("[eval history] No prediction/score overlaps; skipping.")
        return pd.DataFrame()

    merged["actual_home_win"] = merged.apply(
        lambda r: float(r.get("home_score") > r.get("away_score")) if not (pd.isna(r.get("home_score")) or pd.isna(r.get("away_score"))) else float("nan"),
        axis=1,
    )
    merged["model_home_prob"] = pd.to_numeric(merged.get("model_home_prob"), errors="coerce")

    merged["brier"] = (merged["model_home_prob"] - merged["actual_home_win"]) ** 2

    eps = 1e-6
    merged["log_loss"] = -(
        merged["actual_home_win"] * np.log(np.clip(merged["model_home_prob"], eps, 1 - eps))
        + (1.0 - merged["actual_home_win"]) * np.log(np.clip(1.0 - merged["model_home_prob"], eps, 1 - eps))
    )

    def _ats_outcome(row) -> Tuple[str, str]:
        try:
            model_spread = float(row.get("model_spread_home"))
            closing_spread = float(row.get("closing_home_spread"))
            hs = float(row.get("home_score"))
            ascore = float(row.get("away_score"))
        except Exception:
            return ("", "")

        if any(np.isnan(x) for x in [model_spread, closing_spread, hs, ascore]):
            return ("", "")

        pick = ""
        if model_spread < closing_spread:
            pick = "home"
        elif model_spread > closing_spread:
            pick = "away"

        margin = hs - ascore + closing_spread
        cover = "push"
        if margin > 0:
            cover = "home"
        elif margin < 0:
            cover = "away"

        if pick and cover != "push":
            outcome = "win" if pick == cover else "loss"
        else:
            outcome = "push"
        return (pick, outcome)

    def _totals_outcome(row) -> Tuple[str, str]:
        try:
            model_total = float(row.get("model_total_final", row.get("model_total")))
            closing_total = float(row.get("closing_total_points"))
            hs = float(row.get("home_score"))
            ascore = float(row.get("away_score"))
        except Exception:
            return ("", "")

        if any(np.isnan(x) for x in [model_total, closing_total, hs, ascore]):
            return ("", "")

        pick = ""
        if model_total > closing_total:
            pick = "over"
        elif model_total < closing_total:
            pick = "under"

        total_scored = hs + ascore
        result = "push"
        if total_scored > closing_total:
            result = "over"
        elif total_scored < closing_total:
            result = "under"

        if pick and result != "push":
            outcome = "win" if pick == result else "loss"
        else:
            outcome = "push"
        return (pick, outcome)

    merged[["ats_pick", "ats_outcome"]] = merged.apply(lambda r: pd.Series(_ats_outcome(r)), axis=1)
    merged[["totals_pick", "totals_outcome"]] = merged.apply(lambda r: pd.Series(_totals_outcome(r)), axis=1)

    keep_cols = [
        "game_key",
        "event_id",
        "score_date",
        "home",
        "away",
        "model_home_prob",
        "actual_home_win",
        "brier",
        "log_loss",
        "model_spread_home",
        "closing_home_spread",
        "ats_pick",
        "ats_outcome",
        "model_total",
        "model_total_final",
        "closing_total_points",
        "totals_pick",
        "totals_outcome",
        "home_score",
        "away_score",
        "__source_file",
    ]

    merged = merged[[c for c in keep_cols if c in merged.columns]]

    history = pd.DataFrame()
    try:
        history = pd.read_csv(out_path)
    except Exception:
        history = pd.DataFrame(columns=keep_cols)

    combined = pd.concat([history, merged], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_key"], keep="last")

    combined.to_csv(out_path, index=False)
    print(f"[eval history] Saved {len(combined)} rows -> {out_path}")

    combined_sorted = combined.copy()
    combined_sorted["score_date"] = pd.to_datetime(combined_sorted["score_date"], errors="coerce")
    combined_sorted = combined_sorted.sort_values("score_date")

    last30 = combined_sorted.tail(30)
    last100 = combined_sorted.tail(100)
    brier30 = float(last30["brier"].mean(skipna=True)) if not last30.empty else float("nan")
    brier100 = float(last100["brier"].mean(skipna=True)) if not last100.empty else float("nan")

    ats_games = combined_sorted[combined_sorted["ats_outcome"].isin(["win", "loss"])]
    ats_win_pct = float((ats_games["ats_outcome"] == "win").mean()) if not ats_games.empty else float("nan")

    totals_games = combined_sorted[combined_sorted["totals_outcome"].isin(["win", "loss"])]
    totals_win_pct = float((totals_games["totals_outcome"] == "win").mean()) if not totals_games.empty else float("nan")

    print(
        "last_30_games_brier={:.4f}, last_100_games_brier={:.4f}, ATS win%={:.3f}, totals win%={:.3f}".format(
            brier30, brier100, ats_win_pct, totals_win_pct
        )
    )

    return combined


def _normalize_market_label(market: object) -> str:
    label = str(market or "").upper().strip()
    if "SPREAD" in label or label == "ATS":
        return "ATS"
    if label in {"MONEYLINE", "ML"}:
        return "ML"
    if label in {"TOTAL", "TOTALS"}:
        return "TOTAL"
    return label


def update_eval_history_with_bet_predictions(
    *,
    sport: str,
    preds_dir: str,
    out_path: str,
    days_back: int = 30,
) -> pd.DataFrame:
    today = datetime.utcnow().date()
    preds = _load_recent_predictions(sport, preds_dir=preds_dir, today=today, days_back=days_back)
    if preds.empty:
        print("[eval history] No recent predictions found; skipping bet history update.")
        return pd.DataFrame()

    sport_key = SPORT_TO_ODDS_KEY.get(sport)
    if not sport_key:
        print(f"[eval history] Unsupported sport for rolling eval: {sport}")
        return pd.DataFrame()

    events = fetch_scores_history_by_day(sport_key, as_of_date=today, days_back=days_back)
    scores_df = _scores_events_to_df(events)
    if scores_df.empty:
        print("[eval history] No completed scores fetched; skipping bet history update.")
        return pd.DataFrame()

    market_col = "primary_market" if "primary_market" in preds.columns else "market_type"
    side_col = "primary_side" if "primary_side" in preds.columns else "side"
    preds = preds.copy()
    preds["market_type"] = preds.get(market_col, "").apply(_normalize_market_label)
    preds["side"] = preds.get(side_col, "").astype(str).str.upper().str.strip()

    if "p_model_final" in preds.columns:
        preds["model_prob"] = pd.to_numeric(preds.get("p_model_final"), errors="coerce")
    else:
        preds["model_prob"] = pd.to_numeric(preds.get("model_prob"), errors="coerce")
    preds["model_prob_raw"] = pd.to_numeric(preds.get("p_model_raw"), errors="coerce")
    preds["market_prob"] = pd.to_numeric(preds.get("p_market"), errors="coerce")

    preds["price"] = pd.to_numeric(preds.get("primary_price"), errors="coerce")
    ml_home = pd.to_numeric(preds.get("home_ml"), errors="coerce")
    ml_away = pd.to_numeric(preds.get("away_ml"), errors="coerce")
    preds.loc[(preds["market_type"] == "ML") & (preds["side"] == "HOME") & (~np.isfinite(preds["price"])), "price"] = ml_home
    preds.loc[(preds["market_type"] == "ML") & (preds["side"] == "AWAY") & (~np.isfinite(preds["price"])), "price"] = ml_away
    preds.loc[(preds["market_type"] == "ATS") & (~np.isfinite(preds["price"])), "price"] = pd.to_numeric(
        preds.get("spread_price"), errors="coerce"
    )
    preds.loc[(preds["market_type"] == "TOTAL") & (~np.isfinite(preds["price"])), "price"] = pd.to_numeric(
        preds.get("total_over_price"), errors="coerce"
    )

    preds["line"] = np.nan
    preds.loc[preds["market_type"] == "ATS", "line"] = pd.to_numeric(preds.get("home_spread"), errors="coerce")
    preds.loc[preds["market_type"] == "TOTAL", "line"] = pd.to_numeric(preds.get("total_points"), errors="coerce")

    merged = preds.merge(scores_df, on="game_key", how="inner", suffixes=("", "_score"))
    if merged.empty:
        print("[eval history] No prediction/score overlaps; skipping bet history update.")
        return pd.DataFrame()

    def _actual_outcome(row: pd.Series) -> float:
        market = str(row.get("market_type", "")).upper()
        side = str(row.get("side", "")).upper()
        try:
            hs = float(row.get("home_score"))
            aw = float(row.get("away_score"))
        except Exception:
            return float("nan")
        if np.isnan(hs) or np.isnan(aw):
            return float("nan")
        margin = hs - aw
        if market == "ML":
            actual_home_win = 1.0 if margin > 0 else 0.0
            return actual_home_win if side == "HOME" else (1.0 - actual_home_win)
        if market == "ATS":
            spread = pd.to_numeric(row.get("line"), errors="coerce")
            if not np.isfinite(spread):
                return float("nan")
            cover_margin = margin - float(spread)
            if abs(float(cover_margin)) < 1e-6:
                return float("nan")
            return 1.0 if ((cover_margin > 0 and side == "HOME") or (cover_margin < 0 and side == "AWAY")) else 0.0
        if market == "TOTAL":
            total_line = pd.to_numeric(row.get("line"), errors="coerce")
            if not np.isfinite(total_line):
                return float("nan")
            total_scored = hs + aw
            diff = total_scored - float(total_line)
            if abs(float(diff)) < 1e-6:
                return float("nan")
            return 1.0 if ((diff > 0 and side == "OVER") or (diff < 0 and side == "UNDER")) else 0.0
        return float("nan")

    merged["actual_result"] = merged.apply(_actual_outcome, axis=1)

    keep_cols = [
        "game_key",
        "event_id",
        "sport",
        "date",
        "market_type",
        "side",
        "price",
        "line",
        "model_prob_raw",
        "model_prob",
        "market_prob",
        "actual_result",
        "home",
        "away",
        "score_date",
        "closing_home_spread",
        "closing_spread_price",
        "closing_total_points",
        "closing_over_price",
        "closing_under_price",
        "__source_file",
    ]
    merged = merged[[c for c in keep_cols if c in merged.columns]]

    history = pd.DataFrame()
    try:
        history = pd.read_csv(out_path)
    except Exception:
        history = pd.DataFrame(columns=keep_cols)

    combined = pd.concat([history, merged], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_key", "market_type", "side"], keep="last")

    combined.to_csv(out_path, index=False)
    print(f"[eval history] Saved {len(combined)} bet rows -> {out_path}")
    return combined
