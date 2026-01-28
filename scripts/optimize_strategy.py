#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _american_profit_per_unit(price: float) -> float:
    try:
        odds = float(price)
    except Exception:
        return float("nan")
    if not np.isfinite(odds) or odds == 0:
        return float("nan")
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def _kelly_fraction(p: float, price: float) -> float:
    odds = _american_profit_per_unit(price)
    if not np.isfinite(odds) or odds <= 0:
        return 0.0
    p = max(0.0, min(1.0, float(p)))
    q = 1.0 - p
    frac = (odds * p - q) / odds
    return max(frac, 0.0)


def _kelly_units(p: float, price: float, *, kelly_mult: float, max_pct: float) -> float:
    frac = _kelly_fraction(p, price)
    frac_adj = min(frac * float(kelly_mult), float(max_pct))
    return max(frac_adj / 0.04, 0.0)


def _date_bucket(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.to_period("W").astype(str)


@dataclass
class Candidate:
    min_edge: float
    calibration_risk_multiplier: float
    uncertainty_unit_scale: float
    longshot_cap_units: float
    disagree_cap_units: float
    max_units: float


@dataclass
class Score:
    roi: float
    clv: float
    score: float
    bets: int


def _prep_bets(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    work = df.copy()
    if "sport" in work.columns:
        work = work[work["sport"].astype(str).str.lower() == sport.lower()].copy()
    if work.empty:
        return work

    for col in (
        "edge_prob_final",
        "edge",
        "model_prob",
        "market_prob",
        "price_at_bet",
        "price",
        "result",
    ):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "edge_prob_final" not in work.columns and "edge" in work.columns:
        work["edge_prob_final"] = work["edge"]
    if "edge_prob_final" not in work.columns and {"model_prob", "market_prob"}.issubset(work.columns):
        work["edge_prob_final"] = work["model_prob"] - work["market_prob"]

    if "price_at_bet" not in work.columns and "price" in work.columns:
        work["price_at_bet"] = work["price"]

    work["decision_flags"] = work.get("decision_flags", "").astype(str)
    work["uncertainty"] = pd.to_numeric(work.get("effective_uncertainty", work.get("uncertainty", np.nan)), errors="coerce")
    work["goalie_multiplier"] = pd.to_numeric(work.get("goalie_multiplier", 1.0), errors="coerce")
    work["injury_multiplier"] = pd.to_numeric(work.get("injury_multiplier", 1.0), errors="coerce")

    if "date" in work.columns:
        work["week_bucket"] = _date_bucket(work["date"])
    else:
        work["week_bucket"] = "all"
    return work


def _units_for_row(row: pd.Series, cand: Candidate) -> float:
    edge = row.get("edge_prob_final")
    price = row.get("price_at_bet")
    model_prob = row.get("model_prob", row.get("p_model_final"))
    if not np.isfinite(edge) or edge <= 0:
        return 0.0
    if not np.isfinite(price):
        return 0.0

    base_units = 1.0
    if np.isfinite(model_prob):
        base_units = _kelly_units(model_prob, price, kelly_mult=0.25, max_pct=0.015)

    if edge < cand.min_edge:
        base_units = min(base_units, 0.25)

    flags = str(row.get("decision_flags", ""))
    if "UNCALIBRATED" in flags:
        base_units *= float(cand.calibration_risk_multiplier)

    uncertainty = row.get("uncertainty")
    if np.isfinite(uncertainty):
        mult = 1.0 / (1.0 + float(uncertainty) * float(cand.uncertainty_unit_scale))
        base_units *= max(0.1, min(1.0, mult))

    base_units *= float(row.get("goalie_multiplier", 1.0))
    base_units *= float(row.get("injury_multiplier", 1.0))

    if np.isfinite(price) and price >= 400:
        base_units = min(base_units, float(cand.longshot_cap_units))

    model_prob = row.get("model_prob", row.get("p_model_final"))
    market_prob = row.get("market_prob", row.get("p_market"))
    if np.isfinite(model_prob) and np.isfinite(market_prob):
        disagreement = abs(float(model_prob) - float(market_prob))
        if disagreement > 0.20:
            base_units = min(base_units, float(cand.disagree_cap_units))

    return float(min(base_units, float(cand.max_units)))


def _roi_for_candidate(df: pd.DataFrame, cand: Candidate) -> Tuple[float, int]:
    if df.empty:
        return float("nan"), 0
    profit = 0.0
    count = 0
    for _, row in df.iterrows():
        units = _units_for_row(row, cand)
        if units <= 0:
            continue
        price = row.get("price_at_bet")
        result = row.get("result")
        if not np.isfinite(price) or not np.isfinite(result):
            continue
        profit_per = _american_profit_per_unit(price)
        if not np.isfinite(profit_per):
            continue
        profit += units * (profit_per if result == 1.0 else -1.0)
        count += 1
    if count == 0:
        return float("nan"), 0
    return profit / float(count), count


def _walk_forward_score(df: pd.DataFrame, cand: Candidate, *, clv: float, clv_weight: float) -> Score:
    if df.empty:
        return Score(roi=float("nan"), clv=clv, score=float("nan"), bets=0)

    weeks = [w for w in df["week_bucket"].dropna().unique()]
    weeks = sorted(weeks)
    if len(weeks) <= 1:
        roi, bets = _roi_for_candidate(df, cand)
        score = roi + clv_weight * (clv if np.isfinite(clv) else 0.0)
        return Score(roi=roi, clv=clv, score=score, bets=bets)

    rois: List[float] = []
    total_bets = 0
    for wk in weeks[1:]:
        test = df[df["week_bucket"] == wk]
        roi, bets = _roi_for_candidate(test, cand)
        if np.isfinite(roi):
            rois.append(roi)
            total_bets += bets
    avg_roi = float(np.mean(rois)) if rois else float("nan")
    score = avg_roi + clv_weight * (clv if np.isfinite(clv) else 0.0)
    return Score(roi=avg_roi, clv=clv, score=score, bets=total_bets)


def _avg_clv(clv_log: pd.DataFrame, *, sport: str) -> float:
    if clv_log.empty:
        return float("nan")
    df = clv_log.copy()
    if "sport" in df.columns:
        df = df[df["sport"].astype(str).str.lower() == sport.lower()]
    if df.empty:
        return float("nan")
    open_df = df[df["stage"] == "open"]
    close_df = df[df["stage"] == "close"]
    if open_df.empty or close_df.empty:
        return float("nan")
    merged = open_df.merge(
        close_df[["bet_id", "price"]].rename(columns={"price": "close_price"}),
        on="bet_id",
        how="left",
    )
    merged["open_price"] = merged["price"]
    merged["open_prob"] = merged["open_price"].apply(
        lambda x: 100.0 / (x + 100.0) if x > 0 else (-x) / ((-x) + 100.0)
    )
    merged["close_prob"] = merged["close_price"].apply(
        lambda x: 100.0 / (x + 100.0) if x > 0 else (-x) / ((-x) + 100.0)
    )
    merged["clv_prob"] = merged["close_prob"] - merged["open_prob"]
    return float(pd.to_numeric(merged.get("clv_prob", np.nan), errors="coerce").mean())


def _load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_strategy_config(path: str, sport: str, payload: Dict[str, object]) -> None:
    data: Dict[str, object] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict):
            data = existing
    except Exception:
        data = {}

    sports = data.get("sports")
    if not isinstance(sports, dict):
        sports = {}
    sports[str(sport).lower()] = payload
    data["sports"] = sports
    data["updated_at"] = datetime.utcnow().isoformat() + "Z"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="nba")
    parser.add_argument("--bet-log", default="results/bet_log.csv")
    parser.add_argument("--clv-log", default="results/clv_log.csv")
    parser.add_argument("--out", default="results/strategy_config.json")
    parser.add_argument("--clv-weight", type=float, default=0.5)
    args = parser.parse_args()

    bet_log = _load_csv(args.bet_log)
    if bet_log.empty:
        print(f"[optimize] missing or empty bet log: {args.bet_log}")
        return 1

    clv_log = _load_csv(args.clv_log)
    clv_avg = _avg_clv(clv_log, sport=args.sport)

    prepared = _prep_bets(bet_log, args.sport)
    if prepared.empty:
        print("[optimize] no rows for sport.")
        return 1

    candidates: List[Tuple[Candidate, Score]] = []
    for min_edge in (0.01, 0.02, 0.03, 0.04, 0.05):
        for cal_mult in (0.4, 0.55, 0.7, 0.85):
            for unc_scale in (4.0, 6.0, 8.0):
                for longshot_cap in (0.2, 0.25, 0.35):
                    for disagree_cap in (0.2, 0.25, 0.35):
                        for max_units in (0.5, 1.0):
                            cand = Candidate(
                                min_edge=min_edge,
                                calibration_risk_multiplier=cal_mult,
                                uncertainty_unit_scale=unc_scale,
                                longshot_cap_units=longshot_cap,
                                disagree_cap_units=disagree_cap,
                                max_units=max_units,
                            )
                            score = _walk_forward_score(
                                prepared, cand, clv=clv_avg, clv_weight=args.clv_weight
                            )
                            if np.isfinite(score.score):
                                candidates.append((cand, score))

    if not candidates:
        print("[optimize] no valid candidates found.")
        return 1

    best_cand, best_score = max(candidates, key=lambda item: item[1].score)
    payload: Dict[str, object] = {
        "min_edge_cal": best_cand.min_edge,
        "calibration_risk_multiplier": best_cand.calibration_risk_multiplier,
        "uncertainty_unit_scale": best_cand.uncertainty_unit_scale,
        "longshot_cap_units": best_cand.longshot_cap_units,
        "disagree_cap_units": best_cand.disagree_cap_units,
        "max_units": best_cand.max_units,
        "objective": {
            "roi": best_score.roi,
            "clv_weight": args.clv_weight,
            "clv_avg": best_score.clv,
            "score": best_score.score,
            "bets": best_score.bets,
        },
        "notes": "Walk-forward ROI + CLV optimization using bet_log.",
    }

    _write_strategy_config(args.out, args.sport, payload)

    print(f"[optimize] wrote {args.out}")
    print(
        "[optimize] best "
        f"min_edge={best_cand.min_edge:.3f} cal_mult={best_cand.calibration_risk_multiplier:.2f} "
        f"unc_scale={best_cand.uncertainty_unit_scale:.1f} longshot_cap={best_cand.longshot_cap_units:.2f} "
        f"disagree_cap={best_cand.disagree_cap_units:.2f} max_units={best_cand.max_units:.2f} "
        f"roi={best_score.roi:.4f} score={best_score.score:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
