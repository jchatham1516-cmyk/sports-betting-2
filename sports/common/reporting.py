from __future__ import annotations

import json
import os
import math
import importlib.util
from typing import Dict, Optional, Iterable, List

import numpy as np
import pandas as pd

from sports.common.teams import canon_team
from sports.common.util import american_to_decimal, normalize_result_label
from sports.common.bet_config import get_sport_bet_config
from sports.common.prob_uncertainty import load_uncertainty


ODDS_BUCKETS = [
    (100, 150, "+100..+150"),
    (151, 300, "+151..+300"),
    (301, 500, "+301..+500"),
    (501, 10_000, "+500+"),
]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _profit_from_row(price: float, result: str, stake: float) -> float:
    result = normalize_result_label(result)
    if not np.isfinite(stake) or stake <= 0:
        return float("nan")
    if result == "WIN":
        dec = american_to_decimal(price)
        return stake * (dec - 1.0)
    if result == "LOSS":
        return -stake
    if result == "PUSH":
        return 0.0
    return float("nan")


def _bucket_for_odds(price: float) -> str:
    if not np.isfinite(price) or price < 100:
        return "OTHER"
    for low, high, label in ODDS_BUCKETS:
        if low <= price <= high:
            return label
    if price > 500:
        return "+500+"
    return "OTHER"


def _grade_moneyline_from_history(
    bets: pd.DataFrame, history_df: pd.DataFrame
) -> pd.DataFrame:
    if bets.empty or history_df.empty:
        return bets

    hist = history_df.copy()
    hist["home_canon"] = hist["home"].apply(canon_team)
    hist["away_canon"] = hist["away"].apply(canon_team)
    hist = hist[["date", "home_canon", "away_canon", "home_win"]]

    bets = bets.copy()
    bets["home_canon"] = bets["home"].apply(canon_team)
    bets["away_canon"] = bets["away"].apply(canon_team)

    merged = bets.merge(hist, on=["date", "home_canon", "away_canon"], how="left")
    missing = merged["result"].isna() | (merged["result"].astype(str).str.strip() == "")
    if not missing.any():
        return bets

    def _infer_result(row: pd.Series) -> Optional[str]:
        if pd.isna(row.get("home_win")):
            return None
        side = str(row.get("side", "")).upper()
        if side == "HOME":
            return "WIN" if int(row.get("home_win")) == 1 else "LOSS"
        if side == "AWAY":
            return "WIN" if int(row.get("home_win")) == 0 else "LOSS"
        return None

    merged.loc[missing, "result"] = merged[missing].apply(_infer_result, axis=1)
    merged = merged.drop(columns=["home_canon", "away_canon", "home_win"])
    return merged


def generate_backtest_report(
    sport: str,
    history_csv_path: str,
    *,
    bet_log_path: str = "results/tracking/bet_log.csv",
) -> Optional[Dict[str, object]]:
    if not os.path.exists(bet_log_path):
        print(f"[report] No bet log at {bet_log_path}; skipping report.")
        return None

    bets = pd.read_csv(bet_log_path)
    if bets.empty:
        print("[report] Bet log empty; skipping report.")
        return None

    bets = bets[bets["sport"].astype(str).str.lower() == str(sport).lower()].copy()
    if bets.empty:
        print("[report] No bets for sport; skipping report.")
        return None

    history_df = pd.DataFrame()
    if history_csv_path and os.path.exists(history_csv_path):
        history_df = pd.read_csv(history_csv_path)

    if not history_df.empty:
        bets = _grade_moneyline_from_history(bets, history_df)

    bets["result"] = bets["result"].apply(normalize_result_label)
    bets = bets[bets["result"].isin(["WIN", "LOSS", "PUSH"])]
    if bets.empty:
        print("[report] No graded bets; skipping report.")
        return None

    bets["price"] = bets["price_at_bet"].apply(_safe_float)
    bets["units"] = bets["units"].apply(_safe_float)
    bets["unit_dollars"] = bets["unit_dollars"].apply(_safe_float)
    bets["stake"] = bets["units"] * bets["unit_dollars"]
    bets["profit"] = bets.apply(
        lambda r: _profit_from_row(r["price"], r["result"], r["stake"]),
        axis=1,
    )

    def _summary(df: pd.DataFrame) -> Dict[str, float]:
        stake = df["stake"].sum()
        profit = df["profit"].sum()
        wins = int((df["result"] == "WIN").sum())
        losses = int((df["result"] == "LOSS").sum())
        pushes = int((df["result"] == "PUSH").sum())
        bets_count = wins + losses + pushes
        win_pct = wins / (wins + losses) if wins + losses > 0 else 0.0
        avg_odds = df["price"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        roi = profit / stake if stake > 0 else 0.0
        return {
            "bets": bets_count,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_pct": win_pct,
            "roi": roi,
            "avg_odds": avg_odds,
        }

    bets["odds_bucket"] = bets["price"].apply(_bucket_for_odds)

    report = {
        "sport": str(sport).lower(),
        "overall": _summary(bets),
        "by_odds_bucket": {
            bucket: _summary(group)
            for bucket, group in bets.groupby("odds_bucket")
        },
        "by_confidence": {
            str(conf): _summary(group)
            for conf, group in bets.groupby(bets["confidence"].fillna("UNKNOWN"))
        },
        "by_value_tier": {
            str(tier): _summary(group)
            for tier, group in bets.groupby(bets["value_tier"].fillna("UNKNOWN"))
        },
    }

    os.makedirs(os.path.join("results", "reports"), exist_ok=True)
    out_path = os.path.join("results", "reports", f"{sport}_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def self_check_recent_bets(
    *,
    bet_log_path: str = "results/tracking/bet_log.csv",
    last_n: int = 200,
) -> Optional[Dict[str, object]]:
    if not os.path.exists(bet_log_path):
        print(f"[self-check] No bet log at {bet_log_path}; skipping.")
        return None

    bets = pd.read_csv(bet_log_path)
    if bets.empty:
        print("[self-check] Bet log empty; skipping.")
        return None

    bets = bets.tail(int(last_n)).copy()
    bets["result"] = bets["result"].apply(normalize_result_label)
    bets = bets[bets["result"].isin(["WIN", "LOSS", "PUSH"])]
    if bets.empty:
        print("[self-check] No graded bets in recent window; skipping.")
        return None

    bets["units"] = bets["units"].apply(_safe_float)
    bets["unit_dollars"] = bets["unit_dollars"].apply(_safe_float)
    bets["stake"] = bets["units"] * bets["unit_dollars"]
    bets["price"] = bets["price_at_bet"].apply(_safe_float)

    def _result_num(res: str) -> float:
        if res == "WIN":
            return 1.0
        if res == "LOSS":
            return 0.0
        return 0.5

    bets["result_num"] = bets["result"].apply(_result_num)
    bets["model_prob"] = bets["model_prob"].apply(_safe_float)
    bets["market_prob"] = bets["market_prob"].apply(_safe_float)
    bets["edge_prob_final"] = bets["edge_prob_final"].apply(_safe_float)
    bets["abs_edge_prob"] = bets["abs_edge_prob"].apply(_safe_float)

    bets["implied_edge"] = bets["model_prob"] - bets["market_prob"]

    overall_units = float(bets["units"].sum())
    wins = int((bets["result"] == "WIN").sum())
    losses = int((bets["result"] == "LOSS").sum())
    pushes = int((bets["result"] == "PUSH").sum())
    win_rate = wins / (wins + losses) if wins + losses > 0 else 0.0
    avg_implied_edge = float(bets["implied_edge"].replace([np.inf, -np.inf], np.nan).mean())
    avg_model_edge = float(bets["edge_prob_final"].replace([np.inf, -np.inf], np.nan).mean())

    brier = float(
        np.nanmean((bets["model_prob"] - bets["result_num"]) ** 2)
        if bets["model_prob"].notna().any()
        else float("nan")
    )

    by_market = {}
    for market, group in bets.groupby(bets["market_type"].fillna("UNKNOWN")):
        wins_m = int((group["result"] == "WIN").sum())
        losses_m = int((group["result"] == "LOSS").sum())
        pushes_m = int((group["result"] == "PUSH").sum())
        win_rate_m = wins_m / (wins_m + losses_m) if wins_m + losses_m > 0 else 0.0
        by_market[str(market)] = {
            "bets": int(len(group)),
            "wins": wins_m,
            "losses": losses_m,
            "pushes": pushes_m,
            "win_rate": float(win_rate_m),
            "avg_implied_edge": float(group["implied_edge"].replace([np.inf, -np.inf], np.nan).mean()),
            "avg_model_edge": float(group["edge_prob_final"].replace([np.inf, -np.inf], np.nan).mean()),
        }

    def _thin_edge_flag(row: pd.Series) -> bool:
        sport = str(row.get("sport", "")).lower()
        config = get_sport_bet_config(sport)
        edge = row.get("abs_edge_prob")
        return bool(np.isfinite(edge) and float(edge) < float(config.min_edge_cal))

    thin_edge_rate = float(bets.apply(_thin_edge_flag, axis=1).mean())

    report = {
        "sample_size": int(len(bets)),
        "overall_units": overall_units,
        "win_rate": win_rate,
        "avg_implied_edge": avg_implied_edge,
        "avg_model_edge": avg_model_edge,
        "brier": brier,
        "by_market": by_market,
        "thin_edge_rate": thin_edge_rate,
    }

    print(
        "[self-check] "
        f"n={report['sample_size']} units={overall_units:.2f} win_rate={win_rate:.3f} "
        f"avg_implied_edge={avg_implied_edge:.4f} avg_model_edge={avg_model_edge:.4f} brier={brier:.4f}"
    )
    for market, stats in by_market.items():
        print(
            "[self-check] "
            f"{market}: bets={stats['bets']} win_rate={stats['win_rate']:.3f} "
            f"avg_implied_edge={stats['avg_implied_edge']:.4f} avg_model_edge={stats['avg_model_edge']:.4f}"
        )
    if thin_edge_rate > 0.4:
        print(f"[self-check] WARNING: {thin_edge_rate:.0%} of recent bets are thin-edge (<min_edge_cal).")

    return report


def daily_bet_report(
    *,
    bet_log_path: str = "results/tracking/bet_log.csv",
    thin_edge_warn_pct: float = 0.4,
    bucket_size: float = 0.05,
) -> Optional[Dict[str, object]]:
    if not os.path.exists(bet_log_path):
        print(f"[daily-report] No bet log at {bet_log_path}; skipping.")
        return None

    bets = pd.read_csv(bet_log_path)
    if bets.empty:
        print("[daily-report] Bet log empty; skipping.")
        return None

    bets = bets.copy()
    bets["result"] = bets["result"].apply(normalize_result_label)
    bets = bets[bets["result"].isin(["WIN", "LOSS", "PUSH"])]
    if bets.empty:
        print("[daily-report] No graded bets; skipping.")
        return None

    bets["units"] = bets["units"].apply(_safe_float)
    bets["unit_dollars"] = bets["unit_dollars"].apply(_safe_float)
    bets["stake"] = bets["units"] * bets["unit_dollars"]
    bets["price"] = bets["price_at_bet"].apply(_safe_float)
    bets["profit"] = bets.apply(
        lambda r: _profit_from_row(r["price"], r["result"], r["stake"]),
        axis=1,
    )

    def _summary(df: pd.DataFrame) -> Dict[str, float]:
        stake = df["stake"].sum()
        profit = df["profit"].sum()
        wins = int((df["result"] == "WIN").sum())
        losses = int((df["result"] == "LOSS").sum())
        pushes = int((df["result"] == "PUSH").sum())
        bets_count = wins + losses + pushes
        win_pct = wins / (wins + losses) if wins + losses > 0 else 0.0
        roi = profit / stake if stake > 0 else 0.0
        return {
            "bets": bets_count,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_pct,
            "roi": roi,
            "units": float(df["units"].sum()),
        }

    by_sport = {
        str(sport): _summary(group)
        for sport, group in bets.groupby(bets["sport"].fillna("UNKNOWN"))
    }
    by_market = {
        str(market): _summary(group)
        for market, group in bets.groupby(bets["market_type"].fillna("UNKNOWN"))
    }

    prob_col = "model_prob" if "model_prob" in bets.columns else "p_model_final"
    bets["prob_used"] = bets.get(prob_col, np.nan).apply(_safe_float)
    bets = bets[bets["prob_used"].notna()]
    bets["prob_used"] = bets["prob_used"].clip(0.0, 1.0)

    def _bucket_label(p: float) -> str:
        lo = math.floor(float(p) / bucket_size) * bucket_size
        hi = lo + bucket_size
        return f"{lo:.2f}-{hi:.2f}"

    bets["prob_bucket"] = bets["prob_used"].apply(_bucket_label)
    calib = {}
    for bucket, group in bets.groupby("prob_bucket"):
        wins = int((group["result"] == "WIN").sum())
        losses = int((group["result"] == "LOSS").sum())
        win_rate = wins / (wins + losses) if wins + losses > 0 else 0.0
        calib[str(bucket)] = {
            "bets": int(len(group)),
            "avg_prob": float(group["prob_used"].mean()),
            "win_rate": float(win_rate),
        }

    def _thin_edge_flag(row: pd.Series) -> bool:
        sport = str(row.get("sport", "")).lower()
        config = get_sport_bet_config(sport)
        edge = row.get("abs_edge_prob")
        if not np.isfinite(edge):
            return False
        return float(edge) < float(config.min_edge_cal) + 0.01

    thin_edge_rate = float(bets.apply(_thin_edge_flag, axis=1).mean())

    report = {
        "by_sport": by_sport,
        "by_market": by_market,
        "calibration": calib,
        "thin_edge_rate": thin_edge_rate,
    }

    print("[daily-report] Summary by sport:")
    for sport, stats in by_sport.items():
        print(
            f"  {sport}: units={stats['units']:.2f} win_rate={stats['win_rate']:.3f} ROI={stats['roi']:.3f}"
        )
    print("[daily-report] Summary by market:")
    for market, stats in by_market.items():
        print(
            f"  {market}: units={stats['units']:.2f} win_rate={stats['win_rate']:.3f} ROI={stats['roi']:.3f}"
        )
    print("[daily-report] Calibration buckets:")
    for bucket, stats in calib.items():
        print(
            f"  {bucket}: bets={stats['bets']} avg_prob={stats['avg_prob']:.3f} win_rate={stats['win_rate']:.3f}"
        )
    if thin_edge_rate > float(thin_edge_warn_pct):
        print(
            f"[daily-report] WARNING: {thin_edge_rate:.0%} of bets are thin edges "
            f"(<min_edge+0.01)."
        )

    return report


def _get_tabulate():
    if importlib.util.find_spec("tabulate") is None:
        return None
    import tabulate

    return tabulate.tabulate


def _fmt_float(value: object, decimals: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{decimals}f}"


def _fmt_line(value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    if float(val).is_integer():
        return f"{int(val)}"
    return f"{val:.1f}"


def _fmt_price(value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:+.0f}"


def _confidence_score(value: object) -> float:
    tier = str(value or "").upper().strip()
    if "HIGH" in tier:
        return 3.0
    if "MED" in tier:
        return 2.0
    if "LOW" in tier:
        return 1.0
    return float("nan")


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _prob_for_total(row: pd.Series, side: str) -> float:
    p_over = np.nan
    total_sd = _safe_float(row.get("total_sd"))
    model_total = _safe_float(row.get("model_total_final", row.get("model_total")))
    total_points = _safe_float(row.get("total_points"))

    if np.isfinite(total_sd) and total_sd > 1e-6 and np.isfinite(model_total) and np.isfinite(total_points):
        z = (total_points - model_total) / total_sd
        p_over = 1.0 - _norm_cdf(z)
    else:
        pick_side = str(row.get("total_pick_side", "")).upper().strip()
        pick_prob = _safe_float(row.get("total_pick_prob"))
        if np.isfinite(pick_prob):
            if pick_side == "OVER":
                p_over = pick_prob
            elif pick_side == "UNDER":
                p_over = 1.0 - pick_prob

    if not np.isfinite(p_over):
        return float("nan")

    if side == "OVER":
        return float(p_over)
    if side == "UNDER":
        return float(1.0 - p_over)
    return float("nan")


def build_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    def _display_market(row: pd.Series) -> str:
        market = str(row.get("primary_market", row.get("market_type", ""))).upper().strip()
        if "SPREAD" in market:
            market = "ATS"
        if market in {"MONEYLINE", "ML"}:
            return "ML"
        if market in {"ATS", "SPREAD"}:
            return "ATS"
        if market in {"TOTAL", "TOTALS"}:
            return "TOTAL"
        return market

    def _display_pick(row: pd.Series, market: str) -> str:
        side = str(row.get("primary_side", "")).upper().strip()
        if market == "ML":
            price = row.get("primary_price")
            if not np.isfinite(_safe_float(price)):
                if side == "HOME":
                    price = row.get("home_ml")
                elif side == "AWAY":
                    price = row.get("away_ml")
            price_str = _fmt_price(price)
            if price_str:
                return f"{side} {price_str}"
            return side
        if market == "ATS":
            line = row.get("home_spread")
            line_val = _safe_float(line)
            if np.isfinite(line_val) and side == "AWAY":
                line_val = -line_val
            line_str = _fmt_line(line_val)
            price_str = _fmt_price(row.get("spread_price"))
            if price_str:
                return f"{side} {line_str} ({price_str})".strip()
            if line_str:
                return f"{side} {line_str}".strip()
            return side
        if market == "TOTAL":
            line_str = _fmt_line(row.get("total_points"))
            price = row.get("total_over_price") if side == "OVER" else row.get("total_under_price")
            price_str = _fmt_price(price)
            if price_str:
                return f"{side} {line_str} ({price_str})".strip()
            if line_str:
                return f"{side} {line_str}".strip()
            return side
        return side

    def _p_win_display(row: pd.Series, market: str) -> float:
        side = str(row.get("primary_side", row.get("total_pick_side", ""))).upper().strip()
        if market == "ML":
            p_model_final = _safe_float(row.get("p_model_final"))
            if np.isfinite(p_model_final):
                return float(p_model_final)
            p_model = _safe_float(row.get("model_prob"))
            if np.isfinite(p_model):
                return float(p_model)
            p_home = _safe_float(row.get("model_home_prob_final", row.get("model_home_prob")))
            if np.isfinite(p_home):
                if side == "AWAY":
                    return float(1.0 - p_home)
                return float(p_home)
            return float("nan")
        if market == "ATS":
            p_cover = _safe_float(row.get("p_cover_final"))
            if not np.isfinite(p_cover):
                p_cover = _safe_float(row.get("p_home_cover", row.get("ats_home_cover_prob")))
            if not np.isfinite(p_cover):
                return float("nan")
            if side == "AWAY":
                return float(1.0 - p_cover)
            return float(p_cover)
        if market == "TOTAL":
            if side not in {"OVER", "UNDER"}:
                return float("nan")
            return _prob_for_total(row, side)
        return float("nan")

    def _edge_display(row: pd.Series) -> float:
        edge = _safe_float(row.get("edge_prob_final"))
        if np.isfinite(edge):
            return float(edge)
        edge = _safe_float(row.get("edge"))
        if np.isfinite(edge):
            return float(edge)
        return float("nan")

    def _reason_short(row: pd.Series) -> str:
        reason = str(row.get("decision_reason", "") or "").replace("\n", " ").strip()
        if len(reason) > 80:
            return reason[:77] + "..."
        return reason

    out["display_market"] = out.apply(_display_market, axis=1)
    out["display_pick"] = out.apply(lambda r: _display_pick(r, r["display_market"]), axis=1)
    out["p_win_display"] = out.apply(lambda r: _p_win_display(r, r["display_market"]), axis=1)
    out["edge_display"] = out.apply(_edge_display, axis=1)
    out["reason_short"] = out.apply(_reason_short, axis=1)

    return out


def build_rankings(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    if "p_win_display" not in out.columns or "edge_display" not in out.columns:
        out = build_display_columns(out)

    out["_confidence_score"] = out["confidence"].apply(_confidence_score) if "confidence" in out.columns else np.nan
    out["_p_win_display"] = out["p_win_display"].apply(_safe_float)
    out["_edge_display"] = out["edge_display"].apply(_safe_float)
    out["_primary_ev"] = out["primary_ev"].apply(_safe_float) if "primary_ev" in out.columns else np.nan
    out["_abs_edge_prob"] = out["abs_edge_prob"].apply(_safe_float) if "abs_edge_prob" in out.columns else np.nan

    accuracy_sorted = out.sort_values(
        by=["_p_win_display", "_confidence_score", "_edge_display"],
        ascending=[False, False, False],
        kind="mergesort",
        na_position="last",
    )
    value_sorted = out.sort_values(
        by=["_primary_ev", "_abs_edge_prob", "_confidence_score"],
        ascending=[False, False, False],
        kind="mergesort",
        na_position="last",
    )

    out["rank_accuracy"] = pd.Series(
        range(1, len(accuracy_sorted) + 1), index=accuracy_sorted.index
    ).reindex(out.index)
    out["rank_value"] = pd.Series(
        range(1, len(value_sorted) + 1), index=value_sorted.index
    ).reindex(out.index)

    out = out.drop(columns=["_confidence_score", "_p_win_display", "_edge_display", "_primary_ev", "_abs_edge_prob"])
    return out


def _format_table(rows: List[List[object]], headers: Iterable[str]) -> str:
    tabulate = _get_tabulate()
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="github")

    headers_list = list(headers)
    str_rows: List[List[str]] = []
    for row in rows:
        str_rows.append([
            _fmt_float(cell) if isinstance(cell, float) else ("" if cell is None else str(cell))
            for cell in row
        ])

    col_widths = [len(str(h)) for h in headers_list]
    for row in str_rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def _fmt_row(row_vals: List[str]) -> str:
        return " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row_vals))

    lines = [_fmt_row([str(h) for h in headers_list])]
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in str_rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def render_console_report(
    df: pd.DataFrame,
    sport: str,
    date: str,
    *,
    debug: bool = False,
    sort_by: str = "value",
) -> None:
    if df is None or df.empty:
        print(f"[report] {str(sport).upper()} {date} -> no rows")
        return

    report_df = build_rankings(build_display_columns(df))

    play_pass_series = (
        report_df["play_pass"].astype(str)
        if "play_pass" in report_df.columns
        else pd.Series([""] * len(report_df), index=report_df.index)
    )
    plays_mask = play_pass_series == "PLAY"
    plays_count = int(plays_mask.sum())
    pass_count = int((play_pass_series == "PASS").sum())
    total_games = int(len(report_df))

    sort_label = str(sort_by or "value").lower()
    sort_label = "accuracy" if sort_label == "accuracy" else "value"

    print("\n=== Summary ===")
    print(
        f"{str(sport).upper()} | {date} | games={total_games} plays={plays_count} passes={pass_count} "
        f"| sort={sort_label}"
    )

    top_plays = report_df[plays_mask].sort_values(
        by=["rank_value", "rank_accuracy"],
        ascending=[True, True],
        kind="mergesort",
    ).head(10)

    print("\n=== Top Plays ===")
    top_columns = [
        "rank_value",
        "display_market",
        "display_pick",
        "p_win_display",
        "edge_display",
        "primary_ev",
        "confidence",
        "value_tier",
        "reason_short",
    ]
    top_rows = top_plays.reindex(columns=[c for c in top_columns if c in top_plays.columns]).values.tolist()
    top_headers = [c for c in top_columns if c in top_plays.columns]
    print(_format_table(top_rows, top_headers) if top_rows else "No PLAY rows.")

    accuracy_df = report_df.copy()
    accuracy_df = accuracy_df[np.isfinite(accuracy_df["p_win_display"].astype(float))]
    accuracy_df = accuracy_df.sort_values(
        by=["rank_accuracy"], ascending=[True], kind="mergesort"
    ).head(15)

    print("\n=== Accuracy Ranking ===")
    accuracy_columns = [
        "rank_accuracy",
        "display_market",
        "display_pick",
        "p_win_display",
        "edge_display",
        "confidence",
        "play_pass",
    ]
    accuracy_rows = accuracy_df.reindex(columns=[c for c in accuracy_columns if c in accuracy_df.columns]).values.tolist()
    accuracy_headers = [c for c in accuracy_columns if c in accuracy_df.columns]
    print(_format_table(accuracy_rows, accuracy_headers) if accuracy_rows else "No rows with p_win_display.")

    value_df = report_df.sort_values(by=["rank_value"], ascending=[True], kind="mergesort").head(15)
    print("\n=== Value Ranking ===")
    value_columns = [
        "rank_value",
        "display_market",
        "display_pick",
        "primary_ev",
        "edge_display",
        "confidence",
        "play_pass",
    ]
    value_rows = value_df.reindex(columns=[c for c in value_columns if c in value_df.columns]).values.tolist()
    value_headers = [c for c in value_columns if c in value_df.columns]
    print(_format_table(value_rows, value_headers) if value_rows else "No rows.")

    if debug:
        debug_df = report_df.copy()
        uncertainty = load_uncertainty(sport)
        if uncertainty and np.isfinite(_safe_float(uncertainty.get("uncertainty"))):
            debug_df["prob_uncertainty"] = float(uncertainty.get("uncertainty"))

        debug_columns = [
            "display_market",
            "display_pick",
            "decision_flags",
            "ats_cal_used",
            "margin_calibrated",
            "model_home_prob_cal",
            "model_home_prob_final",
            "injury_confidence",
            "inj_points_home",
            "inj_points_away",
            "inj_total_adj",
            "prob_uncertainty",
        ]
        debug_columns = [c for c in debug_columns if c in debug_df.columns]
        debug_rows = debug_df.reindex(columns=debug_columns).head(15).values.tolist()
        print("\n=== Debug Columns ===")
        print(_format_table(debug_rows, debug_columns) if debug_rows else "No debug columns available.")
