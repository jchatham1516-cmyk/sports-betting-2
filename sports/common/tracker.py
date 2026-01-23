from __future__ import annotations

import glob
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.odds_sources import SPORT_TO_ODDS_KEY
from sports.common.scores_sources import fetch_scores_history_by_day
from sports.common.teams import canon_team
from sports.common.util import american_to_decimal, normalize_result_label, safe_float


# ---------------------------
# Helpers
# ---------------------------

DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y")

BET_LOG_COLUMNS = [
    "bet_id",
    "date",
    "sport",
    "home",
    "away",
    "market_type",
    "side",
    "line_at_bet",
    "price_at_bet",
    "model_prob",
    "market_prob",
    "edge",
    "confidence",
    "value_tier",
    "p_model_used",
    "p_market_used",
    "abs_edge_prob",
    "p_model_raw",
    "p_model_cal",
    "p_model_final",
    "p_market",
    "edge_prob_raw",
    "edge_prob_cal",
    "edge_prob_final",
    "units",
    "result",
    "unit_dollars",
    "stake_dollars",
    "profit_dollars",
    "payout_dollars",
    "closing_line",
    "closing_price",
    "clv",
    "notes",
]


def _parse_date(value) -> Optional[date]:
    if value is None:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(str(value), fmt).date()
        except Exception:
            continue
    return None


def parse_tracking_date(value) -> Optional[date]:
    """Public wrapper for date parsing used by CLI layers."""

    return _parse_date(value)


def _normalize_date_str(value) -> str:
    parsed = _parse_date(value)
    if parsed:
        return parsed.isoformat()
    return str(value)


def _find_recs_csv(results_dir: str, target_date: date) -> Optional[str]:
    """Find the recommendations file for a target date.

    Preference order:
      1) Filename contains the date string (MM-DD-YYYY or YYYY-MM-DD)
      2) Fallback to the most recently modified CSV in the directory
    """

    iso = target_date.isoformat()
    mdY = target_date.strftime("%m-%d-%Y")

    patterns = [
        os.path.join(results_dir, f"*{mdY}*.csv"),
        os.path.join(results_dir, f"*{iso}*.csv"),
        os.path.join(results_dir, "*.csv"),
    ]

    candidates: List[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    candidates = [c for c in candidates if "tracking" not in c.lower()]
    if not candidates:
        return None

    def _score(path: str) -> Tuple[int, int]:
        name = os.path.basename(path)
        contains_date = int(mdY in name or iso in name)
        mtime = int(os.path.getmtime(path))
        return (contains_date, mtime)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _american_to_decimal(american: float) -> float:
    try:
        return american_to_decimal(float(american))
    except Exception:
        return float("nan")


def _calc_profit_and_payout(
    units: float,
    unit_dollars: float,
    odds: float,
    result: str,
) -> Tuple[float, float, float]:
    """Returns (stake, profit, payout). Profit excludes stake return."""

    units_val = _safe_number(units)
    unit_dollars_val = _safe_number(unit_dollars)
    if math.isnan(units_val):
        units_val = 0.0
    if math.isnan(unit_dollars_val):
        unit_dollars_val = 0.0

    stake_dollars = units_val * unit_dollars_val

    odds_val = _safe_number(odds)
    if math.isnan(odds_val):
        odds_val = 0.0

    result = normalize_result_label(result)

    if result == "WIN":
        if odds_val > 0:
            decimal = 1.0 + (odds_val / 100.0)
        elif odds_val < 0:
            decimal = 1.0 + (100.0 / abs(odds_val))
        else:
            decimal = 1.0
        profit = stake_dollars * (decimal - 1.0)
        payout = stake_dollars + profit
        return stake_dollars, profit, payout
    if result == "LOSS":
        return stake_dollars, -stake_dollars, 0.0
    if result == "PUSH":
        return stake_dollars, 0.0, stake_dollars
    return stake_dollars, 0.0, 0.0


def _safe_number(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _hash_bet_id(*parts: object) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


# ---------------------------
# Grading logic
# ---------------------------

def grade_moneyline(side: str, home_score: int, away_score: int) -> str:
    if home_score is None or away_score is None or np.isnan(home_score) or np.isnan(away_score):
        return "MISSING_SCORE"
    if home_score == away_score:
        return "PUSH"
    winner = "HOME" if home_score > away_score else "AWAY"
    return "WIN" if str(side).upper() == winner else "LOSS"


def grade_spread(side: str, spread_home: float, home_score: int, away_score: int) -> str:
    if home_score is None or away_score is None or np.isnan(home_score) or np.isnan(away_score):
        return "MISSING_SCORE"
    spread_home = float(spread_home)

    if str(side).upper() == "HOME":
        lhs = home_score + spread_home
        rhs = away_score
    else:
        lhs = away_score - spread_home
        rhs = home_score

    if abs(lhs - rhs) < 1e-9:
        return "PUSH"
    return "WIN" if lhs > rhs else "LOSS"


def grade_total(side: str, total_line: float, home_score: int, away_score: int) -> str:
    if home_score is None or away_score is None or np.isnan(home_score) or np.isnan(away_score):
        return "MISSING_SCORE"
    total = home_score + away_score
    line = float(total_line)
    if abs(total - line) < 1e-9:
        return "PUSH"
    if str(side).upper() == "OVER":
        return "WIN" if total > line else "LOSS"
    return "WIN" if total < line else "LOSS"


def _canon_names(df: pd.DataFrame, home_col: str = "home", away_col: str = "away") -> pd.DataFrame:
    out = df.copy()
    out["home_canon"] = out[home_col].apply(canon_team)
    out["away_canon"] = out[away_col].apply(canon_team)
    return out


def _is_play_row(row: pd.Series) -> bool:
    play = str(row.get("play_pass", "PLAY")).upper()
    return play == "PLAY"


def _is_playable_reco(text: str) -> bool:
    if text is None:
        return False
    u = str(text).upper().strip()
    if u == "":
        return False
    if u.startswith("NO ") or "NO BET" in u:
        return False
    if u.startswith("PASS"):
        return False
    return True


def _bet_edge_for_market(row: pd.Series, market_type: str) -> object:
    market = str(market_type).lower()
    if market == "moneyline":
        return row.get("primary_ev", row.get("ml_ev_best"))
    if market == "spread":
        return row.get("primary_ev", row.get("ats_ev_best"))
    if market == "total":
        return row.get("primary_ev", row.get("total_ev_best"))
    return None


def _build_bet_from_primary(row: pd.Series, sport: str) -> Optional[Dict[str, object]]:
    play = str(row.get("play_pass", "")).upper()
    if play != "PLAY":
        return None

    units = _safe_number(row.get("units"))
    if math.isnan(units) or units <= 0:
        return None

    primary_market = str(row.get("primary_market", "")).upper()
    primary_side = str(row.get("primary_side", "")).upper()
    if not primary_market or not primary_side:
        return None

    market_type = None
    line_at_bet: object = ""
    price_at_bet: object = np.nan
    model_prob = np.nan
    market_prob = np.nan
    side = primary_side

    if "ML" in primary_market:
        market_type = "moneyline"
        price_at_bet = row.get("home_ml") if primary_side == "HOME" else row.get("away_ml")
        model_prob = _safe_number(row.get("model_home_prob"))
        market_prob = _safe_number(row.get("market_home_prob"))
        if primary_side == "AWAY":
            model_prob = 1.0 - model_prob if np.isfinite(model_prob) else model_prob
            market_prob = 1.0 - market_prob if np.isfinite(market_prob) else market_prob
    elif "ATS" in primary_market or "SPREAD" in primary_market:
        market_type = "spread"
        line_at_bet = row.get("home_spread")
        price_at_bet = row.get("spread_price")
    elif "TOTAL" in primary_market:
        market_type = "total"
        line_at_bet = row.get("total_points")
        if primary_side == "OVER":
            price_at_bet = row.get("total_over_price")
        elif primary_side == "UNDER":
            price_at_bet = row.get("total_under_price")

    if market_type is None:
        return None

    unit_dollars = _safe_number(row.get("unit_dollars"))
    if math.isnan(unit_dollars):
        unit_dollars = 10.0
    stake = units * unit_dollars

    price_decimal = _american_to_decimal(price_at_bet)
    date_str = _normalize_date_str(row.get("date"))

    bet_id = _hash_bet_id(
        date_str,
        sport,
        canon_team(row.get("home")),
        canon_team(row.get("away")),
        market_type,
        side,
        line_at_bet,
        price_at_bet,
    )

    return {
        "bet_id": bet_id,
        "date": date_str,
        "sport": sport,
        "home": row.get("home"),
        "away": row.get("away"),
        "market_type": market_type,
        "side": side,
        "line_at_bet": line_at_bet if not pd.isna(line_at_bet) else "",
        "price_at_bet": price_at_bet,
        "price_decimal": price_decimal,
        "model_prob": model_prob,
        "market_prob": market_prob,
        "edge": _bet_edge_for_market(row, market_type),
        "confidence": row.get("confidence"),
        "value_tier": row.get("value_tier"),
        "units": units,
        "unit_dollars": unit_dollars,
        "stake_dollars": stake,
        "result": "",
        "notes": row.get("primary_recommendation", ""),
    }


def _preferred_units(row: pd.Series, default_unit: float) -> Tuple[float, float]:
    unit_dollars = _safe_number(row.get("unit_dollars"))
    if math.isnan(unit_dollars):
        unit_dollars = float(default_unit)

    bet_size = _safe_number(row.get("bet_size"))
    units = _safe_number(row.get("units"))

    if math.isnan(units) and not math.isnan(bet_size) and unit_dollars > 0:
        units = bet_size / unit_dollars
    if math.isnan(units):
        units = 0.0

    stake = bet_size if not math.isnan(bet_size) else units * unit_dollars
    return float(units), float(stake)


def _extract_side(rec: str, side_hint: str, allowed: Iterable[str]) -> Optional[str]:
    if side_hint:
        u = str(side_hint).upper().strip()
        if u in allowed:
            return u
    urec = str(rec).upper()
    for opt in allowed:
        if opt in urec:
            return opt
    return None


def _build_bets_from_row(row: pd.Series, sport: str, unit_dollars_default: float) -> List[Dict[str, object]]:
    if not _is_play_row(row):
        return []

    units, stake = _preferred_units(row, unit_dollars_default)

    base = {
        "date": row.get("date"),
        "sport": sport,
        "home": row.get("home"),
        "away": row.get("away"),
        "home_canon": canon_team(row.get("home")),
        "away_canon": canon_team(row.get("away")),
        "unit_dollars": _safe_number(row.get("unit_dollars")) if not math.isnan(_safe_number(row.get("unit_dollars"))) else float(unit_dollars_default),
        "units": units,
        "stake_dollars": stake,
        "model_prob": row.get("model_home_prob"),
        "market_prob": row.get("market_home_prob"),
        "edge": row.get("edge_home"),
        "confidence": row.get("confidence"),
        "value_tier": row.get("value_tier"),
        "primary_recommendation": row.get("primary_recommendation"),
        "why_primary": row.get("why_primary"),
    }

    bets: List[Dict[str, object]] = []

    # Moneyline
    if _is_playable_reco(row.get("ml_recommendation")):
        side = _extract_side(row.get("ml_recommendation", ""), row.get("ml_ev_side", ""), {"HOME", "AWAY"})
        if side:
            price = row.get("home_ml") if side == "HOME" else row.get("away_ml")
            bets.append({
                **base,
                "market": "moneyline",
                "side": side,
                "line": np.nan,
                "price_american": _safe_number(price),
            })

    # Spread (home spread line)
    if _is_playable_reco(row.get("spread_recommendation")) and not pd.isna(row.get("home_spread")):
        side = _extract_side(row.get("spread_recommendation", ""), row.get("ats_ev_side", ""), {"HOME", "AWAY"})
        if side:
            bets.append({
                **base,
                "market": "spread",
                "side": side,
                "line": _safe_number(row.get("home_spread")),
                "price_american": _safe_number(row.get("spread_price", -110)),
            })

    # Totals
    if _is_playable_reco(row.get("total_recommendation")) and not pd.isna(row.get("total_points")):
        side = _extract_side(row.get("total_recommendation", ""), row.get("total_ev_side", ""), {"OVER", "UNDER"})
        if side:
            price = row.get("total_over_price") if side == "OVER" else row.get("total_under_price")
            bets.append({
                **base,
                "market": "total",
                "side": side,
                "line": _safe_number(row.get("total_points")),
                "price_american": _safe_number(price),
            })

    return bets


def _normalize_bets(df: pd.DataFrame, sport: str, unit_dollars_default: float) -> pd.DataFrame:
    bets: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        bets.extend(_build_bets_from_row(row, sport, unit_dollars_default))
    if not bets:
        return pd.DataFrame()
    out = pd.DataFrame(bets)
    out["date"] = out["date"].apply(lambda d: _parse_date(d) or d)
    out["price_decimal"] = out["price_american"].apply(_american_to_decimal)
    out["bet_id"] = out.apply(
        lambda r: f"{r.get('date')}|{r.get('sport')}|{r.get('home_canon')}|{r.get('away_canon')}|{r.get('market')}|{r.get('side')}|{r.get('line')}",
        axis=1,
    )
    return out


def append_bets_from_predictions(preds_df: pd.DataFrame, sport: str, bet_log_path: str = "results/bet_log.csv") -> int:
    """Append PLAY rows from today's predictions into the long-term bet_log.

    Returns the number of new bets written (deduped by bet_id).
    """

    if preds_df is None or preds_df.empty:
        return 0

    bets: List[Dict[str, object]] = []
    for _, row in preds_df.iterrows():
        bet = _build_bet_from_primary(row, sport)
        if bet:
            bets.append(bet)

    if not bets:
        return 0

    os.makedirs(os.path.dirname(bet_log_path) or ".", exist_ok=True)

    new_df = pd.DataFrame(bets)
    existing_cols: List[str] = []
    existing_ids: set = set()
    log_exists = os.path.exists(bet_log_path)
    if log_exists:
        existing = pd.read_csv(bet_log_path)
        existing_cols = list(existing.columns)
        if "bet_id" in existing.columns:
            existing_ids = set(existing["bet_id"].astype(str))

    if existing_ids:
        new_df = new_df[~new_df["bet_id"].astype(str).isin(existing_ids)]

    if new_df.empty:
        return 0

    if existing_cols:
        column_order = existing_cols
    else:
        column_order = list(dict.fromkeys(BET_LOG_COLUMNS + list(new_df.columns)))

    for col in column_order:
        if col not in new_df.columns:
            new_df[col] = np.nan

    new_df = new_df[column_order]
    new_df.to_csv(bet_log_path, mode="a", header=not log_exists, index=False)

    return len(new_df)


def _events_to_scores_df(events: List[Dict[str, object]], target_date: date) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ev in events or []:
        commence = ev.get("commence_time") or ""
        try:
            dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00")).date()
        except Exception:
            dt = None
        if dt and dt != target_date:
            continue

        home = canon_team(ev.get("home_team")) if ev.get("home_team") else None
        away = canon_team(ev.get("away_team")) if ev.get("away_team") else None
        if not home or not away:
            continue

        home_score = away_score = np.nan
        for sc in ev.get("scores", []) or []:
            try:
                name = canon_team(sc.get("name"))
                score_val = float(sc.get("score"))
            except Exception:
                continue
            if name == home:
                home_score = score_val
            elif name == away:
                away_score = score_val

        rows.append({
            "home_canon": home,
            "away_canon": away,
            "home_score": home_score,
            "away_score": away_score,
        })

    return pd.DataFrame(rows)


def _fetch_scores_df(sport: str, target_date: date) -> pd.DataFrame:
    sport_key = SPORT_TO_ODDS_KEY.get(sport)
    if not sport_key:
        return pd.DataFrame()
    events = fetch_scores_history_by_day(sport_key, as_of_date=target_date, days_back=1)
    return _events_to_scores_df(events, target_date)


def _grade_bets(bets: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets
    merged = bets.merge(scores, on=["home_canon", "away_canon"], how="left")

    results: List[str] = []
    profits: List[float] = []
    payouts: List[float] = []
    stakes: List[float] = []

    for _, row in merged.iterrows():
        market = str(row.get("market", "")).lower()
        side = str(row.get("side", ""))
        hs = row.get("home_score")
        aws = row.get("away_score")
        line = row.get("line")

        if market == "moneyline":
            result = grade_moneyline(side, hs, aws)
        elif market == "spread":
            result = grade_spread(side, line, hs, aws) if not pd.isna(line) else "MISSING_SCORE"
        elif market == "total":
            result = grade_total(side, line, hs, aws) if not pd.isna(line) else "MISSING_SCORE"
        else:
            result = "MISSING_SCORE"

        stake, profit, payout = _calc_profit_and_payout(
            row.get("units", 0.0),
            row.get("unit_dollars", 0.0),
            row.get("price_american", row.get("price_at_bet", np.nan)),
            result,
        )
        results.append(result)
        profits.append(profit)
        payouts.append(payout)
        stakes.append(stake)

    merged["result"] = results
    merged["profit_dollars"] = profits
    merged["payout_dollars"] = payouts
    merged["stake_dollars"] = stakes
    return merged


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cum = series.cumsum()
    peaks = cum.cummax()
    drawdowns = peaks - cum
    return float(drawdowns.max()) if not drawdowns.empty else 0.0


def _summaries(history: pd.DataFrame) -> Dict[str, object]:
    def _summary_for(df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            return {
                "bets": 0,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "win_pct": 0.0,
                "roi": 0.0,
                "profit": 0.0,
                "stake": 0.0,
                "max_drawdown": 0.0,
            }

        graded = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]
        wins = int((graded["result"] == "WIN").sum())
        losses = int((graded["result"] == "LOSS").sum())
        pushes = int((graded["result"] == "PUSH").sum())
        total_stake = graded.get("stake_dollars", pd.Series(dtype=float)).sum()
        profit = graded.get("profit_dollars", pd.Series(dtype=float)).sum()
        roi = float(profit / total_stake) if total_stake else 0.0
        win_pct = float(wins / (wins + losses)) if (wins + losses) else 0.0
        return {
            "bets": int(len(graded)),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_pct": win_pct,
            "roi": roi,
            "profit": float(profit),
            "stake": float(total_stake),
            "max_drawdown": _max_drawdown(graded.get("profit_dollars", pd.Series(dtype=float))),
        }

    def _group_summary(df: pd.DataFrame, col: str) -> Dict[str, Dict[str, object]]:
        if df.empty or col not in df.columns:
            return {}
        grouped: Dict[str, Dict[str, object]] = {}
        for key, sub in df.groupby(col):
            if pd.isna(key) or str(key).strip() == "":
                continue
            grouped[str(key)] = {
                "win_pct": _summary_for(sub).get("win_pct", 0.0),
                "roi": _summary_for(sub).get("roi", 0.0),
                "bets": _summary_for(sub).get("bets", 0),
            }
        return grouped

    def _longshot_filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "price_at_bet" in df.columns:
            price = pd.to_numeric(df["price_at_bet"], errors="coerce")
            return df[price >= 500]
        if "price_decimal" in df.columns:
            price = pd.to_numeric(df["price_decimal"], errors="coerce")
            return df[price >= 6.0]
        return df.iloc[0:0]

    def _odds_bucket(value: object) -> str:
        try:
            odds = float(value)
        except Exception:
            return "UNKNOWN"
        if not np.isfinite(odds):
            return "UNKNOWN"
        if odds >= 400:
            return ">=+400"
        if odds >= 250:
            return "+250 to +399"
        if odds >= 100:
            return "+100 to +249"
        if odds >= -110:
            return "-110 to +99"
        if odds >= -200:
            return "-200 to -111"
        return "<=-200"

    today = date.today()
    last_30 = today - timedelta(days=30)

    hist_with_date = history.copy()
    hist_with_date["date"] = hist_with_date["date"].apply(lambda d: _parse_date(d) or d)
    if "price_at_bet" in hist_with_date.columns:
        hist_with_date["odds_bucket"] = hist_with_date["price_at_bet"].apply(_odds_bucket)
    recent = hist_with_date[hist_with_date["date"] >= last_30] if not hist_with_date.empty else pd.DataFrame()

    return {
        "lifetime": _summary_for(hist_with_date),
        "last_30_days": _summary_for(recent),
        "by_sport": _group_summary(hist_with_date, "sport"),
        "by_confidence": _group_summary(hist_with_date, "confidence"),
        "by_value_tier": _group_summary(hist_with_date, "value_tier"),
        "by_odds_bucket": _group_summary(hist_with_date, "odds_bucket"),
        "longshots": _summary_for(_longshot_filter(hist_with_date)),
    }


def _write_outputs(
    bets: pd.DataFrame,
    target_date: date,
    tracking_dir: str,
) -> Tuple[str, pd.DataFrame, Dict[str, object]]:
    os.makedirs(tracking_dir, exist_ok=True)

    date_str = target_date.isoformat()
    daily_path = os.path.join(tracking_dir, f"bets_{date_str}_graded.csv")
    bets.to_csv(daily_path, index=False)

    history_path = os.path.join(tracking_dir, "bet_history.csv")
    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, bets], ignore_index=True)
    combined = combined.drop_duplicates(subset=["bet_id"], keep="last")
    combined.sort_values(by=["date", "sport", "home_canon", "away_canon", "market"], inplace=True)
    combined.to_csv(history_path, index=False)

    summary = _summaries(combined)
    summary_path = os.path.join(tracking_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return daily_path, combined, summary


def _grade_bet_log_rows(bets: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return bets

    working = bets.copy()
    working["home_canon"] = working["home"].apply(canon_team)
    working["away_canon"] = working["away"].apply(canon_team)
    merged = working.merge(scores, on=["home_canon", "away_canon"], how="left")

    results: List[str] = []
    profits: List[float] = []
    price_decimals: List[float] = []
    stakes: List[float] = []
    payouts: List[float] = []

    for _, row in merged.iterrows():
        market = str(row.get("market_type", "")).lower()
        side = str(row.get("side", ""))
        hs = row.get("home_score")
        aws = row.get("away_score")
        line = row.get("line_at_bet")

        price_decimal = row.get("price_decimal")
        if pd.isna(price_decimal) or price_decimal == "":
            price_decimal = _american_to_decimal(row.get("price_at_bet"))

        units = _safe_number(row.get("units"))
        unit_dollars = _safe_number(row.get("unit_dollars"))
        if math.isnan(unit_dollars):
            unit_dollars = 10.0

        if market == "moneyline":
            result = grade_moneyline(side, hs, aws)
        elif market == "spread":
            result = grade_spread(side, line, hs, aws) if not pd.isna(line) else "MISSING_SCORE"
        elif market == "total":
            result = grade_total(side, line, hs, aws) if not pd.isna(line) else "MISSING_SCORE"
        else:
            result = "MISSING_SCORE"

        stake, profit, payout = _calc_profit_and_payout(
            units,
            unit_dollars,
            row.get("price_at_bet"),
            result,
        )

        results.append(result)
        profits.append(profit)
        payouts.append(payout)
        price_decimals.append(price_decimal)
        stakes.append(stake)

    merged["result"] = results
    merged["profit_dollars"] = profits
    merged["payout_dollars"] = payouts
    merged["price_decimal"] = price_decimals
    merged["stake_dollars"] = stakes
    return merged


# ---------------------------
# Public API
# ---------------------------

@dataclass
class TrackingResult:
    recs_path: Optional[str]
    daily_path: Optional[str]
    summary: Dict[str, object]
    bets_df: pd.DataFrame
    history_df: pd.DataFrame
    ok: bool
    reason: str = ""


@dataclass
class BetLogTrackingResult:
    updated_log: pd.DataFrame
    graded_bets: pd.DataFrame
    summary: Dict[str, object]
    ok: bool
    reason: str = ""


def track_date(
    *,
    sport: str,
    target_date: date,
    results_dir: str = "results",
    tracking_dir: str = "results/tracking",
    unit_dollars_default: float = 10.0,
) -> TrackingResult:
    recs_path = _find_recs_csv(results_dir, target_date)
    if not recs_path or not os.path.exists(recs_path):
        return TrackingResult(None, None, {}, pd.DataFrame(), pd.DataFrame(), ok=False, reason="No recommendations CSV found")

    df = pd.read_csv(recs_path)
    if df.empty:
        return TrackingResult(recs_path, None, {}, pd.DataFrame(), pd.DataFrame(), ok=False, reason="Recommendations file empty")

    if "date" not in df.columns or "home" not in df.columns or "away" not in df.columns:
        return TrackingResult(recs_path, None, {}, pd.DataFrame(), pd.DataFrame(), ok=False, reason="Missing required columns (date/home/away)")

    # Filter to the target date if present
    df = df.copy()
    df["date"] = df["date"].apply(lambda d: _parse_date(d) or d)
    df = df[df["date"] == target_date]
    if df.empty:
        return TrackingResult(
            recs_path,
            None,
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            ok=False,
            reason="No rows for target date",
        )

    bets = _normalize_bets(df, sport=sport, unit_dollars_default=unit_dollars_default)
    if bets.empty:
        return TrackingResult(recs_path, None, {}, bets, pd.DataFrame(), ok=False, reason="No playable bets found")

    scores_df = _fetch_scores_df(sport, target_date)
    graded = _grade_bets(bets, scores_df)

    daily_path, history_df, summary = _write_outputs(graded, target_date, tracking_dir)
    return TrackingResult(recs_path, daily_path, summary, graded, history_df, ok=True)


def track_yesterday(
    sport: str,
    *,
    results_dir: str = "results",
    tracking_dir: str = "results/tracking",
    unit_dollars_default: float = 10.0,
) -> TrackingResult:
    yday = date.today() - timedelta(days=1)
    return track_date(
        sport=sport,
        target_date=yday,
        results_dir=results_dir,
        tracking_dir=tracking_dir,
        unit_dollars_default=unit_dollars_default,
    )


def track_bets_for_date(
    sport: str,
    target_date: date,
    *,
    bet_log_path: str = "results/tracking/bet_log.csv",
    unit_dollars_default: float = 10.0,
) -> BetLogTrackingResult:
    if not os.path.exists(bet_log_path):
        return BetLogTrackingResult(pd.DataFrame(), pd.DataFrame(), {}, ok=False, reason="bet_log.csv not found")

    log_df = pd.read_csv(bet_log_path)
    if log_df.empty:
        return BetLogTrackingResult(log_df, pd.DataFrame(), {}, ok=False, reason="bet_log.csv is empty")

    working = log_df.copy()
    working["date_parsed"] = working["date"].apply(lambda d: _parse_date(d) or d)
    working["sport_lower"] = working.get("sport", "").astype(str).str.lower()

    mask_date = working["date_parsed"] == target_date
    mask_sport = working["sport_lower"] == str(sport).lower()
    mask_ungraded = working["result"].isna() | (working["result"].astype(str) == "")

    pending = working[mask_date & mask_sport & mask_ungraded]
    if pending.empty:
        return BetLogTrackingResult(log_df, pd.DataFrame(), {}, ok=False, reason="No open bets to grade")

    pending = pending.copy()
    pending["unit_dollars"] = pending.get("unit_dollars", unit_dollars_default).fillna(unit_dollars_default)
    pending["units"] = pending.get("units", 0.0).fillna(0.0)

    scores_df = _fetch_scores_df(sport, target_date)
    graded = _grade_bet_log_rows(pending, scores_df)

    os.makedirs(os.path.dirname(bet_log_path) or ".", exist_ok=True)
    graded_out_path = os.path.join(
        os.path.dirname(bet_log_path) or ".",
        f"graded_{sport}_{target_date.isoformat()}.csv",
    )
    graded.to_csv(graded_out_path, index=False)

    log_indexed = log_df.set_index("bet_id")
    for _, row in graded.iterrows():
        bid = row.get("bet_id")
        if bid not in log_indexed.index:
            continue
        for col in [
            "result",
            "home_score",
            "away_score",
            "price_decimal",
            "stake_dollars",
            "profit_dollars",
            "payout_dollars",
        ]:
            log_indexed.loc[bid, col] = row.get(col)

    log_df_updated = log_indexed.reset_index()
    log_df_updated.to_csv(bet_log_path, index=False)

    tracking_dir = os.path.dirname(bet_log_path) or "results/tracking"
    os.makedirs(tracking_dir, exist_ok=True)
    summary_full = _summaries(log_df_updated)
    summary_path = os.path.join(tracking_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_full, f, indent=2)

    played = graded[graded["result"].isin(["WIN", "LOSS", "PUSH"])]
    wins = int((played["result"] == "WIN").sum())
    losses = int((played["result"] == "LOSS").sum())
    pushes = int((played["result"] == "PUSH").sum())
    total_stake = float(played.get("stake_dollars", pd.Series(dtype=float)).sum())
    profit = float(played.get("profit_dollars", pd.Series(dtype=float)).sum())
    roi = float(profit / total_stake) if total_stake else 0.0

    summary = {
        "graded": int(len(graded)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "profit": profit,
        "roi": roi,
        "graded_path": graded_out_path,
        "summary_path": summary_path,
        "by_sport": summary_full.get("by_sport", {}),
        "by_confidence": summary_full.get("by_confidence", {}),
        "by_value_tier": summary_full.get("by_value_tier", {}),
        "by_odds_bucket": summary_full.get("by_odds_bucket", {}),
    }

    return BetLogTrackingResult(log_df_updated, graded, summary, ok=True)
