from __future__ import annotations

import glob
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
from sports.common.util import american_to_decimal, safe_float


# ---------------------------
# Helpers
# ---------------------------

DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y")


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


def _calc_profit_and_payout(stake: float, price_decimal: float, result: str) -> Tuple[float, float]:
    """Returns (profit, payout). Profit excludes stake return."""

    try:
        st = float(stake)
    except Exception:
        st = 0.0

    if st <= 0:
        return 0.0, 0.0

    try:
        dec = float(price_decimal)
    except Exception:
        dec = float("nan")

    if result == "WIN":
        if math.isnan(dec) or dec <= 0:
            return 0.0, st
        profit = st * (dec - 1.0)
        return profit, st + profit
    if result == "LOSS":
        return -st, 0.0
    if result == "PUSH":
        return 0.0, st
    return 0.0, 0.0


def _safe_number(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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

        profit, payout = _calc_profit_and_payout(row.get("stake_dollars", 0.0), row.get("price_decimal", np.nan), result)
        results.append(result)
        profits.append(profit)
        payouts.append(payout)

    merged["result"] = results
    merged["profit_dollars"] = profits
    merged["payout_dollars"] = payouts
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

    today = date.today()
    last_30 = today - timedelta(days=30)

    hist_with_date = history.copy()
    hist_with_date["date"] = hist_with_date["date"].apply(lambda d: _parse_date(d) or d)
    recent = hist_with_date[hist_with_date["date"] >= last_30] if not hist_with_date.empty else pd.DataFrame()

    return {"lifetime": _summary_for(hist_with_date), "last_30_days": _summary_for(recent)}


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
