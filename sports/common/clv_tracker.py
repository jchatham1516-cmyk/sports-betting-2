# sports/common/clv_tracker.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


CLV_LOG_PATH = os.getenv("CLV_LOG_PATH", "results/clv_log.csv")


# -----------------------------
# Odds helpers
# -----------------------------
def american_to_prob(ml: float) -> float:
    ml = float(ml)
    if ml == 0:
        return float("nan")
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)


def no_vig_two_way_probs(price_a: float, price_b: float) -> Tuple[float, float]:
    """
    Convert two American prices into no-vig implied probabilities.
    Returns (p_a, p_b) that sum to 1.
    """
    pa = american_to_prob(price_a)
    pb = american_to_prob(price_b)
    if np.isnan(pa) or np.isnan(pb) or (pa + pb) <= 0:
        return (float("nan"), float("nan"))
    s = pa + pb
    return (pa / s, pb / s)


def normalize_team(s: str) -> str:
    return " ".join(str(s or "").strip().split())


def bet_id(
    sport: str,
    date_str: str,
    home: str,
    away: str,
    market: str,
    side: str,
) -> str:
    """
    Stable identifier for a specific bet.
    """
    return f"{sport}|{date_str}|{normalize_team(home)}|{normalize_team(away)}|{market}|{side}"


# -----------------------------
# Core CLV logging
# -----------------------------
@dataclass
class BetSnapshot:
    sport: str
    date: str
    home: str
    away: str
    market: str  # ML / ATS / TOTAL
    side: str    # HOME/AWAY or OVER/UNDER
    open_price: float
    close_price: Optional[float] = None
    open_line: Optional[float] = None   # spread or total
    close_line: Optional[float] = None  # spread or total


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _read_existing_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_open_snapshot(
    *,
    predictions_csv: str,
    sport: str,
    out_path: str = CLV_LOG_PATH,
    only_play_pass: bool = True,
) -> int:
    """
    Reads your predictions CSV and logs OPEN snapshots for bets you plan to take.
    You should run this at the time you place bets (the "open").
    """
    df = pd.read_csv(predictions_csv)

    if only_play_pass and "play_pass" in df.columns:
        df = df[df["play_pass"].astype(str).str.upper().eq("PLAY")].copy()

    if df.empty:
        _ensure_parent(out_path)
        # still create file if missing
        if not os.path.exists(out_path):
            pd.DataFrame().to_csv(out_path, index=False)
        return 0

    rows = []
    for _, r in df.iterrows():
        date_str = str(r.get("date", ""))
        home = str(r.get("home", ""))
        away = str(r.get("away", ""))

        primary = str(r.get("primary_recommendation", ""))
        # Decide market/side/price/line from your columns
        market, side, price, line = _extract_bet_from_prediction_row(r)

        if market is None:
            continue

        bid = bet_id(sport, date_str, home, away, market, side)

        rows.append(
            {
                "bet_id": bid,
                "sport": sport,
                "date": date_str,
                "home": home,
                "away": away,
                "market": market,
                "side": side,
                "open_price": price,
                "open_line": line,
                "open_ts_utc": datetime.utcnow().isoformat(timespec="seconds"),
                "close_price": np.nan,
                "close_line": np.nan,
                "close_ts_utc": "",
                "clv_price_american": np.nan,
                "clv_prob_no_vig": np.nan,
                "clv_notes": "",
                "primary_recommendation": primary,
            }
        )

    if not rows:
        return 0

    _ensure_parent(out_path)
    old = _read_existing_log(out_path)

    new = pd.DataFrame(rows)

    # Dedup: don't re-add an open snapshot if bet_id already exists
    if not old.empty and "bet_id" in old.columns:
        existing = set(old["bet_id"].astype(str).tolist())
        new = new[~new["bet_id"].astype(str).isin(existing)].copy()

    if new.empty:
        return 0

    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new
    combined.to_csv(out_path, index=False)
    return int(len(new))


def update_close_snapshot_from_predictions(
    *,
    predictions_csv_close: str,
    sport: str,
    out_path: str = CLV_LOG_PATH,
) -> int:
    """
    Update the CLV log with "close" prices/lines using a later predictions CSV
    (i.e., rerun your script near game time to capture near-close odds).
    """
    close_df = pd.read_csv(predictions_csv_close)
    log = _read_existing_log(out_path)
    if log.empty:
        return 0
    if "bet_id" not in log.columns:
        return 0

    # Index log by bet_id for updates
    log_idx = {str(b): i for i, b in enumerate(log["bet_id"].astype(str).tolist())}

    updated = 0
    for _, r in close_df.iterrows():
        date_str = str(r.get("date", ""))
        home = str(r.get("home", ""))
        away = str(r.get("away", ""))

        market, side, close_price, close_line = _extract_bet_from_prediction_row(r)
        if market is None:
            continue

        bid = bet_id(sport, date_str, home, away, market, side)
        i = log_idx.get(bid)
        if i is None:
            continue

        # Only update if close not already set (or if you want to overwrite blanks)
        open_price = _safe_float(log.loc[i, "open_price"])
        open_line = _safe_float(log.loc[i, "open_line"])

        log.loc[i, "close_price"] = close_price
        log.loc[i, "close_line"] = close_line
        log.loc[i, "close_ts_utc"] = datetime.utcnow().isoformat(timespec="seconds")

        # CLV metrics
        log.loc[i, "clv_price_american"] = _american_clv(open_price, close_price)
        log.loc[i, "clv_prob_no_vig"] = _prob_clv_proxy(market, side, r, open_price, close_price)
        updated += 1

    log.to_csv(out_path, index=False)
    return int(updated)


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _american_clv(open_price: float, close_price: float) -> float:
    """
    Positive is "better for you" if you got a better number than close.
    For American odds: For favorites (negative), getting closer to 0 is better.
    For underdogs (positive), getting bigger is better.
    A simple way: compare implied prob (no-vig is better, but we often only have one price).
    Here we just do: open_implied - close_implied (positive means you beat the close).
    """
    op = american_to_prob(open_price)
    cp = american_to_prob(close_price)
    if np.isnan(op) or np.isnan(cp):
        return float("nan")
    return float(cp - op)  # close_prob - open_prob; if you took lower implied prob, you beat close


def _prob_clv_proxy(market: str, side: str, row: pd.Series, open_price: float, close_price: float) -> float:
    """
    If you can provide both sides at open/close you can do true no-vig CLV.
    With just one side, we do implied prob delta.
    """
    op = american_to_prob(open_price)
    cp = american_to_prob(close_price)
    if np.isnan(op) or np.isnan(cp):
        return float("nan")
    # Positive means you beat close (you got a better price)
    return float(cp - op)


def _extract_bet_from_prediction_row(r: pd.Series) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
    """
    Convert your prediction row to a canonical bet:
      - market: "ML" | "ATS" | "TOTAL"
      - side: "HOME" | "AWAY" | "OVER" | "UNDER"
      - price: American odds
      - line: spread or total number (optional)
    Requires your CSV to include:
      ML: home_ml, away_ml
      ATS: home_spread, spread_price
      TOTAL: total_points, total_over_price, total_under_price
    and a usable primary_recommendation (or ml/spread/total recommendation).
    """
    # Prefer explicit columns if present
    primary = str(r.get("primary_recommendation", "")).upper()

    # TOTAL
    if "TOTAL" in primary or "PICK TOTAL" in primary:
        side = "OVER" if "OVER" in primary else ("UNDER" if "UNDER" in primary else None)
        if side is None:
            return (None, None, None, None)
        total = _safe_float(r.get("total_points"))
        price = _safe_float(r.get("total_over_price" if side == "OVER" else "total_under_price"))
        return ("TOTAL", side, price, total)

    # ATS
    if "ATS" in primary:
        # your model uses "Model PICK ATS: HOME/AWAY"
        if "HOME" in primary:
            side = "HOME"
        elif "AWAY" in primary:
            side = "AWAY"
        else:
            return (None, None, None, None)
        line = _safe_float(r.get("home_spread"))
        price = _safe_float(r.get("spread_price"))
        return ("ATS", side, price, line)

    # ML default
    # If primary doesn't mention, fall back to ml_recommendation / spread_recommendation / total_recommendation
    mlr = str(r.get("ml_recommendation", "")).upper()
    if "HOME" in mlr:
        side = "HOME"
    elif "AWAY" in mlr:
        side = "AWAY"
    elif "HOME" in primary:
        side = "HOME"
    elif "AWAY" in primary:
        side = "AWAY"
    else:
        return (None, None, None, None)

    price = _safe_float(r.get("home_ml" if side == "HOME" else "away_ml"))
    return ("ML", side, price, None)
