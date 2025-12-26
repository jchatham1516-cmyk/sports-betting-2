# sports/common/clv_tracker.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.odds_sources import load_odds_for_date_from_api, SPORT_TO_ODDS_KEY


# -----------------------------
# Basic pricing helpers
# -----------------------------
def american_to_prob(price: float) -> float:
    try:
        price = float(price)
        if price == 0:
            return float("nan")
        if price > 0:
            return 100.0 / (price + 100.0)
        return (-price) / ((-price) + 100.0)
    except Exception:
        return float("nan")


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _canon_str(x) -> str:
    return str(x).strip().lower().replace("  ", " ")


def make_bet_id(
    *,
    sport: str,
    game_date: str,   # "MM/DD/YYYY"
    home: str,
    away: str,
    market: str,      # "ML" | "ATS" | "TOTAL"
    side: str,        # "HOME"|"AWAY"|"OVER"|"UNDER"
    line: Optional[float],
) -> str:
    """
    Deterministic ID so we can match open <-> close even across separate runs.

    IMPORTANT: do NOT include price in the bet_id — price can change between open/close.
    """
    line_s = "na" if line is None or (isinstance(line, float) and np.isnan(line)) else f"{float(line):.3f}"
    return "|".join(
        [
            _canon_str(game_date),
            _canon_str(sport),
            _canon_str(home),
            _canon_str(away),
            _canon_str(market),
            _canon_str(side),
            line_s,
        ]
    )


# -----------------------------
# CLV log schema
# -----------------------------
CLV_COLUMNS = [
    "ts_open_utc",
    "ts_close_utc",
    "sport",
    "game_date",
    "home",
    "away",
    "market",
    "side",
    "bet_id",

    # open
    "open_price",
    "open_line",

    # close
    "close_price",
    "close_line",

    # CLV metrics
    "open_imp_prob",
    "close_imp_prob",
    "clv_imp_prob",     # close_imp_prob - open_imp_prob (positive = good)

    "clv_line",         # favorable line movement in points/goals
    "notes",
]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _read_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=CLV_COLUMNS)
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return pd.DataFrame(columns=CLV_COLUMNS)
        return df
    except Exception:
        return pd.DataFrame(columns=CLV_COLUMNS)


def _append_rows(path: str, rows: list[dict]) -> None:
    _ensure_dir(path)
    df_old = _read_log(path)
    df_new = pd.DataFrame(rows)
    df = pd.concat([df_old, df_new], ignore_index=True)
    # de-dupe on bet_id + ts_open_utc (keep first)
    if "bet_id" in df.columns and "ts_open_utc" in df.columns:
        df = df.drop_duplicates(subset=["bet_id", "ts_open_utc"], keep="first")
    df.to_csv(path, index=False)


# -----------------------------
# Extract "what did we bet?"
# -----------------------------
def _extract_bet_fields(row: pd.Series) -> Tuple[str, str, float, float]:
    """
    Returns:
      market, side, line, price
    """
    primary = str(row.get("primary_recommendation", "") or "")
    mlr = str(row.get("ml_recommendation", "") or "")
    sr = str(row.get("spread_recommendation", "") or "")
    tr = str(row.get("total_recommendation", "") or "")

    # Prefer primary; fallback to any "Model PICK ..."
    s = primary
    if not s.startswith("Model"):
        if mlr.startswith("Model"):
            s = mlr
        elif sr.startswith("Model"):
            s = sr
        elif tr.startswith("Model"):
            s = tr

    # Defaults
    market = "ML"
    side = "HOME"
    line = np.nan
    price = np.nan

    # TOTAL
    if "TOTAL" in s or "Model PICK TOTAL" in s:
        market = "TOTAL"
        if "OVER" in s:
            side = "OVER"
            line = _safe_float(row.get("total_points"))
            price = _safe_float(row.get("total_over_price"))
        elif "UNDER" in s:
            side = "UNDER"
            line = _safe_float(row.get("total_points"))
            price = _safe_float(row.get("total_under_price"))
        return market, side, line, price

    # ATS
    if "ATS" in s:
        market = "ATS"
        if "HOME" in s:
            side = "HOME"
            line = _safe_float(row.get("home_spread"))
            price = _safe_float(row.get("spread_price"))
        elif "AWAY" in s:
            side = "AWAY"
            # away line is the negative of home_spread
            hs = _safe_float(row.get("home_spread"))
            line = -hs if not np.isnan(hs) else np.nan
            price = _safe_float(row.get("spread_price"))
        return market, side, line, price

    # ML
    market = "ML"
    if "AWAY" in s:
        side = "AWAY"
        price = _safe_float(row.get("away_ml"))
    else:
        side = "HOME"
        price = _safe_float(row.get("home_ml"))
    line = np.nan
    return market, side, line, price


# -----------------------------
# Public API: log "open" lines
# -----------------------------
def log_open_from_predictions(
    preds_df: pd.DataFrame,
    *,
    sport: str,
    game_date: str,
    clv_log_path: str = "results/clv_log.csv",
    only_plays: bool = True,
) -> pd.DataFrame:
    """
    Writes "open" lines to CLV log. Returns preds_df with bet_id column added.

    For best results:
      - run this when you actually place bets (your "open" snapshot)
      - then run update_closing_lines_from_api() later near tip/puck drop/kickoff
    """
    if preds_df is None or preds_df.empty:
        return preds_df

    out = preds_df.copy()
    if "bet_id" not in out.columns:
        out["bet_id"] = ""

    rows = []
    ts_open = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for i in out.index:
        if only_plays:
            if str(out.loc[i, "play_pass"] or "").upper() != "PLAY":
                continue

        home = str(out.loc[i, "home"])
        away = str(out.loc[i, "away"])

        market, side, line, price = _extract_bet_fields(out.loc[i])

        # must have a price to track CLV
        if np.isnan(_safe_float(price)):
            continue

        bet_id = make_bet_id(
            sport=sport,
            game_date=game_date,
            home=home,
            away=away,
            market=market,
            side=side,
            line=None if np.isnan(_safe_float(line)) else float(line),
        )
        out.loc[i, "bet_id"] = bet_id

        open_imp = american_to_prob(price)

        rows.append(
            {
                "ts_open_utc": ts_open,
                "ts_close_utc": "",
                "sport": sport,
                "game_date": game_date,
                "home": home,
                "away": away,
                "market": market,
                "side": side,
                "bet_id": bet_id,
                "open_price": float(price),
                "open_line": (np.nan if np.isnan(_safe_float(line)) else float(line)),
                "close_price": np.nan,
                "close_line": np.nan,
                "open_imp_prob": float(open_imp) if not np.isnan(open_imp) else np.nan,
                "close_imp_prob": np.nan,
                "clv_imp_prob": np.nan,
                "clv_line": np.nan,
                "notes": "",
            }
        )

    if rows:
        _append_rows(clv_log_path, rows)

    return out


# -----------------------------
# Public API: update "close" lines
# -----------------------------
def update_closing_lines_from_api(
    *,
    sport: str,
    game_date: str,  # "MM/DD/YYYY"
    days_padding: int = 1,
    clv_log_path: str = "results/clv_log.csv",
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """
    Reads clv_log.csv and fills close_price/close_line for matching sport+date rows
    using the CURRENT odds snapshot from Odds API.

    Run this near game start to approximate "closing" lines.
    """
    df = _read_log(clv_log_path)
    if df.empty:
        return df

    # filter rows that need close and match sport/date
    mask = (
        (df.get("sport", "").astype(str).str.lower() == str(sport).lower())
        & (df.get("game_date", "").astype(str) == str(game_date))
        & (df.get("close_price").isna())
    )
    needs = df[mask].copy()
    if needs.empty:
        return df

    # Pull current odds snapshot (this is your "close" approximation)
    try:
        dt = datetime.strptime(game_date, "%m/%d/%Y")
    except Exception:
        # if parse fails, just return
        return df

    commence_from = dt.replace(tzinfo=timezone.utc) - timedelta(days=int(days_padding))
    commence_to = dt.replace(tzinfo=timezone.utc) + timedelta(days=int(days_padding), hours=23, minutes=59)

    sport_key = SPORT_TO_ODDS_KEY.get(str(sport).lower())
    if not sport_key:
        return df

    odds_dict = load_odds_for_date_from_api(
        sport_key=sport_key,
        commence_from=commence_from,
        commence_to=commence_to,
        markets="h2h,spreads,totals",
    )

    ts_close = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for idx in needs.index:
        home = str(df.loc[idx, "home"])
        away = str(df.loc[idx, "away"])
        market = str(df.loc[idx, "market"])
        side = str(df.loc[idx, "side"])

        oi = odds_dict.get((home, away))
        if oi is None:
            # Sometimes key ordering differs; try reverse
            oi = odds_dict.get((away, home))

        if not isinstance(oi, dict):
            continue

        close_price = np.nan
        close_line = np.nan

        if market == "ML":
            if side == "HOME":
                close_price = _safe_float(oi.get("home_ml"))
            else:
                close_price = _safe_float(oi.get("away_ml"))
        elif market == "ATS":
            hs = _safe_float(oi.get("home_spread"))
            sp = _safe_float(oi.get("spread_price"))
            if side == "HOME":
                close_line = hs
                close_price = sp
            else:
                close_line = -hs if not np.isnan(hs) else np.nan
                close_price = sp
        elif market == "TOTAL":
            tp = _safe_float(oi.get("total_points"))
            op = _safe_float(oi.get("over_price"))
            up = _safe_float(oi.get("under_price"))
            close_line = tp
            close_price = op if side == "OVER" else up

        if np.isnan(_safe_float(close_price)):
            continue

        df.loc[idx, "ts_close_utc"] = ts_close
        df.loc[idx, "close_price"] = float(close_price)
        if not np.isnan(_safe_float(close_line)):
            df.loc[idx, "close_line"] = float(close_line)

        # implied-prob CLV
        open_p = _safe_float(df.loc[idx, "open_imp_prob"])
        close_p = american_to_prob(close_price)
        df.loc[idx, "close_imp_prob"] = float(close_p) if not np.isnan(close_p) else np.nan
        if not np.isnan(open_p) and not np.isnan(close_p):
            df.loc[idx, "clv_imp_prob"] = float(close_p - open_p)

        # line CLV (favorable movement)
        # For ATS:
        #   HOME: you want close_line < open_line if you bet HOME -x, or close_line smaller if you bet +x
        #   A robust rule: for HOME, favorable = open_line - close_line. For AWAY, favorable = close_line - open_line.
        # For TOTAL:
        #   OVER: favorable = open_total - close_total (line goes down)
        #   UNDER: favorable = close_total - open_total (line goes up)
        ol = _safe_float(df.loc[idx, "open_line"])
        cl = _safe_float(df.loc[idx, "close_line"])
        if not np.isnan(ol) and not np.isnan(cl):
            if market == "ATS":
                if side == "HOME":
                    df.loc[idx, "clv_line"] = float(ol - cl)
                else:
                    df.loc[idx, "clv_line"] = float(cl - ol)
            elif market == "TOTAL":
                if side == "OVER":
                    df.loc[idx, "clv_line"] = float(ol - cl)
                else:
                    df.loc[idx, "clv_line"] = float(cl - ol)

        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))

    _ensure_dir(clv_log_path)
    df.to_csv(clv_log_path, index=False)
    return df


# -----------------------------
# Market health / stop-betting report
# -----------------------------
@dataclass
class MarketStopRule:
    min_bets: int = 30
    lookback_days: int = 60
    # If your average CLV is below this, stop.
    # For clv_imp_prob: -0.005 means you're losing ~0.5% implied-prob vs close on average.
    max_negative_mean_clv_imp_prob: float = -0.005


def market_health_report(
    *,
    clv_log_path: str = "results/clv_log.csv",
    rule: MarketStopRule = MarketStopRule(),
) -> pd.DataFrame:
    """
    Uses CLV as a *proxy* for whether your bets are beating the market.

    Output columns:
      market, n, mean_clv_imp_prob, median_clv_imp_prob, mean_clv_line
      recommendation: "OK" or "STOP"
    """
    df = _read_log(clv_log_path)
    if df.empty:
        return pd.DataFrame(
            columns=["market", "n", "mean_clv_imp_prob", "median_clv_imp_prob", "mean_clv_line", "recommendation"]
        )

    # time filter
    try:
        df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], errors="coerce", utc=True)
    except Exception:
        pass

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(rule.lookback_days))
    if "ts_open_utc" in df.columns:
        df = df[df["ts_open_utc"] >= cutoff]

    # only rows with close
    df = df[~df.get("clv_imp_prob").isna()]
    if df.empty:
        return pd.DataFrame(
            columns=["market", "n", "mean_clv_imp_prob", "median_clv_imp_prob", "mean_clv_line", "recommendation"]
        )

    out_rows = []
    for market, g in df.groupby("market"):
        n = int(len(g))
        mean_clv = float(np.mean(g["clv_imp_prob"].astype(float)))
        med_clv = float(np.median(g["clv_imp_prob"].astype(float)))
        mean_line = float(np.mean(g["clv_line"].astype(float))) if "clv_line" in g.columns else float("nan")

        rec = "OK"
        if n >= int(rule.min_bets) and mean_clv <= float(rule.max_negative_mean_clv_imp_prob):
            rec = "STOP"

        out_rows.append(
            {
                "market": str(market),
                "n": n,
                "mean_clv_imp_prob": mean_clv,
                "median_clv_imp_prob": med_clv,
                "mean_clv_line": mean_line,
                "recommendation": rec,
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["recommendation", "mean_clv_imp_prob"])
    return out


def markets_to_stop(
    *,
    clv_log_path: str = "results/clv_log.csv",
    rule: MarketStopRule = MarketStopRule(),
) -> Dict[str, bool]:
    """
    Returns dict like {"ML": False, "ATS": True, "TOTAL": True}
    """
    rep = market_health_report(clv_log_path=clv_log_path, rule=rule)
    stop = {"ML": False, "ATS": False, "TOTAL": False}
    if rep is None or rep.empty:
        return stop
    for _, r in rep.iterrows():
        m = str(r.get("market"))
        stop[m] = (str(r.get("recommendation")) == "STOP")
    return stop
