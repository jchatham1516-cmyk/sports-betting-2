# sports/common/clv_tracker.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _american_to_prob(price: float) -> float:
    price = float(price)
    if price == 0:
        return float("nan")
    if price > 0:
        return 100.0 / (price + 100.0)
    return (-price) / ((-price) + 100.0)


def _no_vig_probs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    hp = _american_to_prob(home_ml)
    ap = _american_to_prob(away_ml)
    if np.isnan(hp) or np.isnan(ap) or (hp + ap) <= 0:
        return (float("nan"), float("nan"))
    s = hp + ap
    return (hp / s, ap / s)


def make_bet_id(
    *,
    sport: str,
    date_str: str,
    home: str,
    away: str,
    market: str,
    side: str,
    line: Optional[float] = None,
) -> str:
    """
    Stable ID so we can match an "open" record to a "close" record later.
    """
    ln = "" if line is None or (isinstance(line, float) and np.isnan(line)) else f"{float(line):g}"
    return f"{sport}|{date_str}|{home}|{away}|{market}|{side}|{ln}"


def infer_market_and_side(row: pd.Series) -> Tuple[str, str, Optional[float], Optional[float]]:
    """
    Returns:
      market: ML / ATS / TOTAL
      side: HOME/AWAY or OVER/UNDER
      line: spread or total line if applicable
      price: best-effort american price from row fields (if present)
    """
    primary = str(row.get("primary_recommendation", "") or "")
    home_spread = _safe_float(row.get("home_spread"))
    spread_price = _safe_float(row.get("spread_price"))
    total_points = _safe_float(row.get("total_points"))
    over_price = _safe_float(row.get("total_over_price"))
    under_price = _safe_float(row.get("total_under_price"))
    home_ml = _safe_float(row.get("home_ml"))
    away_ml = _safe_float(row.get("away_ml"))

    if primary.startswith("Model PICK TOTAL:"):
        market = "TOTAL"
        if "OVER" in primary:
            return (market, "OVER", total_points if not np.isnan(total_points) else None, over_price if not np.isnan(over_price) else None)
        if "UNDER" in primary:
            return (market, "UNDER", total_points if not np.isnan(total_points) else None, under_price if not np.isnan(under_price) else None)
        return (market, "NONE", total_points if not np.isnan(total_points) else None, None)

    if primary.startswith("Model PICK ATS:"):
        market = "ATS"
        if "HOME" in primary:
            return (market, "HOME", home_spread if not np.isnan(home_spread) else None, spread_price if not np.isnan(spread_price) else None)
        if "AWAY" in primary:
            # away line is -home_spread, price usually same; we keep home_spread as stored
            return (market, "AWAY", home_spread if not np.isnan(home_spread) else None, spread_price if not np.isnan(spread_price) else None)
        return (market, "NONE", home_spread if not np.isnan(home_spread) else None, spread_price if not np.isnan(spread_price) else None)

    # default to ML
    market = "ML"
    if "HOME" in primary:
        return (market, "HOME", None, home_ml if not np.isnan(home_ml) else None)
    if "AWAY" in primary:
        return (market, "AWAY", None, away_ml if not np.isnan(away_ml) else None)

    # If primary is a lean/pick text without HOME/AWAY keywords, fall back to ML rec
    mlr = str(row.get("ml_recommendation", "") or "")
    if "HOME" in mlr:
        return (market, "HOME", None, home_ml if not np.isnan(home_ml) else None)
    if "AWAY" in mlr:
        return (market, "AWAY", None, away_ml if not np.isnan(away_ml) else None)

    return (market, "NONE", None, None)


def log_open_bets(
    preds_df: pd.DataFrame,
    *,
    clv_log_path: str = "results/clv_log.csv",
) -> pd.DataFrame:
    """
    Writes/append an "open" snapshot of the bets you intend to place.
    You should call this AFTER you compute primary_recommendation + play_pass.
    """
    if preds_df is None or preds_df.empty:
        return preds_df

    os.makedirs(os.path.dirname(clv_log_path) or ".", exist_ok=True)

    rows = []
    for _, r in preds_df.iterrows():
        if str(r.get("play_pass", "PASS")) != "PLAY":
            continue

        sport = str(r.get("sport", "") or "")
        date_str = str(r.get("date", "") or "")
        home = str(r.get("home", "") or "")
        away = str(r.get("away", "") or "")

        market, side, line, price = infer_market_and_side(r)
        bet_id = make_bet_id(sport=sport, date_str=date_str, home=home, away=away, market=market, side=side, line=line)

        rows.append(
            {
                "ts": _utc_now_iso(),
                "stage": "open",
                "bet_id": bet_id,
                "sport": sport,
                "date": date_str,
                "home": home,
                "away": away,
                "market": market,
                "side": side,
                "line": np.nan if line is None else float(line),
                "price": np.nan if price is None else float(price),
                # store model context too:
                "model_home_prob": _safe_float(r.get("model_home_prob")),
                "edge_home": _safe_float(r.get("edge_home")),
                "pick_score": _safe_float(r.get("pick_score")),
            }
        )

    if not rows:
        return preds_df

    new_df = pd.DataFrame(rows)

    # append with de-dupe on bet_id+stage+date
    if os.path.exists(clv_log_path):
        try:
            old = pd.read_csv(clv_log_path)
            all_df = pd.concat([old, new_df], ignore_index=True)
            all_df = all_df.drop_duplicates(subset=["bet_id", "stage"], keep="last")
        except Exception:
            all_df = new_df
    else:
        all_df = new_df

    all_df.to_csv(clv_log_path, index=False)
    return preds_df


def summarize_clv(
    clv_log_path: str = "results/clv_log.csv",
) -> pd.DataFrame:
    """
    Summary by sport/market based on open vs close prices (if you log close later).
    This does NOT require results/settlement to be useful.
    """
    if not os.path.exists(clv_log_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(clv_log_path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # separate open/close and merge
    open_df = df[df["stage"] == "open"].copy()
    close_df = df[df["stage"] == "close"].copy()

    if close_df.empty:
        # still provide counts
        g = open_df.groupby(["sport", "market"]).size().reset_index(name="open_bets")
        return g.sort_values(["sport", "market"])

    m = open_df.merge(
        close_df[["bet_id", "price"]].rename(columns={"price": "close_price"}),
        on="bet_id",
        how="left",
    )
    m["open_price"] = m["price"]
    m["open_prob"] = m["open_price"].apply(lambda x: _american_to_prob(x) if not pd.isna(x) else np.nan)
    m["close_prob"] = m["close_price"].apply(lambda x: _american_to_prob(x) if not pd.isna(x) else np.nan)

    # CLV as probability improvement (positive is good)
    m["clv_prob"] = m["close_prob"] - m["open_prob"]

    g = (
        m.groupby(["sport", "market"])
        .agg(
            open_bets=("bet_id", "count"),
            close_observed=("close_price", lambda s: int(np.sum(~pd.isna(s)))),
            avg_clv_prob=("clv_prob", "mean"),
            med_clv_prob=("clv_prob", "median"),
        )
        .reset_index()
        .sort_values(["sport", "market"])
    )
    return g
