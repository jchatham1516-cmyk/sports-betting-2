from __future__ import annotations

import pandas as pd


def run_backtest(predictions_path: str) -> dict:
    df = pd.read_csv(predictions_path)
    bets = df[df["decision"] == "bet"].copy()
    if bets.empty:
        return {"bets": 0, "roi": 0.0, "win_rate": 0.0}

    bets["won"] = bets["outcome"].astype(int)
    bets["pnl"] = bets.apply(
        lambda r: r["recommended_units"] * ((r["sportsbook_odds"] / 100) if r["sportsbook_odds"] > 0 else (100 / abs(r["sportsbook_odds"]))) if r["won"] == 1 else -r["recommended_units"],
        axis=1,
    )
    total_risk = bets["recommended_units"].sum()
    roi = bets["pnl"].sum() / total_risk if total_risk else 0.0
    return {
        "bets": int(len(bets)),
        "units_risked": float(total_risk),
        "units_won": float(bets["pnl"].sum()),
        "roi": float(roi),
        "win_rate": float(bets["won"].mean()),
        "avg_edge": float(bets["edge"].mean()),
    }
