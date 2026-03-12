from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sports_betting.models.types import MarketPrediction


def write_outputs(predictions: list[MarketPrediction], output_dir: Path, stamp: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [p.to_dict() for p in predictions]
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"daily_recommendations_{stamp}.csv"
    json_path = output_dir / f"daily_recommendations_{stamp}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def render_console_card(date: str, predictions: list[MarketPrediction], top_n: int = 5) -> str:
    plays = [p for p in predictions if p.decision == "bet"]
    plays.sort(key=lambda x: (x.expected_value * 0.5 + x.edge * 0.3 + x.confidence * 0.2), reverse=True)
    lines = ["=== DAILY BETTING CARD ===", f"Date: {date}", ""]
    for sport in ["NBA", "NFL", "NHL"]:
        lines.append(f"[{sport}]")
        sport_plays = [p for p in plays if p.sport.upper() == sport][:top_n]
        if not sport_plays:
            lines.append("- No qualifying plays")
        for idx, pick in enumerate(sport_plays, start=1):
            lines.append(
                f"{idx}. {pick.side} {pick.line or ''} ({pick.sportsbook_odds}) {pick.game} | "
                f"P={pick.model_probability:.1%} Edge={pick.edge:.1%} EV={pick.expected_value:.3f} "
                f"Conf={pick.confidence:.2f} Units={pick.recommended_units}"
            )
            lines.append(f"   Reason: {'; '.join(pick.explanation)}")
        lines.append("")
    return "\n".join(lines)
