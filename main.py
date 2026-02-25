# main.py
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG = logging.getLogger("sports-betting")


# ----------------------------
# Helpers
# ----------------------------
def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dirs() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/results").mkdir(parents=True, exist_ok=True)
    Path("data/inputs").mkdir(parents=True, exist_ok=True)


def parse_date(s: str) -> dt.date:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid --date '{s}'. Use YYYY-MM-DD (or YYYY/MM/DD).")


def daterange(start: dt.date, days: int) -> List[dt.date]:
    days = max(int(days), 1)
    return [start + dt.timedelta(days=i) for i in range(days)]


def load_optional_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        LOG.warning("Optional input missing: %s (continuing without it)", path)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        LOG.warning("Failed reading %s: %s (continuing without it)", path, e)
        return None


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    # Minimal CSV writer (no pandas required)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    # Stable header order: union of keys, sorted but with common columns up front
    preferred = [
        "date",
        "sport",
        "home",
        "away",
        "market_type",
        "model_prob",
        "market_prob",
        "edge",
        "odds",
        "ev",
        "bet_units",
        "play_pass",
        "decision_reason",
        "flags",
        "inputs_used",
        "model_version",
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    extras = [k for k in sorted(keys) if k not in preferred]
    header = [k for k in preferred if k in keys] + extras

    def esc(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        if any(ch in s for ch in [",", "\n", '"']):
            s = '"' + s.replace('"', '""') + '"'
        return s

    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(esc(r.get(k)) for k in header))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------
# Main runner
# ----------------------------
def run() -> int:
    parser = argparse.ArgumentParser(description="Daily sports betting picks runner")
    parser.add_argument("--sport", required=True, choices=["nba", "nhl", "nfl"], help="Sport to run")
    parser.add_argument("--date", required=False, default=None, help="Target date YYYY-MM-DD (or YYYY/MM/DD)")
    parser.add_argument("--days", required=False, default="1", help="How many days forward to scan (default 1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    setup_logging(args.debug)
    ensure_dirs()

    odds_api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not odds_api_key:
        LOG.error("ODDS_API_KEY is not set. Add it as a GitHub Actions secret or export it locally.")
        return 2

    # Date handling
    if args.date:
        target_date = parse_date(args.date)
    else:
        target_date = dt.date.today()

    try:
        days = int(args.days)
    except ValueError:
        LOG.error("Invalid --days '%s' (must be an integer)", args.days)
        return 2

    dates = daterange(target_date, days)
    LOG.info("Sport=%s Days=%s StartDate=%s", args.sport, days, target_date.isoformat())

    # Optional local inputs
    injuries = load_optional_json("data/inputs/injuries.json")
    goalies = load_optional_json("data/inputs/goalies.json")
    weather = load_optional_json("data/inputs/weather.json")

    # Dispatch to sport model
    rows: List[Dict[str, Any]] = []

    try:
        if args.sport == "nba":
            from sports.nba.model import run_nba  # type: ignore

            rows = run_nba(
                dates=dates,
                odds_api_key=odds_api_key,
                injuries=injuries,
                weather=weather,
                debug=args.debug,
            )

        elif args.sport == "nhl":
            from sports.nhl.model import run_nhl  # type: ignore

            rows = run_nhl(
                dates=dates,
                odds_api_key=odds_api_key,
                injuries=injuries,
                goalies=goalies,
                debug=args.debug,
            )

        elif args.sport == "nfl":
            from sports.nfl.model import run_nfl  # type: ignore

            rows = run_nfl(
                dates=dates,
                odds_api_key=odds_api_key,
                injuries=injuries,
                weather=weather,
                debug=args.debug,
            )

        else:
            LOG.error("Unsupported sport: %s", args.sport)
            return 2

    except ModuleNotFoundError as e:
        LOG.error("Missing module: %s", e)
        LOG.error("Make sure you have sports/%s/model.py created.", args.sport)
        return 3
    except Exception as e:
        LOG.exception("Run failed: %s", e)
        return 1

    # Output
    stamp = target_date.strftime("%Y%m%d")
    out_path = Path(f"data/results/picks_{args.sport}_{stamp}.csv")
    write_csv(rows, out_path)

    LOG.info("Wrote %d rows to %s", len(rows), out_path.as_posix())

    # Print a quick console preview (first 10)
    if rows:
        preview = rows[:10]
        LOG.info("Preview (first %d):", len(preview))
        for r in preview:
            home = r.get("home", "")
            away = r.get("away", "")
            market_type = r.get("market_type", "")
            edge = r.get("edge", "")
            play_pass = r.get("play_pass", "")
            bet_units = r.get("bet_units", "")
            LOG.info("  %s @ %s | %s | edge=%s | %s | units=%s", away, home, market_type, edge, play_pass, bet_units)

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
