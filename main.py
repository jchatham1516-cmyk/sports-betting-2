# main.py
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

LOG = logging.getLogger("sports-betting")


# ----------------------------
# Logging / FS
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


# ----------------------------
# Parsing helpers
# ----------------------------
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
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

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
# Dynamic runner detection
# ----------------------------
def pick_runner(module: Any, sport: str) -> Callable[..., Any]:
    """
    Finds the "runner" function inside sports.<sport>.model regardless of what Codex named it.
    """
    candidates = [
        f"run_{sport}",
        f"run_daily_{sport}",
        f"run_{sport}_model",
        "run_daily",
        "run_model",
        "run",
        "main",
    ]

    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            LOG.info("Using runner: %s.%s()", module.__name__, name)
            return fn

    # fallback: pick first callable that starts with "run"
    run_like = []
    for name, obj in vars(module).items():
        if callable(obj) and name.startswith("run"):
            run_like.append((name, obj))
    if run_like:
        name, fn = run_like[0]
        LOG.info("Using runner (fallback): %s.%s()", module.__name__, name)
        return fn

    raise RuntimeError(
        f"Could not find a runner function in {module.__name__}. "
        f"Expected one of: {', '.join(candidates)}"
    )


def call_with_supported_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """
    Calls fn with only the kwargs it accepts (prevents signature mismatch crashes).
    """
    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v

    # If fn has **kwargs it will accept anything; we can pass everything
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        accepted = kwargs

    LOG.debug("Calling %s with kwargs: %s", getattr(fn, "__name__", str(fn)), sorted(accepted.keys()))
    return fn(**accepted)


# ----------------------------
# Main
# ----------------------------
def run() -> int:
    parser = argparse.ArgumentParser(description="Daily sports betting picks runner")
    parser.add_argument("--sport", required=True, choices=["nba", "nhl", "nfl"])
    parser.add_argument("--date", required=False, default=None, help="YYYY-MM-DD (or YYYY/MM/DD)")
    parser.add_argument("--days", required=False, default="1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(args.debug)
    ensure_dirs()

    odds_api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not odds_api_key:
        LOG.error("ODDS_API_KEY is not set. Add it as a GitHub Actions secret or export it.")
        return 2

    # Date parsing
    if args.date:
        try:
            target_date = parse_date(args.date)
        except Exception as e:
            LOG.error(str(e))
            return 2
    else:
        target_date = dt.date.today()

    try:
        days = int(args.days)
    except ValueError:
        LOG.error("Invalid --days '%s' (must be an integer)", args.days)
        return 2

    dates = daterange(target_date, days)
    LOG.info("Sport=%s Days=%s StartDate=%s", args.sport, days, target_date.isoformat())

    # Optional inputs
    injuries = load_optional_json("data/inputs/injuries.json")
    goalies = load_optional_json("data/inputs/goalies.json")
    weather = load_optional_json("data/inputs/weather.json")

    # Import sport module + find runner
    module_name = f"sports.{args.sport}.model"
    try:
        model_module = importlib.import_module(module_name)
    except Exception as e:
        LOG.exception("Failed to import %s: %s", module_name, e)
        return 3

    try:
        runner = pick_runner(model_module, args.sport)
    except Exception as e:
        LOG.error(str(e))
        return 3

    # Prepare kwargs; runner will only receive what it supports
    call_kwargs: Dict[str, Any] = {
        "dates": dates,
        "date": target_date,
        "days": days,
        "sport": args.sport,
        "odds_api_key": odds_api_key,
        "api_key": odds_api_key,
        "injuries": injuries,
        "goalies": goalies,
        "weather": weather,
        "debug": args.debug,
        "raw_dir": "data/raw",
        "results_dir": "data/results",
    }

    try:
        result = call_with_supported_kwargs(runner, call_kwargs)
    except Exception as e:
        LOG.exception("Runner crashed: %s", e)
        return 1

    # Normalize result into rows
    rows: List[Dict[str, Any]]
    if result is None:
        rows = []
    elif isinstance(result, list):
        rows = result  # expected: list[dict]
    else:
        # Some implementations might return dict with "rows" or a pandas DataFrame-like
        if isinstance(result, dict) and "rows" in result and isinstance(result["rows"], list):
            rows = result["rows"]
        else:
            LOG.warning("Unexpected runner return type: %s. Writing empty output.", type(result))
            rows = []

    # Ensure sport/date fields exist
    for r in rows:
        r.setdefault("sport", args.sport)
        r.setdefault("date", target_date.isoformat())

    stamp = target_date.strftime("%Y%m%d")
    out_path = Path(f"data/results/picks_{args.sport}_{stamp}.csv")
    write_csv(rows, out_path)

    LOG.info("Wrote %d rows to %s", len(rows), out_path.as_posix())

    # quick preview
    for r in rows[:10]:
        LOG.info(
            "Pick: %s @ %s | %s | edge=%s | %s | units=%s",
            r.get("away", ""),
            r.get("home", ""),
            r.get("market_type", ""),
            r.get("edge", ""),
            r.get("play_pass", ""),
            r.get("bet_units", ""),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
