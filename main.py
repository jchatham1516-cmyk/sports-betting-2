from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sports.common.betting import BettingConfig, size_bet_units
from sports.common.calibration import load_calibrator
from sports.common.elo import EloEngine
from sports.common.ev import edge, expected_value_two_way
from sports.common.io import DATA_DIR, read_json, write_csv, write_json
from sports.common.normalization import bounded_probability, remove_vig_two_way
from sports.common.odds_api import fetch_odds, normalize_odds
from sports.nba.model import NbaModel
from sports.nfl.model import NflModel
from sports.nhl.model import NhlModel

MODEL_VERSION = "v1.0.0"
LOGGER = logging.getLogger("sports-betting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sports betting EV engine")
    parser.add_argument("--sport", required=True, choices=["nba", "nhl", "nfl"])
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def run() -> int:
    args = parse_args()
    setup_logging(args.debug)
    load_dotenv()

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is required")

    def parse_date(s: str) -> dt.date:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Invalid --date '{s}'. Use YYYY-MM-DD (or YYYY/MM/DD).")

target_date = parse_date(args.date)
    raw = fetch_odds(api_key=api_key, sport=args.sport, date=target_date, days=args.days)

    raw_path = DATA_DIR / "raw" / f"{args.sport}_{target_date.strftime('%Y%m%d')}.json"
    write_json(raw_path, raw)
    events = normalize_odds(raw, args.sport)

    elo = EloEngine()
    calibrator = load_calibrator(DATA_DIR / "models", args.sport, "h2h")

    injuries = read_json(DATA_DIR / "inputs" / "injuries.json", default={}) or {}
    goalies = read_json(DATA_DIR / "inputs" / "goalies.json", default={}) or {}
    weather = read_json(DATA_DIR / "inputs" / "weather.json", default={}) or {}

    nba_model = NbaModel(elo)
    nhl_model = NhlModel(elo)
    nfl_model = NflModel(elo)
    betting_config = BettingConfig()

    picks: list[dict] = []

    for event in events:
        for market in event.markets:
            model_prob_home = 0.5
            flags: list[str] = []
            inputs_used: list[str] = ["elo"]

            if args.sport == "nba":
                rest = injuries.get("nba_rest_days", {})
                model_prob_home = nba_model.predict_home_win_prob(
                    event.home_team,
                    event.away_team,
                    rest_days_home=float(rest.get(event.home_team, 1.0)),
                    rest_days_away=float(rest.get(event.away_team, 1.0)),
                )
                inputs_used.append("rest")
            elif args.sport == "nhl":
                goalie_delta = goalies.get(event.event_id, {}).get("home_goalie_delta")
                model_prob_home, nhl_flags = nhl_model.predict_home_win_prob(event.home_team, event.away_team, goalie_delta)
                flags.extend(nhl_flags)
                if goalie_delta is not None:
                    inputs_used.append("goalies")
            elif args.sport == "nfl":
                inj = injuries.get(event.event_id, {})
                weather_delta = weather.get(event.event_id, {}).get("home_weather_delta")
                model_prob_home, nfl_flags = nfl_model.predict_home_win_prob(
                    event.home_team,
                    event.away_team,
                    injury_delta=inj.get("home_injury_delta"),
                    qb_delta=inj.get("home_qb_delta"),
                    weather_delta=weather_delta,
                )
                flags.extend(nfl_flags)
                if inj:
                    inputs_used.append("injuries")
                if weather_delta is not None:
                    inputs_used.append("weather")

            if calibrator:
                model_prob_home = float(calibrator.predict([model_prob_home])[0])
                inputs_used.append("calibration")
            model_prob_home = bounded_probability(model_prob_home)

            mkt_home, mkt_away = remove_vig_two_way(market.home_price_decimal, market.away_price_decimal)

            side = "home" if (model_prob_home - mkt_home) >= ((1.0 - model_prob_home) - mkt_away) else "away"
            model_prob = model_prob_home if side == "home" else 1.0 - model_prob_home
            market_prob = mkt_home if side == "home" else mkt_away
            odds = market.home_price_decimal if side == "home" else market.away_price_decimal

            model_edge = edge(model_prob, market_prob)
            ev = expected_value_two_way(model_prob, odds)
            bet_units, reason = size_bet_units(model_prob, odds, model_edge, betting_config)

            picks.append(
                {
                    "date": target_date.isoformat(),
                    "sport": args.sport,
                    "home": event.home_team,
                    "away": event.away_team,
                    "market_type": market.market_type,
                    "model_prob": round(model_prob, 4),
                    "market_prob": round(market_prob, 4),
                    "edge": round(model_edge, 4),
                    "odds": round(odds, 4),
                    "ev": round(ev, 4),
                    "bet_units": bet_units,
                    "play_pass": "play" if reason == "play" else "pass",
                    "decision_reason": reason,
                    "flags": "|".join(flags),
                    "inputs_used": "|".join(sorted(set(inputs_used))),
                    "model_version": MODEL_VERSION,
                }
            )

    frame = pd.DataFrame(picks)
    out_path = DATA_DIR / "results" / f"picks_{target_date.strftime('%Y%m%d')}.csv"
    write_csv(out_path, frame)
    LOGGER.info("Saved %s picks to %s", len(frame), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
