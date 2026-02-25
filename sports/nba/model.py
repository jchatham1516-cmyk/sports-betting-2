from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sports.common.betting import BettingConfig, size_bet_units
from sports.common.calibration import load_calibrator
from sports.common.elo import EloEngine
from sports.common.ev import edge, expected_value_two_way, market_probability
from sports.common.normalization import bounded_probability, remove_vig_two_way, safe_mean
from sports.common.odds_api import fetch_odds, normalize_odds

LOGGER = logging.getLogger(__name__)


@dataclass
class NbaModelConfig:
    rest_days_cap: float = 3.0
    rest_weight: float = 0.02


class NbaModel:
    def __init__(self, elo: EloEngine, config: NbaModelConfig | None = None) -> None:
        self.elo = elo
        self.config = config or NbaModelConfig()

    def predict_home_win_prob(self, home: str, away: str, rest_days_home: float = 1.0, rest_days_away: float = 1.0) -> float:
        base = self.elo.expected_home_win_prob(home, away)
        rest_diff = max(-self.config.rest_days_cap, min(self.config.rest_days_cap, rest_days_home - rest_days_away))
        adjusted = base + (rest_diff * self.config.rest_weight)
        return bounded_probability(adjusted)


def _coerce_dates(dates: Any) -> list[dt.date]:
    if dates is None:
        return [dt.date.today()]
    if isinstance(dates, dt.date):
        return [dates]

    coerced: list[dt.date] = []
    for item in dates:
        if isinstance(item, dt.date):
            coerced.append(item)
            continue
        if isinstance(item, str):
            coerced.append(dt.date.fromisoformat(item[:10]))
            continue
        raise TypeError(f"Unsupported date value: {item!r}")
    return coerced or [dt.date.today()]


def _event_inputs(injuries: dict[str, Any] | None, event_date: dt.date, home: str, away: str) -> tuple[float, float, list[str]]:
    if not injuries:
        return 1.0, 1.0, ["injury_data_missing"]

    by_date = injuries.get(event_date.isoformat(), {})
    home_data = by_date.get(home, {}) if isinstance(by_date, dict) else {}
    away_data = by_date.get(away, {}) if isinstance(by_date, dict) else {}

    home_rest = float(home_data.get("rest_days", 1.0)) if isinstance(home_data, dict) else 1.0
    away_rest = float(away_data.get("rest_days", 1.0)) if isinstance(away_data, dict) else 1.0

    flags: list[str] = []
    if not isinstance(home_data, dict) or "rest_days" not in home_data:
        flags.append("home_rest_missing")
    if not isinstance(away_data, dict) or "rest_days" not in away_data:
        flags.append("away_rest_missing")

    return home_rest, away_rest, flags


def _apply_calibration(sport: str, market: str, probs: Iterable[float], results_dir: str) -> list[float]:
    calibrator = load_calibrator(Path(results_dir), sport=sport, market=market)
    p_list = [float(p) for p in probs]
    if calibrator is None or calibrator.model is None:
        return [bounded_probability(p) for p in p_list]
    try:
        calibrated = calibrator.predict(p_list)
        return [bounded_probability(float(p)) for p in calibrated]
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to apply calibrator for %s/%s: %s", sport, market, exc)
        return [bounded_probability(p) for p in p_list]


def run_nba(
    dates,
    odds_api_key: str,
    injuries=None,
    weather=None,
    debug: bool = False,
    raw_dir: str = "data/raw",
    results_dir: str = "data/results",
) -> list[dict]:
    del weather  # accepted for parity with runner interface

    target_dates = _coerce_dates(dates)
    if not target_dates:
        return []

    if debug:
        LOGGER.setLevel(logging.DEBUG)

    start_date = min(target_dates)
    days = (max(target_dates) - start_date).days + 1

    raw_events = fetch_odds(api_key=odds_api_key, sport="nba", date=start_date, days=days)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    raw_out = Path(raw_dir) / f"odds_nba_{start_date.strftime('%Y%m%d')}_{days}d.json"
    raw_out.write_text(json.dumps(raw_events, indent=2), encoding="utf-8")

    normalized = normalize_odds(raw_events, sport="nba")
    if not normalized:
        return []

    model = NbaModel(EloEngine())
    bet_cfg = BettingConfig()

    target_iso = {d.isoformat() for d in target_dates}
    base_probs: list[float] = []
    candidates: list[dict[str, Any]] = []

    for event in normalized:
        event_day = event.commence_time[:10]
        if event_day not in target_iso:
            continue

        event_date = dt.date.fromisoformat(event_day)
        home_rest, away_rest, input_flags = _event_inputs(injuries, event_date, event.home_team, event.away_team)
        home_prob = model.predict_home_win_prob(event.home_team, event.away_team, home_rest, away_rest)

        h2h_offers = [m for m in event.markets if m.market_type == "h2h"]
        if not h2h_offers:
            continue

        market_home_probs = []
        for offer in h2h_offers:
            no_vig_home, _ = remove_vig_two_way(offer.home_price_decimal, offer.away_price_decimal)
            market_home_probs.append(no_vig_home)

        best_offer = max(h2h_offers, key=lambda o: o.home_price_decimal)
        model_market_prob = market_probability(best_offer.home_price_decimal)
        market_prob_consensus = safe_mean(market_home_probs, default=model_market_prob)

        base_probs.append(home_prob)
        candidates.append(
            {
                "date": event_day,
                "sport": "nba",
                "home": event.home_team,
                "away": event.away_team,
                "market_type": "h2h",
                "book": best_offer.book,
                "odds": round(best_offer.home_price_decimal, 4),
                "market_prob": round(market_prob_consensus, 6),
                "market_prob_offer": round(model_market_prob, 6),
                "_model_prob_raw": home_prob,
                "_flags": input_flags,
            }
        )

    if not candidates:
        return []

    calibrated_probs = _apply_calibration("nba", "h2h", base_probs, results_dir=results_dir)

    rows: list[dict[str, Any]] = []
    for candidate, calibrated_prob in zip(candidates, calibrated_probs):
        market_prob_value = float(candidate["market_prob_offer"])
        edge_value = edge(calibrated_prob, market_prob_value)
        ev_value = expected_value_two_way(calibrated_prob, float(candidate["odds"]))
        units, decision = size_bet_units(calibrated_prob, float(candidate["odds"]), edge_value, bet_cfg)

        rows.append(
            {
                "date": candidate["date"],
                "sport": "nba",
                "home": candidate["home"],
                "away": candidate["away"],
                "market_type": "h2h",
                "book": candidate["book"],
                "model_prob": round(calibrated_prob, 6),
                "market_prob": round(market_prob_value, 6),
                "edge": round(edge_value, 6),
                "odds": candidate["odds"],
                "ev": round(ev_value, 6),
                "bet_units": units,
                "play_pass": "play" if decision == "play" else "pass",
                "decision_reason": decision,
                "flags": ";".join(candidate["_flags"]),
                "inputs_used": "odds_api+elo+rest_days",
                "model_version": "nba.v1",
            }
        )

    return rows
