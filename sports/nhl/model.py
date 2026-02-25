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
class NhlModelConfig:
    goalie_cap: float = 0.06
    goalie_shrink: float = 0.5
    uncertainty_penalty_missing_goalies: float = 0.015


class NhlModel:
    def __init__(self, elo: EloEngine, config: NhlModelConfig | None = None) -> None:
        self.elo = elo
        self.config = config or NhlModelConfig()

    def predict_home_win_prob(self, home: str, away: str, goalie_delta: float | None = None) -> tuple[float, list[str]]:
        flags: list[str] = []
        base = self.elo.expected_home_win_prob(home, away)
        if goalie_delta is None:
            flags.append("goalie_data_missing_uncertainty_penalty")
            return bounded_probability(base - self.config.uncertainty_penalty_missing_goalies), flags

        shrunk = goalie_delta * self.config.goalie_shrink
        capped = max(-self.config.goalie_cap, min(self.config.goalie_cap, shrunk))
        return bounded_probability(base + capped), flags


def _coerce_dates(dates: Any) -> list[dt.date]:
    if dates is None:
        return [dt.date.today()]
    if isinstance(dates, dt.date):
        return [dates]

    parsed: list[dt.date] = []
    for item in dates:
        if isinstance(item, dt.date):
            parsed.append(item)
        elif isinstance(item, str):
            parsed.append(dt.date.fromisoformat(item[:10]))
        else:
            raise TypeError(f"Unsupported date value: {item!r}")
    return parsed or [dt.date.today()]


def _goalie_delta(goalies: dict[str, Any] | None, event_date: dt.date, home: str, away: str) -> float | None:
    if not goalies:
        return None
    by_date = goalies.get(event_date.isoformat(), {})
    if not isinstance(by_date, dict):
        return None
    home_data = by_date.get(home, {})
    away_data = by_date.get(away, {})
    if not isinstance(home_data, dict) or not isinstance(away_data, dict):
        return None
    home_rating = home_data.get("goalie_rating")
    away_rating = away_data.get("goalie_rating")
    if home_rating is None or away_rating is None:
        return None
    try:
        return float(home_rating) - float(away_rating)
    except (TypeError, ValueError):
        return None


def _apply_calibration(sport: str, market: str, probs: Iterable[float], results_dir: str) -> list[float]:
    calibrator = load_calibrator(Path(results_dir), sport=sport, market=market)
    p_list = [float(p) for p in probs]
    if calibrator is None or calibrator.model is None:
        return [bounded_probability(p) for p in p_list]
    try:
        calibrated = calibrator.predict(p_list)
        return [bounded_probability(float(p)) for p in calibrated]
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to apply calibrator for %s/%s: %s", sport, market, exc)
        return [bounded_probability(p) for p in p_list]


def run_nhl(
    dates,
    odds_api_key: str,
    injuries=None,
    goalies=None,
    debug=False,
    raw_dir="data/raw",
    results_dir="data/results",
) -> list[dict]:
    del injuries  # signature compatibility; NHL model currently uses goalie inputs

    target_dates = _coerce_dates(dates)
    if not target_dates:
        return []

    if debug:
        LOGGER.setLevel(logging.DEBUG)

    start_date = min(target_dates)
    days = (max(target_dates) - start_date).days + 1

    raw_events = fetch_odds(api_key=odds_api_key, sport="nhl", date=start_date, days=days)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    raw_out = Path(raw_dir) / f"odds_nhl_{start_date.strftime('%Y%m%d')}_{days}d.json"
    raw_out.write_text(json.dumps(raw_events, indent=2), encoding="utf-8")

    normalized = normalize_odds(raw_events, sport="nhl")
    if not normalized:
        return []

    model = NhlModel(EloEngine())
    bet_cfg = BettingConfig()

    target_iso = {d.isoformat() for d in target_dates}
    base_probs: list[float] = []
    candidates: list[dict[str, Any]] = []

    for event in normalized:
        event_day = event.commence_time[:10]
        if event_day not in target_iso:
            continue

        event_date = dt.date.fromisoformat(event_day)
        goalie_delta = _goalie_delta(goalies, event_date, event.home_team, event.away_team)
        home_prob, model_flags = model.predict_home_win_prob(event.home_team, event.away_team, goalie_delta)

        h2h_offers = [m for m in event.markets if m.market_type == "h2h"]
        if not h2h_offers:
            continue

        market_home_probs = []
        for offer in h2h_offers:
            no_vig_home, _ = remove_vig_two_way(offer.home_price_decimal, offer.away_price_decimal)
            market_home_probs.append(no_vig_home)

        best_offer = max(h2h_offers, key=lambda o: o.home_price_decimal)
        offer_prob = market_probability(best_offer.home_price_decimal)
        consensus_prob = safe_mean(market_home_probs, default=offer_prob)

        base_probs.append(home_prob)
        candidates.append(
            {
                "date": event_day,
                "sport": "nhl",
                "home": event.home_team,
                "away": event.away_team,
                "market_type": "h2h",
                "book": best_offer.book,
                "odds": round(best_offer.home_price_decimal, 4),
                "market_prob": round(consensus_prob, 6),
                "market_prob_offer": round(offer_prob, 6),
                "_flags": model_flags,
            }
        )

    if not candidates:
        return []

    calibrated_probs = _apply_calibration("nhl", "h2h", base_probs, results_dir=results_dir)

    rows: list[dict] = []
    for candidate, calibrated_prob in zip(candidates, calibrated_probs):
        market_prob_value = float(candidate["market_prob_offer"])
        edge_value = edge(calibrated_prob, market_prob_value)
        ev_value = expected_value_two_way(calibrated_prob, float(candidate["odds"]))
        units, decision = size_bet_units(calibrated_prob, float(candidate["odds"]), edge_value, bet_cfg)

        rows.append(
            {
                "date": candidate["date"],
                "sport": "nhl",
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
                "inputs_used": "odds_api+elo+goalies",
                "model_version": "nhl.v1",
            }
        )

    return rows
