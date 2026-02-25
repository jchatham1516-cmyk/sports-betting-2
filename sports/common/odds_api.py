from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import requests

from sports.common.normalization import american_to_decimal, canonicalize_team_name

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.the-odds-api.com/v4/sports"
SPORT_KEYS = {
    "nba": "basketball_nba",
    "nhl": "icehockey_nhl",
    "nfl": "americanfootball_nfl",
}


@dataclass
class MarketOffer:
    market_type: str
    book: str
    home_price_decimal: float
    away_price_decimal: float


@dataclass
class NormalizedEvent:
    event_id: str
    commence_time: str
    sport: str
    home_team: str
    away_team: str
    markets: list[MarketOffer]


def fetch_odds(
    api_key: str,
    sport: str,
    date: dt.date,
    days: int,
    markets: str = "h2h,spreads,totals",
    regions: str = "us",
    odds_format: str = "american",
) -> list[dict]:
    sport_key = SPORT_KEYS[sport]
    start = dt.datetime.combine(date, dt.time.min).isoformat() + "Z"
    end = (dt.datetime.combine(date, dt.time.min) + dt.timedelta(days=days)).isoformat() + "Z"
    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
        "commenceTimeFrom": start,
        "commenceTimeTo": end,
    }
    LOGGER.debug("Fetching odds with params=%s", params)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_odds(raw_events: list[dict], sport: str) -> list[NormalizedEvent]:
    normalized: list[NormalizedEvent] = []
    for event in raw_events:
        home = canonicalize_team_name(event["home_team"])
        away = canonicalize_team_name(event["away_team"])
        offers: list[MarketOffer] = []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue
                prices = {canonicalize_team_name(o["name"]): o.get("price") for o in outcomes}
                if home not in prices or away not in prices:
                    continue
                try:
                    home_dec = american_to_decimal(float(prices[home]))
                    away_dec = american_to_decimal(float(prices[away]))
                except (TypeError, ValueError):
                    continue
                offers.append(
                    MarketOffer(
                        market_type=market["key"],
                        book=bookmaker["key"],
                        home_price_decimal=home_dec,
                        away_price_decimal=away_dec,
                    )
                )
        if offers:
            normalized.append(
                NormalizedEvent(
                    event_id=event["id"],
                    commence_time=event["commence_time"],
                    sport=sport,
                    home_team=home,
                    away_team=away,
                    markets=offers,
                )
            )
    return normalized
