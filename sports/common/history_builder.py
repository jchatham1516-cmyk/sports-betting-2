from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from sports.common.odds_sources import ODDS_API_HOST, SPORT_TO_ODDS_KEY, _select_moneyline_pair
from sports.common.teams import canon_team
from sports.common.util import implied_prob_from_american, remove_vig_two_way

DEFAULT_TIMEOUT = 20
HIST_MAX_REQUESTS = int(os.getenv("ODDS_HIST_BUILDER_MAX_REQUESTS", "2000"))
HIST_MAX_EVENTS_PER_DAY = int(os.getenv("ODDS_HIST_BUILDER_MAX_EVENTS_PER_DAY", "50"))
HIST_MAX_EVENT_ODDS_CALLS = int(os.getenv("ODDS_HIST_BUILDER_MAX_EVENT_ODDS_CALLS", "2000"))
HIST_SLEEP_S = float(os.getenv("ODDS_HIST_BUILDER_SLEEP_S", "0.2"))
HIST_MINUTES_BEFORE_COMMENCE = int(os.getenv("ODDS_HIST_BUILDER_MINUTES_BEFORE", "30"))
HIST_BOOKMAKERS = os.getenv("ODDS_HIST_BUILDER_BOOKMAKERS", "")


@dataclass
class _HistBudget:
    limit: int
    used: int = 0
    hard_stop: bool = False

    def allow_one_more(self) -> bool:
        if self.hard_stop:
            return False
        if self.used >= self.limit:
            self.hard_stop = True
            return False
        self.used += 1
        return True


def _get_api_key() -> Optional[str]:
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _debug_headers(r: requests.Response) -> None:
    try:
        print(f"[hist_builder DEBUG] status: {r.status_code} url: {r.url}")
        print(f"[hist_builder DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
        print(f"[hist_builder DEBUG] used: {r.headers.get('x-requests-used')}")
    except Exception:
        pass


def _get_json(url: str, params: dict, budget: _HistBudget) -> Any:
    if not budget.allow_one_more():
        return None

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            _debug_headers(r)
            if r.status_code == 401:
                print("[hist_builder] WARNING: 401 unauthorized; stopping historical builder calls.")
                budget.hard_stop = True
                return None
            if r.status_code == 429:
                sleep_s = 1.5 + attempt
                print(f"[hist_builder] Rate limited (429); sleeping {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            time.sleep(HIST_SLEEP_S)
            return r.json()
        except Exception as exc:
            if attempt == 2:
                print(f"[hist_builder] WARNING: request failed: {exc}")
                return None
            time.sleep(0.5 + attempt)
    return None


def _extract_events(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    if isinstance(payload, list):
        return payload
    return []


def _extract_bookmakers(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            return data.get("bookmakers") or []
    return []


def parse_season(season: str) -> Tuple[int, int]:
    text = str(season).strip()
    if not text:
        raise ValueError("season string required")
    if "-" in text:
        start, end = text.split("-", 1)
        return int(start), int(end)
    year = int(text)
    return year, year + 1


def season_start_date(sport: str, season: str) -> date:
    start_year, _ = parse_season(season)
    sport_key = str(sport).lower()
    if sport_key == "nfl":
        return date(start_year, 9, 1)
    return date(start_year, 10, 1)


def season_string_for_date(sport: str, target_date: date) -> str:
    sport_key = str(sport).lower()
    year = target_date.year
    if sport_key in {"nba", "nhl"}:
        start_year = year if target_date.month >= 10 else year - 1
    else:
        start_year = year if target_date.month >= 9 else year - 1
    return f"{start_year}-{start_year + 1}"


def build_historical_dataset(
    sport: str,
    season: str,
    *,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    sport_key = str(sport).lower()
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing ODDS_API_KEY for historical builder.")

    odds_key = SPORT_TO_ODDS_KEY.get(sport_key)
    if not odds_key:
        raise ValueError(f"Unsupported sport: {sport}")

    start = season_start_date(sport_key, season)
    today = datetime.now(timezone.utc).date()
    end = min(today, start + timedelta(days=366 * 2))

    budget = _HistBudget(limit=HIST_MAX_REQUESTS)
    odds_calls = 0

    rows: List[Dict[str, Any]] = []

    cur = start
    while cur <= end:
        if budget.hard_stop:
            print("[hist_builder] Hard stop reached; ending early.")
            break

        noon = datetime(cur.year, cur.month, cur.day, 12, 0, 0, tzinfo=timezone.utc)
        events_url = f"{ODDS_API_HOST}/v4/historical/sports/{odds_key}/events"
        events_params = {"apiKey": api_key, "date": _iso_z(noon), "dateFormat": "iso"}
        payload = _get_json(events_url, events_params, budget)
        events = _extract_events(payload)
        if not events:
            cur += timedelta(days=1)
            continue

        odds_list_url = f"{ODDS_API_HOST}/v4/historical/sports/{odds_key}/odds"
        odds_list_params = {
            "apiKey": api_key,
            "date": _iso_z(noon),
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if HIST_BOOKMAKERS:
            odds_list_params["bookmakers"] = HIST_BOOKMAKERS

        odds_list_payload = _get_json(odds_list_url, odds_list_params, budget)
        odds_events = _extract_events(odds_list_payload)
        odds_by_event: Dict[str, Dict[str, Any]] = {}
        odds_by_match: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for odds_ev in odds_events:
            ev_id = odds_ev.get("id")
            home_name = canon_team(odds_ev.get("home_team"))
            away_name = canon_team(odds_ev.get("away_team"))
            if ev_id:
                odds_by_event[str(ev_id)] = odds_ev
            if home_name and away_name:
                odds_by_match[(home_name, away_name)] = odds_ev

        for ev in events[: int(HIST_MAX_EVENTS_PER_DAY)]:
            if budget.hard_stop or odds_calls >= HIST_MAX_EVENT_ODDS_CALLS:
                break

            event_id = ev.get("id")
            home_raw = ev.get("home_team")
            away_raw = ev.get("away_team")
            commence = ev.get("commence_time")
            if not event_id or not home_raw or not away_raw:
                continue

            try:
                commence_dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
            except Exception:
                commence_dt = noon

            scores = ev.get("scores") or []
            home_score = away_score = None
            for sc in scores:
                try:
                    name = canon_team(sc.get("name"))
                    score_val = float(sc.get("score"))
                except Exception:
                    continue
                if name == canon_team(home_raw):
                    home_score = score_val
                elif name == canon_team(away_raw):
                    away_score = score_val

            if home_score is None or away_score is None:
                continue

            odds_dt = commence_dt - timedelta(minutes=int(HIST_MINUTES_BEFORE_COMMENCE))
            odds_url = f"{ODDS_API_HOST}/v4/historical/sports/{odds_key}/events/{event_id}/odds"
            odds_params = {
                "apiKey": api_key,
                "date": _iso_z(odds_dt),
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
            if HIST_BOOKMAKERS:
                odds_params["bookmakers"] = HIST_BOOKMAKERS

            odds_payload = _get_json(odds_url, odds_params, budget)
            if odds_payload is None:
                odds_payload = odds_by_event.get(str(event_id)) or odds_by_match.get(
                    (canon_team(home_raw), canon_team(away_raw))
                )
            else:
                odds_calls += 1

            bookmakers = _extract_bookmakers(odds_payload)
            if not bookmakers and isinstance(odds_payload, dict):
                bookmakers = odds_payload.get("bookmakers") or []
            home_ml, away_ml, _, _ = _select_moneyline_pair(bookmakers, str(home_raw), str(away_raw))

            market_home_prob = float("nan")
            if home_ml is not None and away_ml is not None:
                p_home = implied_prob_from_american(home_ml)
                p_away = implied_prob_from_american(away_ml)
                nv = remove_vig_two_way(p_home, p_away)
                if nv is not None:
                    market_home_prob = float(nv[0])

            home = canon_team(home_raw)
            away = canon_team(away_raw)
            game_date = commence_dt.date().isoformat()

            rows.append(
                {
                    "sport": sport_key,
                    "event_id": event_id,
                    "date": game_date,
                    "home": home,
                    "away": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_win": 1 if home_score > away_score else 0,
                    "home_ml": home_ml,
                    "away_ml": away_ml,
                    "market_home_prob": market_home_prob,
                }
            )

        cur += timedelta(days=1)

    df = pd.DataFrame(rows)
    if output_path is None:
        output_path = os.path.join("data", "historical", f"{sport_key}_{season}.csv")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[hist_builder] Saved {len(df)} rows -> {output_path}")

    return df
