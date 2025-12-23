# sports/common/historical_totals.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

ODDS_API_HOST = "https://api.the-odds-api.com"
DEFAULT_TIMEOUT = 20


def _get_api_key() -> Optional[str]:
    return os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDSAPI_KEY")


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _req_json(url: str, params: dict, retries: int = 3) -> Any:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            # Helpful debug
            try:
                print(f"[hist_totals DEBUG] status: {r.status_code} url: {r.url}")
                print(f"[hist_totals DEBUG] remaining: {r.headers.get('x-requests-remaining')}")
                print(f"[hist_totals DEBUG] used: {r.headers.get('x-requests-used')}")
                print(f"[hist_totals DEBUG] last: {r.headers.get('x-requests-last')}")
            except Exception:
                pass

            if r.status_code == 401:
                print("[hist_totals] WARNING: 401 unauthorized. Check ODDS_API_KEY / plan.")
                return None

            if r.status_code == 429:
                sleep_s = 2 + 2 * attempt
                print(f"[hist_totals] 429 rate limit; sleeping {sleep_s}s...")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"[hist_totals] WARNING: request failed: {type(e).__name__}: {e}")
                return None
            time.sleep(1 + attempt)
    return None


def fetch_historical_events(
    *,
    sport_key: str,
    snapshot_dt: datetime,
) -> List[Dict[str, Any]]:
    """
    Uses:
      GET /v4/historical/sports/{sport}/events?apiKey=...&date=...
    Docs: historical events + date snapshot. 3
    """
    api_key = _get_api_key()
    if not api_key:
        print("[hist_totals] WARNING: ODDS_API_KEY missing.")
        return []

    url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events"
    params = {"apiKey": api_key, "date": _iso(snapshot_dt)}
    data = _req_json(url, params=params)
    if not isinstance(data, dict):
        return []
    events = data.get("data")
    if not isinstance(events, list):
        return []
    return events


def fetch_historical_event_totals(
    *,
    sport_key: str,
    event_id: str,
    snapshot_dt: datetime,
    regions: str = "us",
    odds_format: str = "american",
    date_format: str = "iso",
) -> Optional[float]:
    """
    Uses:
      GET /v4/historical/sports/{sport}/events/{eventId}/odds?...&markets=totals&date=...
    Docs: historical event odds + markets + date snapshot. 4

    Returns: the consensus total points (float) if found, else None.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    url = f"{ODDS_API_HOST}/v4/historical/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "totals",
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "date": _iso(snapshot_dt),
    }
    data = _req_json(url, params=params)
    if not isinstance(data, dict):
        return None

    # historical event odds response shape is typically:
    # { "data": { "id":..., "bookmakers":[...] }, "timestamp":... }
    ev = data.get("data")
    if not isinstance(ev, dict):
        return None

    books = ev.get("bookmakers") or []
    if not isinstance(books, list) or not books:
        return None

    totals: List[float] = []
    for b in books:
        markets = b.get("markets") or []
        for m in markets:
            if m.get("key") != "totals":
                continue
            outcomes = m.get("outcomes") or []
            for o in outcomes:
                # outcomes include Over/Under with a "point" field
                pt = o.get("point")
                try:
                    if pt is not None:
                        totals.append(float(pt))
                except Exception:
                    continue

    if not totals:
        return None

    # Simple consensus: average across all books/outcomes points gathered
    return float(sum(totals) / len(totals))


def build_team_historical_total_lines(
    *,
    sport_key: str,
    days_back: int = 14,
    minutes_before_commence: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Builds per-team list of historical market total lines (pregame).
    This is NOT actual scored totals; it’s the sportsbook totals line history.

    Returns:
      { team_name_raw_from_api: {"lines": [float,...], "games": int, "avg": float, "sd": float} }
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=int(days_back))

    per_team: Dict[str, List[float]] = {}

    # One snapshot per day (cheap) to get events list, then per-event odds (more expensive).
    d = start.date()
    while d <= now.date():
        # Snapshot near end of day UTC so it “knows about” that day’s events.
        snap = datetime(d.year, d.month, d.day, 23, 0, 0, tzinfo=timezone.utc)
        events = fetch_historical_events(sport_key=sport_key, snapshot_dt=snap)

        for ev in events:
            ev_id = ev.get("id")
            home = ev.get("home_team")
            away = ev.get("away_team")
            ct = _parse_iso(ev.get("commence_time") or "")
            if not ev_id or not home or not away or ct is None:
                continue

            # We want a pregame snapshot shortly before tip
            pre = ct - timedelta(minutes=int(minutes_before_commence))
            if pre < start:
                continue
            if pre > now:
                continue

            tot = fetch_historical_event_totals(
                sport_key=sport_key,
                event_id=str(ev_id),
                snapshot_dt=pre,
            )
            if tot is None:
                continue

            per_team.setdefault(home, []).append(float(tot))
            per_team.setdefault(away, []).append(float(tot))

        d = d + timedelta(days=1)

    out: Dict[str, Dict[str, Any]] = {}
    for team, lines in per_team.items():
        if not lines:
            continue
        n = len(lines)
        avg = sum(lines) / n
        # sample sd
        if n >= 2:
            mu = avg
            var = sum((x - mu) ** 2 for x in lines) / (n - 1)
            sd = var ** 0.5
        else:
            sd = float("nan")
        out[team] = {"lines": lines, "games": n, "avg": float(avg), "sd": float(sd)}
    return out
