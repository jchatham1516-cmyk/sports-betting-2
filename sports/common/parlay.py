from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.bet_config import get_sport_bet_config
from sports.common.bet_rules import implied_prob_american


CONFIDENCE_WEIGHTS = {
    "HIGH": 1.0,
    "MEDIUM": 0.85,
    "LOW": 0.7,
}

RELIABILITY_WEIGHTS = {
    "ML": 1.0,
    "ATS": 0.9,
    "TOTAL": 0.85,
}

MARKET_LIMITS = {
    "TOTAL": 3,
    "ATS": 4,
}

SMART_PARLAY_MARKET_WEIGHTS = {
    "moneyline": 1.00,
    "spread": 0.70,
    "total": 0.60,
}

SMART_PARLAY_CONFIDENCE_WEIGHTS = {
    "HIGH": 1.00,
    "MEDIUM": 0.90,
    "LOW": 0.80,
}

SMART_PARLAY_DEFAULTS = {
    "min_edge_by_sport": {"nba": 0.055, "nhl": 0.050},
    "min_pwin_by_sport": {"nba": 0.58, "nhl": 0.56},
    "min_ev": 0.02,
    "unit_dollars": 10.0,
}

SMART_PARLAY_BLOCKED_FLAGS = {
    "UNCALIBRATED_FALLBACK",
    "ATS_UNCALIBRATED_MARGIN",
    "ATS_GATED_INVALID_SPREAD",
    "TOTAL_GATED_LOW_QUALITY",
    "TOTAL_SANITY_FAIL_PASS",
}

SMART_PARLAY_ALLOWED_MARKETS = {"moneyline", "spread", "total"}

PARLAY_DEFAULT_UNIT_DOLLARS = 10.0


@dataclass
class ParlayLeg:
    sport: str
    game_key: str
    market: str
    side: str
    team: str
    opponent: str
    price: float
    implied_prob: float
    model_prob: float
    edge_prob: float
    parlay_score: float


@dataclass
class DailyParlayLeg:
    date: str
    sport: str
    game_key: str
    market: str
    side: str
    team: str
    opponent: str
    matchup: str
    line: object
    price: float
    implied_prob: float
    model_prob: float
    market_prob: float
    edge_prob: float
    parlay_score: float
    reliability_weight: float


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def normalize_market_name(value: object) -> Optional[str]:
    market = str(value or "").strip().lower()
    if market in {"ml", "moneyline", "money line"}:
        return "moneyline"
    if market in {"spread", "ats", "against the spread"}:
        return "spread"
    if market in {"total", "totals", "over/under", "ou"}:
        return "total"
    return None


def american_to_decimal(price: object) -> float:
    try:
        odds = float(price)
    except Exception:
        return float("nan")
    if np.isnan(odds):
        return float("nan")
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return float("nan")


def _market_from_row(row: pd.Series) -> str:
    primary_market = str(row.get("primary_market", "")).upper()
    if primary_market in {"ML", "ATS", "TOTAL"}:
        return primary_market
    reco = str(row.get("primary_recommendation", "")).upper()
    if "TOTAL" in reco:
        return "TOTAL"
    if "ATS" in reco or "SPREAD" in reco:
        return "ATS"
    if "ML" in reco:
        return "ML"
    return "UNKNOWN"


def _price_for_leg(row: pd.Series, market: str, side: str) -> float:
    side = str(side).upper()
    if market == "ML":
        return _safe_float(row.get("home_ml") if side == "HOME" else row.get("away_ml"))
    if market == "ATS":
        return _safe_float(row.get("spread_price"))
    if market == "TOTAL":
        if side == "OVER":
            return _safe_float(row.get("total_over_price"))
        if side == "UNDER":
            return _safe_float(row.get("total_under_price"))
    return float("nan")


def _game_key(row: pd.Series) -> str:
    event_id = row.get("event_id")
    date = row.get("date")
    home = row.get("home")
    away = row.get("away")
    return f"{event_id or ''}|{date or ''}|{home or ''}|{away or ''}".strip("|")


def _team_for_leg(row: pd.Series, market: str, side: str) -> Tuple[str, str]:
    home = str(row.get("home", ""))
    away = str(row.get("away", ""))
    side = str(side).upper()
    if market == "ML":
        return (home if side == "HOME" else away, away if side == "HOME" else home)
    if market == "ATS":
        return (home if side == "HOME" else away, away if side == "HOME" else home)
    return (home, away)


def _line_for_leg(row: pd.Series, market: str, side: str) -> object:
    market = str(market).upper()
    side = str(side).upper()
    if market == "ATS":
        line = row.get("home_spread")
        if side == "AWAY" and line is not None and pd.notna(line):
            try:
                return -float(line)
            except Exception:
                return line
        return line
    if market == "TOTAL":
        return row.get("total_points")
    return ""


def _confidence_weight(confidence: str) -> float:
    return float(CONFIDENCE_WEIGHTS.get(str(confidence).upper(), 0.75))


def _reliability_weight(market: str) -> float:
    return float(RELIABILITY_WEIGHTS.get(str(market).upper(), 0.8))


def _parlay_score(edge_prob: float, confidence: str, market: str) -> float:
    if not np.isfinite(edge_prob):
        return float("nan")
    return float(edge_prob) * _confidence_weight(confidence) * _reliability_weight(market)


def _daily_parlay_score(p_win: float, edge_prob: float, reliability_weight: float) -> float:
    if not np.isfinite(p_win) or not np.isfinite(edge_prob) or not np.isfinite(reliability_weight):
        return float("nan")
    return float(p_win) ** 1.5 * (1.0 + 2.0 * float(edge_prob)) * float(reliability_weight)


def _filtered_reason_summary(counts: Dict[str, int]) -> List[str]:
    reasons: List[str] = []
    for key, val in counts.items():
        if val > 0:
            reasons.append(f"{key}={val}")
    return reasons


def build_weekly_parlay(
    candidates_df: pd.DataFrame,
    *,
    min_legs: int = 6,
    max_legs: int = 7,
) -> Dict[str, object]:
    if candidates_df is None or candidates_df.empty:
        return {"status": "NO_PARLAY_THIS_WEEK", "reasons": ["no candidates"]}

    work = candidates_df.copy()
    work["play_pass"] = work.get("play_pass", "").astype(str).str.upper()
    work = work[work["play_pass"] == "PLAY"].copy()
    if work.empty:
        return {"status": "NO_PARLAY_THIS_WEEK", "reasons": ["no PLAY candidates"]}

    filtered_counts = {
        "low_edge": 0,
        "uncalibrated": 0,
        "disagreement": 0,
        "missing_model": 0,
        "goalie_unconfirmed": 0,
    }

    legs: List[ParlayLeg] = []
    for _, row in work.iterrows():
        sport = str(row.get("sport", row.get("league", ""))).lower()
        config = get_sport_bet_config(sport)
        edge_prob = _safe_float(row.get("edge_prob_final"))
        if sport == "nhl":
            nhl_min_edge = config.nhl_parlay_min_edge
            if nhl_min_edge is None:
                nhl_min_edge = config.parlay_min_edge
            min_edge = float(nhl_min_edge)
        else:
            min_edge = float(config.parlay_min_edge)
        if not np.isfinite(edge_prob) or edge_prob < float(min_edge):
            filtered_counts["low_edge"] += 1
            continue

        flags = str(row.get("decision_flags") or "")
        if "UNCALIBRATED_FALLBACK" in flags:
            filtered_counts["uncalibrated"] += 1
            continue

        p_model_cal = _safe_float(row.get("p_model_cal"))
        p_market = _safe_float(row.get("p_market"))
        disagreement = abs(float(p_model_cal - p_market)) if np.isfinite(p_model_cal) and np.isfinite(p_market) else 0.0
        if disagreement > float(config.parlay_disagree_cap) and edge_prob < float(config.parlay_disagree_huge_edge):
            filtered_counts["disagreement"] += 1
            continue

        if sport == "nhl":
            p_model_final = _safe_float(row.get("p_model_final"))
            if not np.isfinite(p_model_final):
                filtered_counts["missing_model"] += 1
                continue
            flags = str(row.get("decision_flags") or "")
            if "GOALIE_UNCONFIRMED" in flags and edge_prob < float(config.parlay_disagree_huge_edge):
                filtered_counts["goalie_unconfirmed"] += 1
                continue

        market = _market_from_row(row)
        side = str(row.get("primary_side", "")).upper()
        price = _price_for_leg(row, market, side)
        implied_prob = _safe_float(row.get("p_market"))
        if not np.isfinite(implied_prob):
            implied_prob = implied_prob_american(price)
        model_prob = _safe_float(row.get("p_model_final"))
        team, opponent = _team_for_leg(row, market, side)
        score = _parlay_score(edge_prob, str(row.get("confidence", "")), market)

        legs.append(
            ParlayLeg(
                sport=sport,
                game_key=_game_key(row),
                market=market,
                side=side,
                team=team,
                opponent=opponent,
                price=price,
                implied_prob=implied_prob,
                model_prob=model_prob,
                edge_prob=edge_prob,
                parlay_score=score,
            )
        )

    if not legs or len(legs) < min_legs:
        reasons = _filtered_reason_summary(filtered_counts)
        if not reasons:
            reasons = ["insufficient qualified legs"]
        return {"status": "NO_PARLAY_THIS_WEEK", "reasons": reasons}

    legs_sorted = sorted(legs, key=lambda l: l.parlay_score if np.isfinite(l.parlay_score) else -1, reverse=True)
    selected: List[ParlayLeg] = []
    used_games: set[str] = set()
    used_teams: set[str] = set()
    market_counts: Dict[str, int] = {"ML": 0, "ATS": 0, "TOTAL": 0}

    for leg in legs_sorted:
        if len(selected) >= max_legs:
            break
        if leg.game_key and leg.game_key in used_games:
            continue
        if leg.team and leg.team in used_teams:
            continue
        if leg.opponent and leg.opponent in used_teams:
            continue
        market_limit = MARKET_LIMITS.get(leg.market)
        if market_limit is not None and market_counts.get(leg.market, 0) >= market_limit:
            continue

        selected.append(leg)
        if leg.game_key:
            used_games.add(leg.game_key)
        if leg.team:
            used_teams.add(leg.team)
        if leg.opponent:
            used_teams.add(leg.opponent)
        market_counts[leg.market] = market_counts.get(leg.market, 0) + 1

    if len(selected) < min_legs:
        return {
            "status": "NO_PARLAY_THIS_WEEK",
            "reasons": ["correlation filters reduced legs below minimum"],
        }

    combined_decimal = 1.0
    for leg in selected:
        decimal = american_to_decimal(leg.price)
        if np.isfinite(decimal):
            combined_decimal *= float(decimal)

    combined_implied_prob = 1.0 / combined_decimal if combined_decimal > 0 else float("nan")

    return {
        "status": "PARLAY_READY",
        "requested_legs": int(max_legs),
        "legs": [
            {
                "sport": leg.sport,
                "game_key": leg.game_key,
                "market": leg.market,
                "side": leg.side,
                "team": leg.team,
                "opponent": leg.opponent,
                "price": leg.price,
                "implied_prob": leg.implied_prob,
                "model_prob": leg.model_prob,
                "edge_prob_final": leg.edge_prob,
                "parlay_score": leg.parlay_score,
            }
            for leg in selected
        ],
        "combined_decimal_odds": combined_decimal,
        "combined_implied_prob": combined_implied_prob,
    }


def build_daily_parlay(
    candidates_df: pd.DataFrame,
    *,
    min_legs: int = 4,
    max_legs: int = 7,
    date_tag: str | None = None,
) -> Dict[str, object]:
    if candidates_df is None or candidates_df.empty:
        return {"status": "NO_PARLAY_TODAY", "reasons": ["no candidates"]}

    work = candidates_df.copy()
    work["play_pass"] = work.get("play_pass", "").astype(str).str.upper()
    work = work[work["play_pass"] == "PLAY"].copy()
    if work.empty:
        return {"status": "NO_PARLAY_TODAY", "reasons": ["no PLAY candidates"]}

    filtered_counts = {
        "low_edge": 0,
        "low_pwin": 0,
        "uncalibrated": 0,
        "flagged": 0,
        "disagreement": 0,
        "missing_model": 0,
        "missing_market": 0,
    }

    legs: List[DailyParlayLeg] = []
    for _, row in work.iterrows():
        sport = str(row.get("sport", row.get("league", ""))).lower()
        config = get_sport_bet_config(sport)

        edge_prob = _safe_float(row.get("edge_prob_final"))
        if not np.isfinite(edge_prob) or abs(edge_prob) < float(config.parlay_min_edge):
            filtered_counts["low_edge"] += 1
            continue

        flags = {f for f in str(row.get("decision_flags") or "").split(",") if f}
        if "UNCALIBRATED_FALLBACK" in flags:
            filtered_counts["uncalibrated"] += 1
            continue
        blocked_flags = {"ATS_UNCALIBRATED_MARGIN", "TOTAL_GATED_LOW_QUALITY", "GOALIE_UNCONFIRMED"}
        if flags.intersection(blocked_flags):
            filtered_counts["flagged"] += 1
            continue

        p_model_final = _safe_float(row.get("p_model_final"))
        if not np.isfinite(p_model_final):
            filtered_counts["missing_model"] += 1
            continue

        if p_model_final < float(config.parlay_min_pwin):
            filtered_counts["low_pwin"] += 1
            continue

        p_market_used = _safe_float(row.get("p_market_used"))
        if not np.isfinite(p_market_used):
            filtered_counts["missing_market"] += 1
            continue

        disagreement = abs(float(p_model_final - p_market_used))
        if disagreement > float(config.parlay_max_disagreement) and abs(edge_prob) < float(
            config.parlay_big_edge_override
        ):
            filtered_counts["disagreement"] += 1
            continue

        market = _market_from_row(row)
        side = str(row.get("primary_side", "")).upper()
        price = _price_for_leg(row, market, side)
        implied_prob = _safe_float(row.get("p_market_used"))
        if not np.isfinite(implied_prob):
            implied_prob = implied_prob_american(price)
        team, opponent = _team_for_leg(row, market, side)
        matchup = f"{row.get('away', '')} @ {row.get('home', '')}".strip()
        reliability_weight = float(config.parlay_reliability_weights.get(str(market).upper(), 0.6))
        score = _daily_parlay_score(p_model_final, edge_prob, reliability_weight)
        line = _line_for_leg(row, market, side)
        leg_date = str(row.get("date") or date_tag or "")

        legs.append(
            DailyParlayLeg(
                date=leg_date,
                sport=sport,
                game_key=_game_key(row),
                market=market,
                side=side,
                team=team,
                opponent=opponent,
                matchup=matchup,
                line=line,
                price=price,
                implied_prob=implied_prob,
                model_prob=p_model_final,
                market_prob=p_market_used,
                edge_prob=edge_prob,
                parlay_score=score,
                reliability_weight=reliability_weight,
            )
        )

    if not legs or len(legs) < min_legs:
        reasons = _filtered_reason_summary(filtered_counts)
        if not reasons:
            reasons = ["insufficient qualified legs"]
        return {"status": "NO_PARLAY_TODAY", "reasons": reasons}

    def _sort_key(leg: DailyParlayLeg) -> Tuple[float, str, str, str, str, str]:
        score = leg.parlay_score if np.isfinite(leg.parlay_score) else -1e9
        return (-score, leg.sport, leg.game_key, leg.market, leg.side, leg.team)

    legs_sorted = sorted(legs, key=_sort_key)
    available_sports = {leg.sport for leg in legs_sorted if leg.sport}
    available_markets = {leg.market for leg in legs_sorted if leg.market}
    selected: List[DailyParlayLeg] = []
    used_games: set[str] = set()
    used_teams: set[str] = set()
    sport_counts: Dict[str, int] = {}
    market_counts: Dict[str, int] = {"ML": 0, "ATS": 0, "TOTAL": 0}

    for leg in legs_sorted:
        if len(selected) >= max_legs:
            break
        if leg.game_key and leg.game_key in used_games:
            continue
        if leg.team and leg.team in used_teams:
            continue
        if leg.opponent and leg.opponent in used_teams:
            continue

        new_total = len(selected) + 1
        new_sport_count = sport_counts.get(leg.sport, 0) + 1
        if len(available_sports) > 1 and new_total > 1:
            if new_sport_count / new_total > 0.60:
                continue
        new_market_count = market_counts.get(leg.market, 0) + 1
        if len(available_markets) > 1 and new_total > 1:
            if new_market_count / new_total > 0.50:
                continue

        selected.append(leg)
        if leg.game_key:
            used_games.add(leg.game_key)
        if leg.team:
            used_teams.add(leg.team)
        if leg.opponent:
            used_teams.add(leg.opponent)
        sport_counts[leg.sport] = new_sport_count
        market_counts[leg.market] = new_market_count

    if len(selected) < min_legs:
        return {
            "status": "NO_PARLAY_TODAY",
            "reasons": ["correlation filters reduced legs below minimum"],
        }

    combined_decimal = 1.0
    combined_model_prob = 1.0
    for leg in selected:
        decimal = american_to_decimal(leg.price)
        if not np.isfinite(decimal):
            combined_decimal = float("nan")
        else:
            if np.isfinite(combined_decimal):
                combined_decimal *= float(decimal)
        if np.isfinite(leg.model_prob):
            combined_model_prob *= float(leg.model_prob)
        else:
            combined_model_prob = float("nan")

    combined_implied_prob = 1.0 / combined_decimal if np.isfinite(combined_decimal) and combined_decimal > 0 else float(
        "nan"
    )
    parlay_ev_model = (
        combined_model_prob * combined_decimal - 1.0
        if np.isfinite(combined_model_prob) and np.isfinite(combined_decimal)
        else float("nan")
    )

    num_legs = len(selected)
    base_units = min(0.25, 0.05 * num_legs)
    max_units = float(os.getenv("PARLAY_MAX_UNITS", "0.5"))
    parlay_units = min(base_units, max_units)

    return {
        "status": "PARLAY_READY",
        "requested_legs": int(max_legs),
        "num_legs": num_legs,
        "parlay_units": parlay_units,
        "parlay_ev_model": parlay_ev_model,
        "model_ev_note": "Model-based EV estimate; do not treat as guaranteed outcome.",
        "legs": [
            {
                "date": leg.date,
                "legs": num_legs,
                "sport": leg.sport,
                "game_key": leg.game_key,
                "matchup": leg.matchup,
                "market": leg.market,
                "side": leg.side,
                "team": leg.team,
                "opponent": leg.opponent,
                "line": leg.line,
                "price": leg.price,
                "p_win_final": leg.model_prob,
                "p_market_used": leg.market_prob,
                "edge_prob_final": leg.edge_prob,
                "parlay_score": leg.parlay_score,
                "reliability_weight": leg.reliability_weight,
                "combined_implied_prob": combined_implied_prob,
                "combined_model_prob": combined_model_prob,
                "combined_decimal_odds": combined_decimal,
            }
            for leg in selected
        ],
        "combined_decimal_odds": combined_decimal,
        "combined_implied_prob": combined_implied_prob,
        "combined_model_prob": combined_model_prob,
    }


def render_parlay_card(parlay_result: Dict[str, object]) -> None:
    print("\nPARLAY CARD")
    status = parlay_result.get("status", "UNKNOWN")
    if status != "PARLAY_READY":
        print("NO PARLAY TODAY")
        reasons = parlay_result.get("reasons", [])
        if reasons:
            print(f"[parlay] reasons={reasons}")
        return
    legs = parlay_result.get("legs", [])
    print(f"Legs: {len(legs)} | Units: {parlay_result.get('parlay_units')}")
    for i, leg in enumerate(legs, start=1):
        print(
            f"{i}. {leg.get('sport', '').upper()} {leg.get('matchup')} "
            f"{leg.get('market')} {leg.get('side')} "
            f"line={leg.get('line')} price={leg.get('price')} "
            f"p_win={leg.get('p_win_final')}"
        )
    print(
        "Combined: "
        f"model_prob={parlay_result.get('combined_model_prob')} "
        f"implied_prob={parlay_result.get('combined_implied_prob')} "
        f"decimal_odds={parlay_result.get('combined_decimal_odds')}"
    )


def load_recent_predictions(
    *,
    preds_dir: str = "results",
    days_back: int = 7,
) -> pd.DataFrame:
    preds_dir_path = Path(preds_dir)
    cutoff = datetime.utcnow() - timedelta(days=int(days_back))
    frames: List[pd.DataFrame] = []
    for path in preds_dir_path.glob("predictions_*_*.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        date_col = pd.to_datetime(df.get("date"), errors="coerce")
        df = df.copy()
        df["date"] = date_col
        df = df[df["date"].notna()]
        df = df[df["date"] >= cutoff]
        if df.empty:
            continue
        if "sport" not in df.columns:
            stem = path.stem
            sport = stem.split("_")[1] if len(stem.split("_")) > 2 else ""
            df["sport"] = sport
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_primary_market(value: object) -> str:
    return normalize_market_name(value) or str(value or "").strip().lower()


def _normalize_confidence(value: object) -> str:
    return str(value or "").strip().upper()


def _safe_float_or_nan(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _parlay_game_key(row: pd.Series) -> str:
    for key in ("game_key", "event_id"):
        val = row.get(key)
        if pd.notna(val) and str(val).strip():
            return str(val)
    home = str(row.get("home", "")).strip()
    away = str(row.get("away", "")).strip()
    date = str(row.get("date", "")).strip()
    return f"{date}|{away}@{home}".strip("|")


def _teams_for_row(row: pd.Series) -> Tuple[str, str]:
    home = str(row.get("home", "")).strip()
    away = str(row.get("away", "")).strip()
    return home, away


def _leg_price(row: pd.Series) -> float:
    for col in ("primary_price", "price", "line_price"):
        if col in row.index:
            price = _safe_float_or_nan(row.get(col))
            if np.isfinite(price):
                return float(price)
    return float("nan")


def _leg_line(row: pd.Series, market: str, side: str) -> object:
    for col in ("line", "primary_line"):
        if col in row.index and pd.notna(row.get(col)):
            return row.get(col)
    market = str(market).lower()
    side = str(side).upper()
    if market == "spread":
        line = row.get("home_spread")
        if side == "AWAY" and line is not None and pd.notna(line):
            try:
                return -float(line)
            except Exception:
                return line
        return line
    if market == "total":
        return row.get("total_points")
    return ""


def _leg_score(p_win: float, edge_prob: float, market: str, confidence: str) -> float:
    if not np.isfinite(p_win) or not np.isfinite(edge_prob):
        return float("nan")
    rel_weight = SMART_PARLAY_MARKET_WEIGHTS.get(str(market).lower(), 0.60)
    conf_weight = SMART_PARLAY_CONFIDENCE_WEIGHTS.get(_normalize_confidence(confidence), 0.85)
    return float(p_win) ** 1.6 * (1.0 + 2.0 * max(float(edge_prob), 0.0)) * float(rel_weight) * float(conf_weight)


def _goalie_confirmed_for_row(row: pd.Series) -> bool:
    home_status = str(row.get("goalie_home_status") or "").strip().upper()
    away_status = str(row.get("goalie_away_status") or "").strip().upper()
    if home_status or away_status:
        return home_status == "CONFIRMED" and away_status == "CONFIRMED"
    status = str(row.get("goalie_status") or "").strip().upper()
    if status in {"CONFIRMED", "OK"}:
        return True
    if status:
        return False
    return False


def load_daily_picks(date_str: str, sports_list: Iterable[str]) -> pd.DataFrame:
    date_tag = str(date_str).replace("/", "-")
    frames: List[pd.DataFrame] = []
    for sport in sports_list:
        sport_key = str(sport).strip().lower()
        if not sport_key:
            continue
        path = Path("results") / f"picks_{sport_key}_{date_tag}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "sport" not in df.columns:
            df = df.copy()
            df["sport"] = sport_key
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_smart_parlay(
    picks_df: pd.DataFrame,
    *,
    min_legs: int = 4,
    max_legs: int = 7,
    config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if picks_df is None or picks_df.empty:
        return {"status": "NO_PARLAY", "reason": "no candidates", "legs": []}

    cfg = {**SMART_PARLAY_DEFAULTS, **(config or {})}
    min_edge_by_sport = cfg.get("min_edge_by_sport", SMART_PARLAY_DEFAULTS["min_edge_by_sport"])
    min_pwin_by_sport = cfg.get("min_pwin_by_sport", SMART_PARLAY_DEFAULTS["min_pwin_by_sport"])
    min_ev = float(cfg.get("min_ev", SMART_PARLAY_DEFAULTS["min_ev"]))

    work = picks_df.copy()
    work["play_pass"] = work.get("play_pass", "").astype(str).str.upper()
    work = work[work["play_pass"] == "PLAY"].copy()
    if work.empty:
        return {"status": "NO_PARLAY", "reason": "no PLAY candidates", "legs": []}

    eligible_legs: List[Dict[str, object]] = []
    for _, row in work.iterrows():
        sport = str(row.get("sport", "")).lower()
        market = normalize_market_name(row.get("primary_market"))
        if market not in SMART_PARLAY_ALLOWED_MARKETS:
            continue

        p_model_final = _safe_float_or_nan(row.get("p_model_final"))
        p_market_used = _safe_float_or_nan(row.get("p_market_used"))
        edge_prob = _safe_float_or_nan(row.get("edge_prob_final"))
        primary_ev = _safe_float_or_nan(row.get("primary_ev"))
        if not np.isfinite(p_model_final) or not np.isfinite(p_market_used) or not np.isfinite(edge_prob):
            continue
        if not np.isfinite(primary_ev) or primary_ev < min_ev:
            continue

        abs_edge = _safe_float_or_nan(row.get("abs_edge_prob"))
        if not np.isfinite(abs_edge):
            abs_edge = abs(edge_prob)

        min_edge = float(min_edge_by_sport.get(sport, SMART_PARLAY_DEFAULTS["min_edge_by_sport"].get(sport, 0.055)))
        min_pwin = float(min_pwin_by_sport.get(sport, SMART_PARLAY_DEFAULTS["min_pwin_by_sport"].get(sport, 0.58)))

        flags = {f.strip() for f in str(row.get("decision_flags") or "").split(",") if f.strip()}
        if flags.intersection(SMART_PARLAY_BLOCKED_FLAGS):
            continue

        goalie_confirmed = True
        if sport == "nhl":
            goalie_confirmed = _goalie_confirmed_for_row(row)
            if not goalie_confirmed:
                if abs_edge < 0.065 or p_model_final < 0.60:
                    continue

        if abs_edge < min_edge:
            continue
        if p_model_final < min_pwin:
            continue

        price = _leg_price(row)
        decimal_odds = american_to_decimal(price) if np.isfinite(price) else float("nan")
        leg_score = _leg_score(p_model_final, edge_prob, market, row.get("confidence"))
        game_key = _parlay_game_key(row)
        home, away = _teams_for_row(row)
        eligible_legs.append(
            {
                "sport": sport,
                "game_key": game_key,
                "market": market,
                "side": str(row.get("primary_side", "")).upper(),
                "home": home,
                "away": away,
                "matchup": f"{away} @ {home}".strip(),
                "line": _leg_line(row, market, row.get("primary_side", "")),
                "price": price,
                "decimal_odds": decimal_odds,
                "p_model_final": p_model_final,
                "p_market_used": p_market_used,
                "edge_prob_final": edge_prob,
                "primary_ev": primary_ev,
                "confidence": str(row.get("confidence", "")),
                "value_tier": str(row.get("value_tier", "")),
                "decision_flags": str(row.get("decision_flags", "")),
                "reason_short": str(row.get("reason_short", row.get("decision_reason", ""))),
                "leg_score": leg_score,
                "goalie_confirmed": goalie_confirmed if sport == "nhl" else None,
            }
        )

    if len(eligible_legs) < min_legs:
        return {
            "status": "NO_PARLAY",
            "reason": "insufficient qualified legs",
            "legs": [],
        }

    def _sort_key(leg: Dict[str, object]) -> Tuple[float, str, str, str, str, str]:
        score = leg.get("leg_score")
        score_val = float(score) if _is_finite(score) else -1e9
        return (
            -score_val,
            str(leg.get("sport", "")),
            str(leg.get("game_key", "")),
            str(leg.get("market", "")),
            str(leg.get("side", "")),
            str(leg.get("matchup", "")),
        )

    eligible_legs = sorted(eligible_legs, key=_sort_key)
    selected: List[Dict[str, object]] = []
    used_games: set[str] = set()
    used_teams: set[str] = set()
    sport_counts: Dict[str, int] = {}
    market_counts: Dict[str, int] = {}

    for leg in eligible_legs:
        if len(selected) >= max_legs:
            break

        game_key = str(leg.get("game_key") or "")
        if game_key and game_key in used_games:
            continue

        home = str(leg.get("home") or "")
        away = str(leg.get("away") or "")
        if home and home in used_teams:
            continue
        if away and away in used_teams:
            continue

        new_total = len(selected) + 1
        new_sport_count = sport_counts.get(leg["sport"], 0) + 1
        cap_denominator = max(min_legs, new_total)
        if new_total > 1 and new_sport_count / cap_denominator > 0.60:
            continue
        new_market_count = market_counts.get(leg["market"], 0) + 1
        if new_total > 1 and new_market_count / cap_denominator > 0.60:
            continue

        selected.append(leg)
        if game_key:
            used_games.add(game_key)
        if home:
            used_teams.add(home)
        if away:
            used_teams.add(away)
        sport_counts[leg["sport"]] = new_sport_count
        market_counts[leg["market"]] = new_market_count

    if len(selected) < min_legs:
        return {
            "status": "NO_PARLAY",
            "reason": "correlation filters reduced legs below minimum",
            "legs": [],
        }

    combined_decimal_odds = 1.0
    combined_model_prob = 1.0
    combined_market_prob = 1.0
    for leg in selected:
        decimal = leg.get("decimal_odds")
        if np.isfinite(decimal):
            combined_decimal_odds *= float(decimal)
        combined_model_prob *= float(leg["p_model_final"])
        combined_market_prob *= float(leg["p_market_used"])

    correlation_penalty = 1.0 - 0.03 * max(len(selected) - 1, 0)
    combined_model_prob_adj = float(np.clip(combined_model_prob * correlation_penalty, 0.0001, 0.999))
    parlay_ev = combined_model_prob_adj * combined_decimal_odds - 1.0

    num_legs = len(selected)
    base_units = min(0.25, 0.05 * num_legs)
    max_units = float(os.getenv("PARLAY_MAX_UNITS", "0.5"))
    parlay_units = min(base_units, max_units)
    unit_dollars = float(cfg.get("unit_dollars", PARLAY_DEFAULT_UNIT_DOLLARS))
    stake_dollars = parlay_units * unit_dollars

    sports_used = {leg["sport"] for leg in selected if leg.get("sport")}
    why = f"Found {num_legs} qualified legs across {len(sports_used)} sport(s) after correlation filters."

    return {
        "status": "PARLAY_READY",
        "n_legs": num_legs,
        "legs": selected,
        "combined_decimal_odds": combined_decimal_odds,
        "combined_model_prob": combined_model_prob,
        "combined_model_prob_adj": combined_model_prob_adj,
        "combined_market_prob": combined_market_prob,
        "parlay_ev": parlay_ev,
        "parlay_units": parlay_units,
        "stake_dollars": stake_dollars,
        "why": why,
    }


def save_parlay_outputs(date_str: str, parlay_dict: Dict[str, object]) -> None:
    date_tag = str(date_str).replace("/", "-")
    os.makedirs("results", exist_ok=True)
    json_path = Path("results") / f"parlay_{date_tag}.json"
    csv_path = Path("results") / f"parlay_{date_tag}.csv"

    import json

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parlay_dict, f, indent=2)

    legs = parlay_dict.get("legs", [])
    rows: List[Dict[str, object]] = []
    if parlay_dict.get("status") != "PARLAY_READY":
        rows.append(
            {
                "date": date_tag,
                "leg_num": "",
                "sport": "",
                "matchup": "NO PARLAY TODAY",
                "market": "",
                "side": "",
                "line": "",
                "price": "",
                "decimal_odds": "",
                "p_model_final": "",
                "p_market_used": "",
                "edge_prob_final": "",
                "primary_ev": "",
                "confidence": "",
                "value_tier": "",
                "decision_flags": "",
                "reason_short": parlay_dict.get("reason", ""),
            }
        )
    else:
        for idx, leg in enumerate(legs, start=1):
            rows.append(
                {
                    "date": date_tag,
                    "leg_num": idx,
                    "sport": leg.get("sport"),
                    "matchup": leg.get("matchup"),
                    "market": leg.get("market"),
                    "side": leg.get("side"),
                    "line": leg.get("line"),
                    "price": leg.get("price"),
                    "decimal_odds": leg.get("decimal_odds"),
                    "p_model_final": leg.get("p_model_final"),
                    "p_market_used": leg.get("p_market_used"),
                    "edge_prob_final": leg.get("edge_prob_final"),
                    "primary_ev": leg.get("primary_ev"),
                    "confidence": leg.get("confidence"),
                    "value_tier": leg.get("value_tier"),
                    "decision_flags": leg.get("decision_flags"),
                    "reason_short": leg.get("reason_short"),
                }
            )
        rows.append(
            {
                "date": date_tag,
                "leg_num": "SUMMARY",
                "sport": "",
                "matchup": "",
                "market": "",
                "side": "",
                "line": "",
                "price": "",
                "decimal_odds": parlay_dict.get("combined_decimal_odds"),
                "p_model_final": parlay_dict.get("combined_model_prob_adj"),
                "p_market_used": parlay_dict.get("combined_market_prob"),
                "edge_prob_final": "",
                "primary_ev": parlay_dict.get("parlay_ev"),
                "confidence": "",
                "value_tier": "",
                "decision_flags": "",
                "reason_short": "",
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def print_parlay_card(parlay_dict: Dict[str, object]) -> None:
    print("\nPARLAY CARD")
    status = parlay_dict.get("status", "UNKNOWN")
    if status != "PARLAY_READY":
        print("NO PARLAY TODAY")
        reason = parlay_dict.get("reason")
        if reason:
            print(f"[parlay] reason={reason}")
        return
    legs = parlay_dict.get("legs", [])
    print(
        f"Legs: {len(legs)} | Units: {parlay_dict.get('parlay_units')} "
        f"| Stake: ${parlay_dict.get('stake_dollars')}"
    )
    for i, leg in enumerate(legs, start=1):
        print(
            f"{i}. {str(leg.get('sport', '')).upper()} {leg.get('matchup')} "
            f"{leg.get('market')} {leg.get('side')} line={leg.get('line')} price={leg.get('price')} "
            f"p_win={leg.get('p_model_final')}"
        )
    print(
        "Combined: "
        f"model_prob={parlay_dict.get('combined_model_prob_adj')} "
        f"market_prob={parlay_dict.get('combined_market_prob')} "
        f"decimal_odds={parlay_dict.get('combined_decimal_odds')}"
    )
    why = parlay_dict.get("why")
    if why:
        print(f"Why this parlay exists today: {why}")
