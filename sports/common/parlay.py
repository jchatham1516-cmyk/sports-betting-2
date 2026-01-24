from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sports.common.bet_config import get_sport_bet_config
from sports.common.bet_rules import american_to_decimal, implied_prob_american


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
