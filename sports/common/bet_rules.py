# sports/common/bet_rules.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from sports.common.confidence import compute_confidence_score, confidence_bucket, value_tier_from_edges
from sports.common.sanity import sanity_check


# ----------------------------
# ATS-only config
# ----------------------------
# Weekday: Monday=0 ... Sunday=6
ATS_ONLY_DAYS = {
    "nba": set(),      # e.g. {1, 3} for Tue/Thu
    "nfl": set(),      # e.g. {6} for Sundays
    "nhl": set(),      # e.g. {2} for Wednesdays
}


@dataclass
class Thresholds:
    # ML thresholds (probability points)
    ml_play: float = 0.07
    ml_strong: float = 0.10

    # ATS thresholds (spread points)
    ats_play: float = 2.0
    ats_strong: float = 3.0

    # When ATS-only day, use ATS thresholds only
    ats_only_play: float = 2.0
    ats_only_strong: float = 3.0


SPORT_THRESHOLDS = {
    "nba": Thresholds(ml_play=0.07, ml_strong=0.10, ats_play=2.0, ats_strong=3.0, ats_only_play=2.0, ats_only_strong=3.0),
    "nfl": Thresholds(ml_play=0.06, ml_strong=0.09, ats_play=1.5, ats_strong=2.5, ats_only_play=1.5, ats_only_strong=2.5),
    "nhl": Thresholds(ml_play=0.06, ml_strong=0.09, ats_play=0.6, ats_strong=1.0, ats_only_play=0.6, ats_only_strong=1.0),
}


def american_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if np.isnan(o):
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def _weekday_from_date_str(mmddyyyy: str) -> Optional[int]:
    try:
        d = datetime.strptime(mmddyyyy, "%m/%d/%Y")
        return d.weekday()
    except Exception:
        return None


def apply_bet_rules(
    df: pd.DataFrame,
    *,
    sport: str,
    bankroll: float = 250.0,
    unit_pct: float = 0.04,
) -> pd.DataFrame:
    """
    Adds:
      market_home_prob, edge_home/away, spread_edge_home,
      ml_recommendation, spread_recommendation, primary_recommendation,
      confidence, value_tier, why_bet, play_pass, bet_size, unit_dollars, units,
      sanity_flag, sanity_reason
    """
    if df is None or df.empty:
        return df

    th = SPORT_THRESHOLDS.get(sport, SPORT_THRESHOLDS["nba"])

    # Unit sizing
    unit_dollars = round(float(bankroll) * float(unit_pct), 2)
    if unit_dollars <= 0:
        unit_dollars = 10.0

    # Determine ATS-only day (based on first row date)
    weekday = _weekday_from_date_str(str(df.iloc[0].get("date", "")).strip())
    ats_only = (weekday is not None) and (weekday in ATS_ONLY_DAYS.get(sport, set()))

    # Ensure columns exist
    for col in ["home_ml", "away_ml", "home_spread", "model_home_prob", "model_spread_home"]:
        if col not in df.columns:
            df[col] = np.nan

    market_home_probs = []
    edge_homes = []
    edge_aways = []
    spread_edges = []
    sanity_flags = []
    sanity_reasons = []

    ml_recs = []
    ats_recs = []
    primary_recs = []
    confs = []
    tiers = []
    whys = []
    play_passes = []
    bet_sizes = []
    units = []

    for _, r in df.iterrows():
        model_p = float(r.get("model_home_prob")) if r.get("model_home_prob") is not None else 0.5
        model_sp = r.get("model_spread_home")
        model_sp = None if (model_sp is None or (isinstance(model_sp, float) and np.isnan(model_sp))) else float(model_sp)

        home_ml = r.get("home_ml")
        away_ml = r.get("away_ml")
        home_spread = r.get("home_spread")
        home_spread = None if (home_spread is None or (isinstance(home_spread, float) and np.isnan(home_spread))) else float(home_spread)

        mkt_p = american_to_prob(home_ml)
        if mkt_p is None:
            mkt_p = 0.5

        edge_home = float(model_p - mkt_p)
        edge_away = float(-edge_home)

        # Spread edge: positive means value on HOME ATS (market less negative than model)
        # Example: market -6.5 vs model -9.0 => spread_edge_home = +2.5
        spread_edge_home = None
        if home_spread is not None and model_sp is not None:
            spread_edge_home = float(home_spread - model_sp)

        ok, reason = sanity_check(sport=sport, model_spread_home=model_sp, market_spread_home=home_spread)
        sanity_flags.append("OK" if ok else "REJECT")
        sanity_reasons.append(reason)

        # Recommendations
        ml_rec = "No ML bet (edge too small)"
        ats_rec = "Too close to call ATS (edge too small)"
        primary = "NO BET — edges too small"

        # If ATS-only day: ignore ML recs (still compute/display edges)
        if ats_only:
            ml_rec = "ATS-ONLY DAY (ML disabled)"

        # ML recommendation
        if not ats_only and ok:
            if edge_home >= th.ml_strong:
                ml_rec = "Model PICK: HOME ML (strong)"
            elif edge_home >= th.ml_play:
                ml_rec = "Model lean: HOME ML"
            elif edge_home <= -th.ml_strong:
                ml_rec = "Model PICK: AWAY ML (strong)"
            elif edge_home <= -th.ml_play:
                ml_rec = "Model lean: AWAY ML"

        # ATS recommendation
        if ok and spread_edge_home is not None:
            # choose thresholds (ATS-only uses ats_only thresholds)
            play_t = th.ats_only_play if ats_only else th.ats_play
            strong_t = th.ats_only_strong if ats_only else th.ats_strong

            if spread_edge_home >= strong_t:
                ats_rec = "Model PICK ATS: HOME (strong)"
            elif spread_edge_home >= play_t:
                ats_rec = "Model lean ATS: HOME"
            elif spread_edge_home <= -strong_t:
                ats_rec = "Model PICK ATS: AWAY (strong)"
            elif spread_edge_home <= -play_t:
                ats_rec = "Model lean ATS: AWAY"

        # Decide primary recommendation (prefer strongest signal)
        if not ok:
            primary = "NO BET — sanity reject"
        else:
            ml_strength = abs(edge_home)
            ats_strength = abs(spread_edge_home) if spread_edge_home is not None else 0.0

            if ats_only:
                # ATS-only: primary is ATS if it qualifies
                if spread_edge_home is not None and ats_strength >= th.ats_only_play:
                    primary = ats_rec
                else:
                    primary = "NO BET — edges too small"
            else:
                # Normal: pick the stronger of ML vs ATS if it qualifies
                best = None
                if ml_strength >= th.ml_play:
                    best = ("ML", ml_strength)
                if spread_edge_home is not None and ats_strength >= th.ats_play:
                    if (best is None) or (ats_strength > best[1]):
                        best = ("ATS", ats_strength)

                if best is None:
                    primary = "NO BET — edges too small"
                else:
                    primary = ml_rec if best[0] == "ML" else ats_rec

        # Confidence + value tier (normalized)
        score = compute_confidence_score(sport=sport, edge_home=edge_home, spread_edge_home=spread_edge_home)
        conf = confidence_bucket(score)
        tier = value_tier_from_edges(edge_home=edge_home, spread_edge_home=(spread_edge_home or 0.0))

        # Play/pass logic: only PLAY on HIGH confidence and qualifying edge
        play = "PASS"
        bet_size = 0.0
        u = 0.0

        if ok:
            if ats_only:
                if spread_edge_home is not None and abs(spread_edge_home) >= th.ats_only_strong and conf in {"MEDIUM", "HIGH"}:
                    play = "PLAY"
            else:
                if (abs(edge_home) >= th.ml_strong or (spread_edge_home is not None and abs(spread_edge_home) >= th.ats_strong)) and conf in {"MEDIUM", "HIGH"}:
                    play = "PLAY"

        if play == "PLAY":
            bet_size = float(unit_dollars)
            u = 1.0

        # why_bet summary
        ats_txt = ""
        if home_spread is not None and model_sp is not None and spread_edge_home is not None:
            ats_txt = f" | ATS edge={spread_edge_home:+.1f}pts (mkt {home_spread:+.1f} vs model {model_sp:+.1f})"
        why = f"ML edge={edge_home:+.3f} (model {model_p:.3f} vs mkt {mkt_p:.3f}){ats_txt}"
        if ats_only:
            why = "[ATS-ONLY DAY] " + why
        if not ok and reason:
            why = why + f" | {reason}"

        # collect
        market_home_probs.append(float(mkt_p))
        edge_homes.append(float(edge_home))
        edge_aways.append(float(edge_away))
        spread_edges.append(np.nan if spread_edge_home is None else float(spread_edge_home))

        ml_recs.append(ml_rec)
        ats_recs.append(ats_rec)
        primary_recs.append(primary)
        confs.append(conf)
        tiers.append(tier)
        whys.append(why)
        play_passes.append(play)
        bet_sizes.append(float(bet_size))
        units.append(float(u))

    df = df.copy()
    df["market_home_prob"] = market_home_probs
    df["edge_home"] = edge_homes
    df["edge_away"] = edge_aways
    df["spread_edge_home"] = spread_edges
    df["ml_recommendation"] = ml_recs
    df["spread_recommendation"] = ats_recs
    df["primary_recommendation"] = primary_recs
    df["confidence"] = confs
    df["value_tier"] = tiers
    df["why_bet"] = whys
    df["play_pass"] = play_passes
    df["bet_size"] = bet_sizes
    df["unit_dollars"] = float(unit_dollars)
    df["units"] = units
    df["sanity_flag"] = sanity_flags
    df["sanity_reason"] = sanity_reasons

    # normalize model_spread_home_norm column if your pipeline expects it
    if "model_spread_home_norm" not in df.columns and "model_spread_home" in df.columns:
        df["model_spread_home_norm"] = df["model_spread_home"]

    return df
