# current_of_sports_betting_algorithm.py
from __future__ import annotations

import os
import argparse
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

from recommendations import add_recommendations_to_df, Thresholds

from sports.common.eval import (
    evaluate_predictions,
    update_eval_history_with_scores,
    update_eval_history_with_bet_predictions,
)
from sports.common.odds_sources import (
    fetch_odds_for_date_from_odds_api,
    fetch_odds_for_date_from_csv,
    SPORT_TO_ODDS_KEY,
)
from sports.common import tracker
from sports.common.bet_logger import append_plays_to_bet_log

from sports.common.bankroll import (
    DEFAULT_BANKROLL,
    UNIT_PCT,
)
from sports.common.bet_rules import (
    DecisionSettings,
    decide_bet_from_row,
    format_decision_trace,
    ml_probabilities_for_row,
    nhl_anchor_weight_for_row,
    primary_metrics_for_row,
)
from sports.common.bet_config import get_sport_bet_config
from sports.common.prob_calibration import fit_calibrator, update_daily_ml_calibration, update_prob_calibration, MIN_SAMPLES
from sports.common.history_builder import build_historical_dataset, season_string_for_date
from sports.common.reporting import (
    build_display_columns,
    build_bet_card,
    build_rankings,
    daily_bet_report,
    generate_backtest_report,
    render_bet_card_summary,
    render_console_report,
    self_check_recent_bets,
)
from sports.common.parlay import (
    build_smart_parlay,
    build_weekly_parlay,
    load_daily_picks,
    load_recent_predictions,
    print_parlay_card,
    save_parlay_outputs,
)
from sports.common.prob_uncertainty import load_uncertainty, update_uncertainty

from sports.nba.bdl_client import (
    get_bdl_api_key,
    season_start_year_for_date,
    fetch_team_ratings_bdl,
)
from sports.nba.model import run_daily_probs_for_date as run_nba_daily

from sports.nfl.model import run_daily_nfl
from sports.nhl.model import run_daily_nhl


def _cap_to_top_plays(df: pd.DataFrame, max_plays: int | None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if max_plays is None or int(max_plays) <= 0:
        return df
    if "play_pass" not in df.columns:
        return df

    plays = df[df["play_pass"].astype(str) == "PLAY"].copy()
    if plays.empty:
        return df

    if "edge_prob_final" in df.columns:
        plays = plays.assign(_score=plays["edge_prob_final"].astype(float))
        plays = plays.sort_values("_score", ascending=False)
    elif "abs_edge_prob" in df.columns:
        plays = plays.assign(_score=plays["abs_edge_prob"].astype(float))
        plays = plays.sort_values("_score", ascending=False)

    if len(plays) <= int(max_plays):
        return df

    keep_ids = set(plays.head(int(max_plays)).index.tolist())
    for i in df.index:
        if str(df.loc[i, "play_pass"]) == "PLAY" and i not in keep_ids:
            df.loc[i, "play_pass"] = "PASS"
            if "bet_size" in df.columns:
                df.loc[i, "bet_size"] = 0.0
            if "units" in df.columns:
                df.loc[i, "units"] = 0.0
            if "raw_units" in df.columns:
                df.loc[i, "raw_units"] = 0.0
            if "final_units" in df.columns:
                df.loc[i, "final_units"] = 0.0
            if "stake_dollars" in df.columns:
                df.loc[i, "stake_dollars"] = 0.0
            if "why_bet" in df.columns:
                df.loc[i, "why_bet"] = str(df.loc[i, "why_bet"]) + " | filtered: top-N plays"
            if "decision_flags" in df.columns:
                suffix = ",TOP_N_FILTER" if str(df.loc[i, "decision_flags"]).strip() else "TOP_N_FILTER"
                df.loc[i, "decision_flags"] = f"{df.loc[i, 'decision_flags']}{suffix}" if str(df.loc[i, "decision_flags"]).strip() else "TOP_N_FILTER"
            if "decision_reason" in df.columns:
                df.loc[i, "decision_reason"] = str(df.loc[i, "decision_reason"]) + " | filtered: top-N plays"
    return df


def _top_n_default_for_sport(sport: str) -> int:
    per_sport = os.getenv(f"TOP_N_{sport.upper()}")
    if per_sport is not None:
        return int(per_sport)
    legacy = os.getenv("MAX_PLAYS_PER_SPORT_PER_DAY")
    if legacy is not None:
        return int(legacy)
    defaults = {"nba": 5, "nhl": 2, "nfl": 2}
    return int(defaults.get(sport, 3))


def _edge_shrink_factor(row: pd.Series) -> float:
    edge_raw = row.get("edge_prob_raw")
    edge_final = row.get("edge_prob_final")
    try:
        edge_raw_val = float(edge_raw)
        edge_final_val = float(edge_final)
    except Exception:
        return float("nan")
    if not np.isfinite(edge_raw_val) or abs(edge_raw_val) < 1e-9:
        return float("nan")
    if not np.isfinite(edge_final_val):
        return float("nan")
    return float(edge_final_val / edge_raw_val)


def _confidence_rank_value(confidence: object) -> int:
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    key = str(confidence or "").upper()
    return rank.get(key, -1)


def _primary_pass_reason(row: pd.Series) -> str:
    mapping = {
        "LOW_EDGE_PASS": "EDGE_TOO_SMALL",
        "LOW_EV_PASS": "LOW_EV",
        "MISSING_DATA_PASS": "MISSING_DATA",
        "EXTREME_ODDS_PASS": "EXTREME_ODDS",
        "DISAGREE_PASS": "DISAGREE_PASS",
        "LOW_CONFIDENCE_PASS": "LOW_CONFIDENCE",
        "NO_PICK": "NO_PICK",
        "UNCALIBRATED_FALLBACK": "UNCALIBRATED",
        "UNCALIBRATED_ZERO_SAMPLES": "UNCALIBRATED_ZERO_SAMPLES",
        "GOALIE_UNCONFIRMED": "GOALIE_UNCONFIRMED",
    }
    flags = str(row.get("decision_flags", "") or "")
    parts = [p.strip() for p in flags.split(",") if p.strip()]
    for flag in parts:
        if flag in mapping:
            return mapping[flag]
    if parts:
        return parts[0]
    reason = str(row.get("decision_reason", "") or "").strip()
    if reason:
        if reason.upper().startswith("NO BET:"):
            reason = reason.split(":", 1)[-1].strip()
        return reason.split()[0].upper()
    return "UNKNOWN"


def _pass_reason_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "play_pass" not in df.columns:
        return pd.DataFrame(columns=["reason", "count"])
    passes = df[df["play_pass"].astype(str) == "PASS"]
    if passes.empty:
        return pd.DataFrame(columns=["reason", "count"])
    reasons = passes.apply(_primary_pass_reason, axis=1)
    counts = reasons.value_counts().reset_index()
    counts.columns = ["reason", "count"]
    return counts


def _ranked_nhl_picks(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    work = df.copy()
    work["_confidence_rank"] = work.get("confidence", "").apply(_confidence_rank_value)
    for col in ("primary_ev", "abs_edge_prob"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.sort_values(
        ["primary_ev", "_confidence_rank", "abs_edge_prob"],
        ascending=[False, False, False],
        kind="mergesort",
        na_position="last",
    )
    work = work.drop(columns=["_confidence_rank"])
    return work


def _fmt_float(val: object) -> str:
    try:
        return f"{float(val):.3f}"
    except Exception:
        return "nan"


def _maybe_load_results_csv(sport: str, game_date: str) -> pd.DataFrame:
    """Attempt to load a local scores/results CSV for the given sport/date."""
    hyphen_date = game_date.replace("/", "-")
    candidates = [
        f"results/scores_{sport}_{hyphen_date}.csv",
        f"results/final_scores_{sport}_{hyphen_date}.csv",
        f"results/results_{sport}_{hyphen_date}.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"[eval] Loaded results from {path} ({len(df)} rows)")
                return df
            except Exception as e:
                print(f"[eval] WARNING: failed to load results CSV {path}: {e}")
                continue
    return pd.DataFrame()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run sports betting model (NBA/NFL/NHL).")
    parser.add_argument("--sport", type=str, default="nba", choices=["nba", "nfl", "nhl"])
    parser.add_argument("--date", type=str, default=None, help="Game date in MM/DD/YYYY (default: today UTC).")
    parser.add_argument("--days_padding", type=int, default=int(os.getenv("ODDS_DAYS_PADDING", "1")))

    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--sizing", type=str, default="flat", choices=["flat", "kelly"])
    parser.add_argument("--flat_pct", type=float, default=UNIT_PCT)
    parser.add_argument("--kelly_mult", type=float, default=0.25)
    parser.add_argument("--kelly_max_pct", type=float, default=0.015)

    parser.add_argument("--play_require_pick", action="store_true")
    parser.add_argument("--play_value_tier", type=str, default="MEDIUM VALUE")
    parser.add_argument("--play_min_conf", type=str, default="MEDIUM", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--play_max_abs_ml", type=int, default=400)

    parser.add_argument("--track_yesterday", action="store_true", help="Grade and track yesterday's bets after running.")
    parser.add_argument("--track_date", type=str, default=None, help="Explicit date (YYYY-MM-DD or MM/DD/YYYY) to grade.")

    parser.add_argument("--debug_decisions", action="store_true", help="Print decision traces for each game.")

    parser.add_argument("--max_plays", type=int, default=None)
    parser.add_argument("--force_full_rebuild", action="store_true", help="Force full Elo backfill before daily run.")
    parser.add_argument("--build-history", action="store_true", help="Build full-season historical dataset.")
    parser.add_argument("--fit-calibration", action="store_true", help="Fit Platt calibration from historical data.")
    parser.add_argument("--season", type=str, default=None, help="Season string like 2025-2026.")
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a structured console report for the current slate.",
    )
    parser.add_argument("--debug", action="store_true", help="Include debug columns in the console report.")
    parser.add_argument("--sort", type=str, default="value", choices=["value", "accuracy"])
    parser.add_argument("--self-check", action="store_true", help="Print a quick self-check report from recent bets.")
    parser.add_argument("--self-check-n", type=int, default=200, help="Number of recent bets to include in --self-check.")
    parser.add_argument("--daily-report", action="store_true", help="Print daily accuracy/calibration report.")
    parser.add_argument("--weekly-parlay", action="store_true", help="Build a weekly parlay from recent plays.")
    parser.add_argument("--parlay-legs", type=int, default=6, help="Number of parlay legs (6 or 7).")
    parser.add_argument("--parlay", action="store_true", help="Build a daily parlay from today's plays.")
    parser.add_argument("--parlay-min-legs", type=int, default=4, help="Minimum daily parlay legs.")
    parser.add_argument("--parlay-max-legs", type=int, default=7, help="Maximum daily parlay legs.")
    parser.add_argument(
        "--parlay-sports",
        type=str,
        default=None,
        help='Comma-separated sports list for parlay (e.g. "nba,nhl"). Defaults to today\'s run sport.',
    )

    args = parser.parse_args(argv)

    if args.daily_report:
        daily_bet_report(bet_log_path="results/tracking/bet_log.csv")
        return 0

    if args.weekly_parlay:
        requested_legs = int(args.parlay_legs)
        if requested_legs not in {6, 7}:
            raise ValueError("--parlay-legs must be 6 or 7")
        recent_preds = load_recent_predictions(preds_dir="results", days_back=7)
        parlay_result = build_weekly_parlay(
            recent_preds,
            min_legs=6,
            max_legs=requested_legs,
        )
        os.makedirs("results", exist_ok=True)
        date_tag = datetime.utcnow().strftime("%Y-%m-%d")
        json_path = f"results/parlay_weekly_{date_tag}.json"
        csv_path = f"results/parlay_weekly_{date_tag}.csv"
        pd.DataFrame(parlay_result.get("legs", [])).to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            import json

            json.dump(parlay_result, f, indent=2)
        status = parlay_result.get("status", "UNKNOWN")
        if status != "PARLAY_READY":
            print("NO PARLAY THIS WEEK")
            print(f"[parlay] reasons={parlay_result.get('reasons', [])}")
        else:
            print(f"[parlay] Built weekly parlay with {len(parlay_result.get('legs', []))} legs.")
        print(f"[parlay] wrote {json_path} and {csv_path}")
        return 0

    game_date = datetime.utcnow().strftime("%m/%d/%Y") if args.date is None else args.date
    season = args.season
    if season is None:
        try:
            season = season_string_for_date(args.sport, datetime.strptime(game_date, "%m/%d/%Y").date())
        except Exception:
            season = season_string_for_date(args.sport, datetime.utcnow().date())

    if args.build_history:
        build_historical_dataset(args.sport, season)
        if args.fit_calibration:
            hist_path = f"data/historical/{args.sport}_{season}.csv"
            if os.path.exists(hist_path):
                hist_df = pd.read_csv(hist_path)
                fit_calibrator(args.sport, hist_df)
        return

    if args.fit_calibration:
        hist_path = f"data/historical/{args.sport}_{season}.csv"
        if not os.path.exists(hist_path):
            raise FileNotFoundError(f"Historical CSV not found: {hist_path}")
        hist_df = pd.read_csv(hist_path)
        fit_calibrator(args.sport, hist_df)
        return

    print(f"Running {args.sport.upper()} model for {game_date}...")

    odds_dict, spreads_dict = {}, {}

    # Odds (API first, fallback CSV)
    try:
        odds_dict, spreads_dict = fetch_odds_for_date_from_odds_api(
            game_date,
            sport_key=SPORT_TO_ODDS_KEY[args.sport],
            days_padding=int(args.days_padding),
        )
        print(f"[odds_api] Loaded odds for {len(odds_dict)} games.")
    except Exception as e:
        print(f"[odds_api] WARNING: failed to load odds from API: {e}")

    if not odds_dict:
        print("[odds] No odds from API; trying CSV fallback...")
        try:
            odds_dict, spreads_dict = fetch_odds_for_date_from_csv(game_date, sport=args.sport)
            print(f"[odds_csv] games found: {len(odds_dict)}")
        except Exception as e:
            print(f"[odds_csv] WARNING: failed to load odds from CSV: {e}")
            odds_dict, spreads_dict = {}, {}

    # Run sport model
    if args.sport == "nba":
        try:
            update_daily_ml_calibration(
                "nba",
                days_back=int(os.getenv("NBA_PROB_CAL_DAYS", "45")),
                min_samples=int(os.getenv("NBA_PROB_CAL_MIN_SAMPLES", "120")),
            )
        except Exception as e:
            print(f"[calibration] WARNING: NBA ML calibration update failed: {e}")

        api_key = get_bdl_api_key()
        game_date_obj = datetime.strptime(game_date, "%m/%d/%Y").date()
        season_year = season_start_year_for_date(game_date_obj)
        end_date_iso = game_date_obj.strftime("%Y-%m-%d")
        stats_df = fetch_team_ratings_bdl(season_year=season_year, end_date_iso=end_date_iso, api_key=api_key)

        results_df = run_nba_daily(
            game_date=game_date,
            odds_dict=odds_dict,
            spreads_dict=spreads_dict,
            stats_df=stats_df,
            api_key=api_key,
            force_full_rebuild=args.force_full_rebuild,
        )

    elif args.sport == "nfl":
        results_df = run_daily_nfl(game_date, odds_dict=odds_dict)

    elif args.sport == "nhl":
        try:
            update_daily_ml_calibration(
                "nhl",
                days_back=int(os.getenv("NHL_PROB_CAL_DAYS", "45")),
                min_samples=int(os.getenv("NHL_PROB_CAL_MIN_SAMPLES", "120")),
            )
        except Exception as e:
            print(f"[calibration] WARNING: NHL ML calibration update failed: {e}")

        before_count = len(odds_dict)
        filtered_odds = {}
        try:
            target_date = datetime.strptime(game_date, "%m/%d/%Y").date()
        except Exception:
            target_date = datetime.utcnow().date()
        window_start = datetime(
            target_date.year, target_date.month, target_date.day, 5, 0, 0, tzinfo=timezone.utc
        )
        window_end = window_start + timedelta(days=1)
        for matchup, info in (odds_dict or {}).items():
            commence_time = (info or {}).get("commence_time")
            if not commence_time:
                continue
            try:
                commence_dt = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
            except Exception:
                continue
            if window_start <= commence_dt < window_end:
                filtered_odds[matchup] = info
        odds_dict = filtered_odds
        print(
            "[nhl odds] filtered games: "
            f"before={before_count} after={len(odds_dict)} "
            f"window_start={window_start.isoformat()} window_end={window_end.isoformat()}"
        )
        results_df = run_daily_nhl(game_date, odds_dict=odds_dict)

    else:
        raise RuntimeError("Unsupported sport")

    if results_df is None:
        results_df = pd.DataFrame()

    print(f"[model] rows returned: {len(results_df)}")

    # If empty AND no columns, force a harmless header-only CSV so you can see the file is valid.
    if results_df.empty and len(results_df.columns) == 0:
        results_df = pd.DataFrame(columns=["date", "home", "away"])

    debug_df = pd.DataFrame()
    if not results_df.empty:
        ml_probs = [ml_probabilities_for_row(r, sport=args.sport) for _, r in results_df.iterrows()]
        results_df["model_home_prob_raw"] = [p["model_home_prob_raw"] for p in ml_probs]
        results_df["model_home_prob_cal"] = [p["model_home_prob_cal"] for p in ml_probs]
        results_df["model_home_prob_final_pre_goalie"] = [
            p["model_home_prob_final_pre_goalie"] for p in ml_probs
        ]
        results_df["model_home_prob_final"] = [p["model_home_prob_final"] for p in ml_probs]
        results_df["model_away_prob_raw"] = [p["model_away_prob_raw"] for p in ml_probs]
        results_df["model_away_prob_cal"] = [p["model_away_prob_cal"] for p in ml_probs]
        results_df["model_away_prob_final_pre_goalie"] = [
            p["model_away_prob_final_pre_goalie"] for p in ml_probs
        ]
        results_df["model_away_prob_final"] = [p["model_away_prob_final"] for p in ml_probs]
        results_df["market_home_prob"] = [p["market_home_prob"] for p in ml_probs]
        results_df["market_away_prob"] = [p["market_away_prob"] for p in ml_probs]

        uncertainty_data = load_uncertainty(args.sport)
        if uncertainty_data:
            print(f"[uncertainty] loaded {args.sport} -> {uncertainty_data}")
        else:
            print(f"[uncertainty] WARNING: no uncertainty file found for {args.sport}")
        conf_high = 0.18
        conf_med = 0.10
        if args.sport == "nhl":
            conf_high = 0.12
            conf_med = 0.06
        results_df, debug_df = add_recommendations_to_df(
            results_df,
            thresholds=Thresholds(
                ml_edge_strong=0.06,
                ml_edge_lean=0.035,
                ats_edge_strong_pts=3.0,
                ats_edge_lean_pts=1.5,
                conf_high=conf_high,
                conf_med=conf_med,
            ),
            model_spread_home_col="model_spread_home" if "model_spread_home" in results_df.columns else None,
            model_margin_home_col=None,
        )

    play_max_abs_ml = None if int(args.play_max_abs_ml) == 0 else int(args.play_max_abs_ml)
    max_plays = int(args.max_plays) if args.max_plays is not None else _top_n_default_for_sport(args.sport)
    unit_dollars = float(args.bankroll) * UNIT_PCT

    decision_settings = DecisionSettings(
        flat_pct=float(args.flat_pct),
        sizing_mode=str(args.sizing),
        kelly_mult=float(args.kelly_mult),
        kelly_max_pct=float(args.kelly_max_pct),
    )

    bet_log_path = "results/tracking/bet_log.csv"
    if not results_df.empty:

        metrics = [primary_metrics_for_row(r, sport=args.sport) for _, r in results_df.iterrows()]
        results_df["primary_market"] = [m[0] for m in metrics]
        results_df["primary_side"] = [m[1] for m in metrics]
        results_df["p_model_raw"] = [m[2] for m in metrics]
        results_df["p_model_cal"] = [m[3] for m in metrics]
        results_df["p_model_final"] = [m[4] for m in metrics]
        results_df["p_market"] = [m[5] for m in metrics]
        results_df["p_model_used"] = results_df["p_model_final"]
        results_df["p_market_used"] = results_df["p_market"]
        results_df["edge_prob_raw"] = [m[6] for m in metrics]
        results_df["edge_prob_cal"] = [m[7] for m in metrics]
        results_df["edge_prob_final"] = [m[8] for m in metrics]
        results_df["abs_edge_prob"] = results_df["edge_prob_final"].abs()
        results_df["confidence"] = [m[9] for m in metrics]
        results_df["confidence_reason"] = [m[10] for m in metrics]
        results_df["value_tier"] = [m[11] for m in metrics]
        results_df["uncalibrated"] = [m[12] for m in metrics]
        results_df["primary_price"] = [m[13] for m in metrics]
        results_df["primary_ev"] = [m[16] for m in metrics]
        results_df["min_play_edge_abs_used"] = [m[17] for m in metrics]
        results_df["min_primary_edge_abs_used"] = [m[18] for m in metrics]
        if args.sport == "nhl":
            nhl_config = get_sport_bet_config("nhl")
            results_df["nhl_anchor_w_used"] = [
                nhl_anchor_weight_for_row(r, nhl_config) for _, r in results_df.iterrows()
            ]
            results_df["edge_shrink_factor"] = results_df.apply(_edge_shrink_factor, axis=1)

        decisions = [
            decide_bet_from_row(
                r,
                unit_dollars=unit_dollars,
                sport=args.sport,
                settings=decision_settings,
                require_pick=args.play_require_pick,
                require_value_tier=args.play_value_tier,
                min_confidence=args.play_min_conf,
                max_abs_moneyline=play_max_abs_ml,
            )
            for _, r in results_df.iterrows()
        ]

        results_df["play_pass"] = [d.play_pass for d in decisions]
        results_df["bet_size"] = [d.bet_size for d in decisions]
        results_df["unit_dollars"] = [d.unit_dollars for d in decisions]
        results_df["units"] = [d.units for d in decisions]
        results_df["why_bet"] = [d.reason for d in decisions]
        results_df["decision_flags"] = [d.decision_flags for d in decisions]
        results_df["decision_reason"] = [d.decision_reason for d in decisions]
        results_df["raw_units"] = [d.raw_units for d in decisions]
        results_df["final_units"] = [d.final_units for d in decisions]
        results_df["edge_prob_raw"] = [d.edge_prob_raw for d in decisions]
        results_df["edge_prob_cal"] = [d.edge_prob_cal for d in decisions]
        results_df["edge_prob_final"] = [d.edge_prob_final for d in decisions]
        results_df["stake_dollars"] = results_df["units"] * results_df["unit_dollars"]

        for (idx, r), d in zip(results_df.iterrows(), decisions):
            print(
                "[decision] "
                f"{r.get('away', '')} @ {r.get('home', '')} "
                f"edge_final={r.get('edge_prob_final')} value_tier={r.get('value_tier')} "
                f"confidence={r.get('confidence')} primary_ev={r.get('primary_ev')} "
                f"play_pass={d.play_pass} decision_flags={d.decision_flags or 'NONE'} "
                f"decision_reason={d.decision_reason or d.reason}"
            )

        if args.debug_decisions:
            print("\n[debug] Decision traces:")
            for (idx, r), d in zip(results_df.iterrows(), decisions):
                conf = str(r.get("confidence", "")).upper()
                if d.play_pass == "PASS" and conf in {"HIGH", "MEDIUM"}:
                    print(format_decision_trace(r, d))
                elif d.decision_flags:
                    print(format_decision_trace(r, d))

        results_df = _cap_to_top_plays(results_df, max_plays)

    if args.sport == "nhl":
        abs_edges = pd.Series(dtype=float)
        if "edge_prob_cal" in results_df.columns:
            abs_edges = results_df["edge_prob_cal"].abs().astype(float)
            abs_edges = abs_edges.replace([np.inf, -np.inf], np.nan).dropna()
        play_series = (
            results_df["play_pass"].astype(str) if "play_pass" in results_df.columns else pd.Series([], dtype=str)
        )
        plays = int((play_series == "PLAY").sum())
        passes = int((play_series == "PASS").sum())
        mean_edge = float(np.nanmean(abs_edges)) if not abs_edges.empty else 0.0
        median_edge = float(np.nanmedian(abs_edges)) if not abs_edges.empty else 0.0
        p90_edge = float(np.nanpercentile(abs_edges, 90)) if not abs_edges.empty else 0.0
        max_edge = float(np.nanmax(abs_edges)) if not abs_edges.empty else 0.0
        print(
            "[nhl summary] "
            f"games={len(results_df)} plays={plays} passes={passes} "
            f"abs_edge_cal mean={mean_edge:.4f} median={median_edge:.4f} "
            f"p90={p90_edge:.4f} max={max_edge:.4f}"
        )
        ranked_plays = _ranked_nhl_picks(results_df)
        if "play_pass" in ranked_plays.columns:
            top_plays = ranked_plays[ranked_plays["play_pass"].astype(str) == "PLAY"].head(5)
        else:
            top_plays = pd.DataFrame()
        if not top_plays.empty:
            print("[nhl top plays]")
            for _, row in top_plays.iterrows():
                print(
                    "  "
                    f"{row.get('away', '')} @ {row.get('home', '')} | "
                    f"{row.get('primary_recommendation', '')} "
                    f"ev={_fmt_float(row.get('primary_ev'))} "
                    f"conf={row.get('confidence', '')} "
                    f"edge={_fmt_float(row.get('edge_prob_final'))}"
                )
        else:
            print("[nhl top plays] none")
        reason_counts = _pass_reason_counts(results_df)
        if not reason_counts.empty:
            print("[nhl pass reasons]")
            for _, row in reason_counts.iterrows():
                print(f"  {row['reason']}: {int(row['count'])}")

    os.makedirs("results", exist_ok=True)
    if not results_df.empty:
        results_df = build_display_columns(results_df)
        results_df = build_rankings(results_df)
        bet_card = build_bet_card(results_df)

        sort_key = "rank_value" if args.sort == "value" else "rank_accuracy"
        if sort_key in results_df.columns:
            results_df = results_df.sort_values(sort_key, kind="mergesort", na_position="last")

        if args.report:
            render_console_report(
                results_df,
                sport=args.sport,
                date=game_date,
                debug=bool(args.debug),
                sort_by=args.sort,
            )
        render_bet_card_summary(
            bet_card,
            sport=args.sport,
            date=game_date,
            max_rows=10,
        )
        if args.sport == "nba":
            date_tag = game_date.replace("/", "-")
            picks_path = f"results/picks_nba_{date_tag}.csv"
            try:
                bet_card.to_csv(picks_path, index=False)
                print(f"[save] wrote ranked bet card -> {picks_path}")
            except Exception as e:
                print(f"[save] WARNING: failed to write bet card: {e}")
        if args.sport == "nhl":
            date_tag = game_date.replace("/", "-")
            picks_path = f"results/picks_nhl_{date_tag}.csv"
            try:
                nhl_ranked = _ranked_nhl_picks(results_df)
                nhl_ranked.to_csv(picks_path, index=False)
                print(f"[save] wrote ranked NHL picks -> {picks_path}")
            except Exception as e:
                print(f"[save] WARNING: failed to write NHL picks: {e}")

    out_name = f"results/predictions_{args.sport}_{game_date.replace('/', '-')}.csv"
    print(f"[save] writing {len(results_df)} rows -> {out_name}")
    results_df.to_csv(out_name, index=False)

    if debug_df is not None and not debug_df.empty:
        dbg_name = f"results/debug_why_ml_vs_ats_{args.sport}_{game_date.replace('/', '-')}.csv"
        debug_df.to_csv(dbg_name, index=False)

    print(f"\nSaved predictions to {out_name}")
    print(f"Bankroll=${float(args.bankroll):.2f} | 1 unit={UNIT_PCT*100:.1f}% = ${unit_dollars:.2f}")

    if args.parlay:
        if int(args.parlay_max_legs) < int(args.parlay_min_legs):
            raise ValueError("--parlay-max-legs must be >= --parlay-min-legs")
        date_tag = game_date.replace("/", "-")
        parlay_sports = (
            [s.strip().lower() for s in str(args.parlay_sports).split(",") if s.strip()]
            if args.parlay_sports
            else [args.sport]
        )
        picks_frames: List[pd.DataFrame] = []
        if not results_df.empty and args.sport in parlay_sports:
            df_current = results_df.copy()
            if "sport" not in df_current.columns:
                df_current["sport"] = args.sport
            picks_frames.append(df_current)
            parlay_sports = [s for s in parlay_sports if s != args.sport]
        if parlay_sports:
            picks_frames.append(load_daily_picks(date_tag, parlay_sports))
        valid_frames = [df for df in picks_frames if df is not None and not df.empty]
        picks_df = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
        parlay_result = build_smart_parlay(
            picks_df,
            min_legs=int(args.parlay_min_legs),
            max_legs=int(args.parlay_max_legs),
            config={"unit_dollars": unit_dollars},
        )
        save_parlay_outputs(date_tag, parlay_result)
        print_parlay_card(parlay_result)
        print(f"[parlay] wrote results/parlay_{date_tag}.json and results/parlay_{date_tag}.csv")

    if "date" not in results_df.columns:
        results_df["date"] = game_date
    results_df["date"] = results_df["date"].fillna(game_date)

    bet_log_path = "results/tracking/bet_log.csv"
    new_bets = append_plays_to_bet_log(results_df, args.sport, bet_log_path=bet_log_path)
    print(f"[bet_log] Added {new_bets} new bets to {bet_log_path}")

    # Evaluation/sanity checks
    eval_date = None
    try:
        eval_date = datetime.strptime(game_date, "%m/%d/%Y").strftime("%m-%d-%Y")
    except Exception:
        eval_date = game_date.replace("/", "-")

    results_scores_df = _maybe_load_results_csv(args.sport, game_date)
    eval_row = evaluate_predictions(
        results_df,
        results_scores_df if not results_scores_df.empty else None,
        sport=args.sport,
        run_date_str=eval_date,
    )

    eval_out = f"results/eval_{args.sport}_{eval_date}.csv"
    try:
        eval_row.to_csv(eval_out, index=False)
        print(f"[eval] Saved daily evaluation -> {eval_out}")
    except Exception as e:
        print(f"[eval] WARNING: failed to save evaluation CSV: {e}")

    try:
        update_eval_history_with_scores(
            sport=args.sport,
            preds_dir="results",
            out_path=f"results/eval_history_{args.sport}.csv",
            days_back=14,
        )
    except Exception as e:
        print(f"[eval history] WARNING: rolling evaluation update failed: {e}")

    eval_bets = pd.DataFrame()
    try:
        eval_bets = update_eval_history_with_bet_predictions(
            sport=args.sport,
            preds_dir="results",
            out_path="results/eval_history.csv",
            days_back=30,
        )
    except Exception as e:
        print(f"[eval history] WARNING: bet history update failed: {e}")

    try:
        if args.sport in {"nba", "nhl"} and not eval_bets.empty:
            sport_key = str(args.sport).lower()
            ml_hist = eval_bets[
                eval_bets.get("market_type", "").astype(str).str.upper() == "ML"
            ].copy()
            if "sport" in ml_hist.columns:
                ml_hist = ml_hist[ml_hist["sport"].astype(str).str.lower() == sport_key].copy()
            ml_hist["model_prob_raw"] = pd.to_numeric(ml_hist.get("model_prob_raw"), errors="coerce")
            ml_hist["model_prob"] = pd.to_numeric(ml_hist.get("model_prob"), errors="coerce")
            ml_hist["actual_result"] = pd.to_numeric(ml_hist.get("actual_result"), errors="coerce")
            p_raw = ml_hist["model_prob_raw"].fillna(ml_hist["model_prob"]).to_numpy()
            p_final = ml_hist["model_prob"].to_numpy()
            y = ml_hist["actual_result"].to_numpy()
            mask_any = np.isfinite(y) & (np.isfinite(p_raw) | np.isfinite(p_final))
            if mask_any.any():
                p_unc = np.where(np.isfinite(p_final), p_final, p_raw)
                mask_cal = np.isfinite(p_raw) & np.isfinite(y)
                update_uncertainty(
                    sport_key,
                    p_unc[mask_any],
                    y[mask_any],
                    window=int(mask_any.sum()),
                    min_samples=30,
                    market="ML",
                )
                if mask_cal.any():
                    min_samples = int(os.getenv(f"{sport_key.upper()}_PROB_CAL_MIN_SAMPLES", MIN_SAMPLES))
                    update_prob_calibration(
                        sport_key,
                        p_raw[mask_cal],
                        y[mask_cal],
                        window=int(mask_cal.sum()),
                        min_samples=min_samples,
                    )
    except Exception as e:
        print(f"[calibration] WARNING: rolling calibration update failed: {e}")

    track_target = None
    if args.track_date:
        track_target = tracker.parse_tracking_date(args.track_date)
        if track_target is None:
            print(f"[tracking] WARNING: could not parse --track_date={args.track_date}; skipping tracking.")
    elif args.track_yesterday:
        track_target = datetime.utcnow().date() - timedelta(days=1)

    if track_target:
        print(f"[tracking] Grading bets for {track_target.isoformat()}...")
        track_result = tracker.track_bets_for_date(sport=args.sport, target_date=track_target, bet_log_path=bet_log_path)
        if not track_result.ok:
            print(f"[tracking] WARNING: tracking failed: {track_result.reason}")
        else:
            summary = track_result.summary
            print(
                "Graded "
                f"{summary.get('graded', 0)} bets: "
                f"{summary.get('wins', 0)}-{summary.get('losses', 0)}-{summary.get('pushes', 0)}"
                f", Profit ${summary.get('profit', 0.0):.2f}, ROI {summary.get('roi', 0.0)*100:.2f}%"
            )
            if summary.get("by_sport"):
                print("[tracking] ROI by sport:")
                for sport, stats in summary.get("by_sport", {}).items():
                    print(
                        f"  {sport}: ROI {stats.get('roi', 0.0)*100:.2f}% "
                        f"win% {stats.get('win_pct', 0.0)*100:.1f}% bets {stats.get('bets', 0)}"
                    )
            if summary.get("by_confidence"):
                print("[tracking] ROI by confidence:")
                for tier, stats in summary.get("by_confidence", {}).items():
                    print(
                        f"  {tier}: ROI {stats.get('roi', 0.0)*100:.2f}% "
                        f"win% {stats.get('win_pct', 0.0)*100:.1f}% bets {stats.get('bets', 0)}"
                    )
            if summary.get("by_value_tier"):
                print("[tracking] ROI by value tier:")
                for tier, stats in summary.get("by_value_tier", {}).items():
                    print(
                        f"  {tier}: ROI {stats.get('roi', 0.0)*100:.2f}% "
                        f"win% {stats.get('win_pct', 0.0)*100:.1f}% bets {stats.get('bets', 0)}"
                    )
            if summary.get("by_odds_bucket"):
                print("[tracking] ROI by odds bucket:")
                for bucket, stats in summary.get("by_odds_bucket", {}).items():
                    print(f"  {bucket}: ROI {stats.get('roi', 0.0)*100:.2f}%")
            graded_path = summary.get("graded_path")
            if graded_path:
                print(f"[tracking] Wrote graded results to {graded_path}")

    if args.self_check:
        self_check_recent_bets(bet_log_path=bet_log_path, last_n=int(args.self_check_n))

    try:
        hist_path = f"data/historical/{args.sport}_{season}.csv"
        report = generate_backtest_report(args.sport, hist_path, bet_log_path=bet_log_path)
        if report:
            overall = report.get("overall", {})
            print(
                "[report] "
                f"{args.sport.upper()} ROI {overall.get('roi', 0.0)*100:.2f}% "
                f"win% {overall.get('win_pct', 0.0)*100:.1f}% bets {overall.get('bets', 0)}"
            )
    except Exception as exc:
        print(f"[report] WARNING: failed to generate report: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
