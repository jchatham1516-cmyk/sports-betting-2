import importlib.util
from pathlib import Path

import pandas as pd


def _load_perf_dashboard_module():
    module_path = Path("scripts/perf_dashboard.py")
    spec = importlib.util.spec_from_file_location("perf_dashboard", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_perf_dashboard_runs_on_fixture(tmp_path):
    bet_log_path = tmp_path / "bet_log.csv"
    data = [
        {
            "date": "2025-01-01",
            "sport": "nba",
            "market_type": "moneyline",
            "price_at_bet": -110,
            "result": "WIN",
            "stake_dollars": 10.0,
            "profit_dollars": 9.09,
            "p_model_final": 0.62,
            "edge_prob_final": 0.08,
        },
        {
            "date": "2025-01-02",
            "sport": "nhl",
            "market_type": "spread",
            "price_at_bet": -105,
            "result": "LOSS",
            "stake_dollars": 10.0,
            "profit_dollars": -10.0,
            "p_model_final": 0.58,
            "edge_prob_final": 0.05,
        },
    ]
    pd.DataFrame(data).to_csv(bet_log_path, index=False)

    out_dir = tmp_path / "tracking"
    module = _load_perf_dashboard_module()
    summary_path, report_path = module.build_performance_dashboard(str(bet_log_path), out_dir=str(out_dir))

    assert summary_path.exists()
    assert report_path.exists()
    assert (out_dir / "perf_by_sport.csv").exists()
    assert (out_dir / "perf_by_market.csv").exists()
    assert (out_dir / "equity_curve.csv").exists()
    assert (out_dir / "drawdowns.csv").exists()
