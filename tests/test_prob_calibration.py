import numpy as np
import pandas as pd

from sports.common import prob_calibration


def test_platt_calibration_fit_and_apply(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prob_calibration._CALIBRATOR_CACHE.clear()

    rng = np.random.default_rng(42)
    n = 400
    p_raw = rng.uniform(0.05, 0.95, size=n)
    a_true, b_true = 1.2, -0.1
    logit = np.log(p_raw / (1 - p_raw))
    p_true = 1.0 / (1.0 + np.exp(-(a_true * logit + b_true)))
    y = rng.uniform(0.0, 1.0, size=n) < p_true

    df = pd.DataFrame(
        {
            "sport": "nba",
            "market_type": "moneyline",
            "p_model_raw": p_raw,
            "result": np.where(y, "WIN", "LOSS"),
        }
    )

    prob_calibration.fit_calibrator("nba", df)
    p_test = 0.2
    expected = 1.0 / (1.0 + np.exp(-(a_true * np.log(p_test / (1 - p_test)) + b_true)))
    calibrated = prob_calibration.calibrate_prob("nba", p_test, market_type="ML")

    assert abs(calibrated - expected) < abs(p_test - expected)


def test_ml_bucket_calibration_prefers_bucket(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    prob_calibration._CALIBRATOR_CACHE.clear()

    rng = np.random.default_rng(7)
    p_raw = rng.uniform(0.1, 0.9, size=350)
    y = rng.uniform(0.0, 1.0, size=350) < p_raw

    prob_calibration.update_prob_calibration(
        "nba",
        p_raw,
        y,
        window=300,
        min_samples=120,
        market_type="ML",
        bucket="favorite",
    )
    p_test = 0.65
    calibrated = prob_calibration.calibrate_prob("nba", p_test, market_type="ML", price=-180)

    assert np.isfinite(calibrated)
    assert calibrated != p_test
