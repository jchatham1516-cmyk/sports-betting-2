from sports.common.ats_calibration import ATSCalibrator
from sports.nba.model import _phi


def test_ats_calibrator_changes_probability():
    calibrator = ATSCalibrator(a=2.0, b=-0.3, n_samples=250)
    x_val = 0.5
    raw_prob = _phi(x_val)
    cal_prob = calibrator.predict_prob_cover(x_val)

    assert abs(cal_prob - raw_prob) > 1e-3
