import json

from sports.common import margin_calibration


def test_zero_calibrator_is_untrained_and_ignored(tmp_path):
    path = tmp_path / "margin_cal.json"
    path.write_text(json.dumps({"a": 0.0, "b": 0.0}), encoding="utf-8")

    cal = margin_calibration.load(str(path))

    assert cal is None
