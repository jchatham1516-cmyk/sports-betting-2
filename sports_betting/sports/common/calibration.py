from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, probabilities: list[float], outcomes: list[int]) -> None:
        self.model.fit(np.array(probabilities), np.array(outcomes))
        self.fitted = True

    def transform(self, probabilities: list[float]) -> list[float]:
        if not self.fitted:
            return [float(min(0.995, max(0.005, p))) for p in probabilities]
        calibrated = self.model.predict(np.array(probabilities))
        return [float(min(0.995, max(0.005, p))) for p in calibrated]
