"""
Probability calibration for sports betting models.
Uses Platt scaling to adjust raw model probabilities.
"""

import numpy as np
import json
import os
from typing import Dict, List


class ProbabilityCalibrator:
    """
    Platt scaling-based calibration for sports betting probabilities.
    Maintains separate calibrators for ML, ATS, and Totals per sport.
    """
    
    def __init__(self, sport: str):
        self.sport = sport
        self.calibrators = {}  # {bet_type: (A, B)} for logistic scaling
        self.load_calibration()
    
    def load_calibration(self):
        """Load pre-fitted calibration parameters"""
        cal_file = f"sports/common/calibration_params_{self.sport}.json"
        if os.path.exists(cal_file):
            with open(cal_file, 'r') as f:
                self.calibrators = json.load(f)
            print(f"[CALIBRATION] Loaded parameters for {self.sport}")
        else:
            # Default: no adjustment (A=1, B=0)
            self.calibrators = {
                'moneyline': {'A': 1.0, 'B': 0.0},
                'spread': {'A': 1.0, 'B': 0.0},
                'total': {'A': 1.0, 'B': 0.0}
            }
            print(f"[CALIBRATION] Using default parameters for {self.sport}")
    
    def calibrate(self, prob: float, bet_type: str) -> float:
        """
        Apply Platt scaling: calibrated = 1 / (1 + exp(A * logit(prob) + B))
        
        Args:
            prob: Raw probability from model [0,1]
            bet_type: 'moneyline', 'spread', or 'total'
        
        Returns:
            Calibrated probability
        """
        if bet_type not in self.calibrators:
            print(f"[CALIBRATION WARNING] Unknown bet type: {bet_type}, using raw prob")
            return prob
        
        params = self.calibrators[bet_type]
        A, B = params['A'], params['B']
        
        # Clip to avoid log(0)
        prob = np.clip(prob, 0.001, 0.999)
        
        # logit transform
        logit = np.log(prob / (1 - prob))
        
        # Apply scaling
        scaled_logit = A * logit + B
        
        # Transform back
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return float(np.clip(calibrated, 0.01, 0.99))


# For backward compatibility
def calibrate_probability(prob: float, sport: str, bet_type: str = 'moneyline') -> float:
    """Simple wrapper function"""
    calibrator = ProbabilityCalibrator(sport)
    return calibrator.calibrate(prob, bet_type)
