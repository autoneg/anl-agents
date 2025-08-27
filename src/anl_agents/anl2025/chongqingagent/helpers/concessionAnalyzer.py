import numpy as np
from scipy import stats


class ConcessionAnalyzer:
    def __init__(self, window_size=5, threshold=0.03, sensitivity=0.1):
        self.history = []
        self.window = window_size
        self.threshold = threshold
        self.sensitivity = sensitivity

    def update_history(self, utility):
        self.history.append(float(utility))

    def get_last_step_concession_probability(self):
        try:
            if len(self.history) < 2:
                return 0.0, {}

            checks = {
                "trend_up": self._trend_analysis(),
                "last_step_up": self._last_step_significant(),
                "accelerating_up": self._accelerating_up(),
                "low_fluctuation": self._low_fluctuation()
            }

            confidence = self._calculate_confidence(checks)
            probability = np.clip(confidence / 100, 0.0, 1.0)
            return round(probability, 2), checks
        except Exception:
            return -1.0, {}

    def _trend_analysis(self):
        if len(self.history) < self.window:
            return False
        window_data = self.history[-self.window:]
        x = np.arange(len(window_data))
        slope, _, _, p_val, _ = stats.linregress(x, window_data)
        return slope > 0 and p_val < 0.1

    def _last_step_significant(self):
        if len(self.history) < 2:
            return False
        last_rise = self.history[-1] - self.history[-2]
        return last_rise >= (self.history[-2] * self.threshold)

    def _accelerating_up(self):
        if len(self.history) < 4:
            return False
        rises = np.diff(self.history[-4:])
        if len(rises) < 3:
            return False
        return rises[-1] > np.mean(rises[:-1]) * 1.2

    def _low_fluctuation(self):
        if len(self.history) < self.window:
            return False
        recent = self.history[-self.window:]
        return np.std(recent) < self.sensitivity

    def _calculate_confidence(self, checks):
        weights = {
            "trend_up": 55,
            "last_step_up": 35,
            "accelerating_up": 25,
            "low_fluctuation": -40 if checks["low_fluctuation"] else 10
        }
        return sum(weights[key] * int(val) for key, val in checks.items())