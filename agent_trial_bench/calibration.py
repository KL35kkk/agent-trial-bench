"""
Calibration utilities for adjusting evaluation scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class CalibrationReport:
    method: str
    pairs_used: int
    parameters: Dict[str, float]


class CalibrationManager:
    """Simple calibration helper supporting linear and isotonic calibration."""

    def __init__(self, method: str = "linear"):
        self.method = (method or "linear").lower()
        self.fitted = False
        self.slope = 1.0
        self.intercept = 0.0
        self._iso_x: List[float] = []
        self._iso_y: List[float] = []

    def fit(self, data_pairs: List[Tuple[float, float]]) -> None:
        if not data_pairs:
            self.fitted = False
            return
        if self.method == "isotonic":
            self._fit_isotonic(data_pairs)
        else:
            self._fit_linear(data_pairs)
        self.fitted = True

    def transform(self, score: float) -> float:
        if not self.fitted:
            return score
        if self.method == "isotonic":
            return self._predict_isotonic(score)
        return max(0.0, min(1.0, self.slope * score + self.intercept))

    def report(self, pairs_used: int) -> CalibrationReport:
        params: Dict[str, float] = {}
        if self.method == "isotonic":
            params["segments"] = len(self._iso_x)
        else:
            params["slope"] = self.slope
            params["intercept"] = self.intercept
        return CalibrationReport(
            method=self.method,
            pairs_used=pairs_used,
            parameters=params,
        )

    # --- internal methods ---

    def _fit_linear(self, pairs: List[Tuple[float, float]]) -> None:
        n = len(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            self.slope = 1.0
            self.intercept = mean_y - self.slope * mean_x
            return
        self.slope = numerator / denominator
        self.intercept = mean_y - self.slope * mean_x

    def _fit_isotonic(self, pairs: List[Tuple[float, float]]) -> None:
        sorted_pairs = sorted(pairs, key=lambda p: p[0])
        xs = [p[0] for p in sorted_pairs]
        ys = [p[1] for p in sorted_pairs]
        weights = [1.0] * len(ys)
        i = 0
        while i < len(ys) - 1:
            if ys[i] <= ys[i + 1]:
                i += 1
                continue
            total_weight = weights[i] + weights[i + 1]
            avg = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / total_weight
            ys[i] = avg
            weights[i] = total_weight
            del ys[i + 1]
            del weights[i + 1]
            del xs[i + 1]
            if i > 0:
                i -= 1
        self._iso_x = xs
        self._iso_y = ys

    def _predict_isotonic(self, score: float) -> float:
        if not self._iso_x:
            return score
        if score <= self._iso_x[0]:
            return self._iso_y[0]
        if score >= self._iso_x[-1]:
            return self._iso_y[-1]
        for i in range(1, len(self._iso_x)):
            if score <= self._iso_x[i]:
                x0, x1 = self._iso_x[i - 1], self._iso_x[i]
                y0, y1 = self._iso_y[i - 1], self._iso_y[i]
                if x1 == x0:
                    return y1
                ratio = (score - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return self._iso_y[-1]
