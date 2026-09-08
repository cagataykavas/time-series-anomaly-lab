from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    threshold: float
    method: str
    samples: int


@dataclass(frozen=True, slots=True)
class QuantileCalibrator:
    quantile: float = 0.995

    def __post_init__(self) -> None:
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("quantile must be between 0.5 and 1.0")

    def fit(self, scores: np.ndarray) -> CalibrationResult:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("scores must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must be finite")
        return CalibrationResult(
            threshold=float(np.quantile(values, self.quantile)),
            method=f"quantile:{self.quantile:.5f}",
            samples=len(values),
        )


@dataclass(frozen=True, slots=True)
class RobustSigmaCalibrator:
    sigma: float = 4.0

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def fit(self, scores: np.ndarray) -> CalibrationResult:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("scores must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must be finite")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = max(1.4826 * mad, 1e-8)
        return CalibrationResult(
            threshold=median + self.sigma * robust_scale,
            method=f"median+{self.sigma:g}mad_sigma",
            samples=len(values),
        )
