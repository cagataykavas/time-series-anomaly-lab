from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np


class AnomalyDetector(Protocol):
    name: str

    def fit(self, values: np.ndarray) -> Self: ...

    def score(self, values: np.ndarray) -> np.ndarray: ...


def _as_series(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return array


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-8)


@dataclass
class CausalRollingMADDetector:
    """Score each point against only the history available before that point."""

    window: int = 101
    min_history: int = 25
    name: str = "causal_rolling_mad"

    def __post_init__(self) -> None:
        if self.window < 5:
            raise ValueError("window must be at least 5")
        if not 3 <= self.min_history <= self.window:
            raise ValueError("min_history must be between 3 and window")
        self._history: np.ndarray | None = None
        self._fallback_center = 0.0
        self._fallback_scale = 1.0

    def fit(self, values: np.ndarray) -> Self:
        series = _as_series(values)
        self._history = series[-self.window :].copy()
        self._fallback_center, self._fallback_scale = _robust_scale(series)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("fit must be called before score")
        series = _as_series(values)
        history = list(self._history)
        scores = np.empty(len(series), dtype=np.float64)

        for index, value in enumerate(series):
            local = np.asarray(history[-self.window :], dtype=np.float64)
            if len(local) >= self.min_history:
                center, scale = _robust_scale(local)
            else:
                center, scale = self._fallback_center, self._fallback_scale
            scores[index] = abs(float(value) - center) / scale
            history.append(float(value))

        return scores


@dataclass
class AdaptiveEWMADetector:
    """Exponentially weighted baseline with clipped state updates after scoring."""

    alpha: float = 0.04
    adaptation_clip: float = 3.0
    name: str = "adaptive_ewma"

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.adaptation_clip <= 0:
            raise ValueError("adaptation_clip must be positive")
        self._mean: float | None = None
        self._variance: float | None = None

    def fit(self, values: np.ndarray) -> Self:
        series = _as_series(values)
        self._mean = float(np.mean(series))
        variance = float(np.var(series, ddof=1)) if len(series) > 1 else 0.0
        self._variance = max(variance, 1e-8)
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        if self._mean is None or self._variance is None:
            raise RuntimeError("fit must be called before score")
        series = _as_series(values)
        mean = self._mean
        variance = self._variance
        scores = np.empty(len(series), dtype=np.float64)

        for index, value in enumerate(series):
            sigma = max(float(np.sqrt(variance)), 1e-8)
            residual = float(value) - mean
            scores[index] = abs(residual) / sigma

            clipped = float(np.clip(residual, -self.adaptation_clip * sigma, self.adaptation_clip * sigma))
            updated_value = mean + clipped
            new_mean = (1.0 - self.alpha) * mean + self.alpha * updated_value
            deviation = updated_value - mean
            variance = (1.0 - self.alpha) * variance + self.alpha * deviation * deviation
            mean = new_mean

        return scores


@dataclass
class PCAWindowReconstructionDetector:
    """Low-rank reconstruction error over causal windows using NumPy SVD."""

    window: int = 48
    components: int = 5
    name: str = "pca_window_reconstruction"

    def __post_init__(self) -> None:
        if self.window < 4:
            raise ValueError("window must be at least 4")
        if not 1 <= self.components < self.window:
            raise ValueError("components must be between 1 and window - 1")
        self._tail: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._basis: np.ndarray | None = None

    @staticmethod
    def _windows(values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < window:
            raise ValueError("series is shorter than the configured window")
        return np.lib.stride_tricks.sliding_window_view(values, window).copy()

    def fit(self, values: np.ndarray) -> Self:
        series = _as_series(values)
        windows = self._windows(series, self.window)
        mean = np.mean(windows, axis=0)
        scale = np.std(windows, axis=0, ddof=1)
        scale = np.where(scale < 1e-8, 1.0, scale)
        standardized = (windows - mean) / scale
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        self._mean = mean
        self._scale = scale
        self._basis = vt[: self.components]
        self._tail = series[-(self.window - 1) :].copy()
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        if self._tail is None or self._mean is None or self._scale is None or self._basis is None:
            raise RuntimeError("fit must be called before score")
        series = _as_series(values)
        combined = np.concatenate([self._tail, series])
        windows = self._windows(combined, self.window)
        standardized = (windows - self._mean) / self._scale
        coefficients = standardized @ self._basis.T
        reconstructed = coefficients @ self._basis
        error = np.mean(np.square(standardized - reconstructed), axis=1)
        if len(error) != len(series):
            raise AssertionError("window-to-point alignment failed")
        return error.astype(np.float64)
