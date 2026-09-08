from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AnomalyEvent:
    start: int
    end: int
    kind: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("event start must be non-negative")
        if self.end <= self.start:
            raise ValueError("event end must be greater than start")
        if not self.kind:
            raise ValueError("event kind must not be empty")

    def shifted(self, offset: int) -> AnomalyEvent:
        return AnomalyEvent(self.start + offset, self.end + offset, self.kind)


@dataclass(frozen=True, slots=True)
class TimeSeriesDataset:
    values: np.ndarray
    labels: np.ndarray
    events: tuple[AnomalyEvent, ...]
    calibration_end: int
    name: str = "synthetic"

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        labels = np.asarray(self.labels)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if len(values) != len(labels):
            raise ValueError("values and labels must have equal length")
        if not 2 <= self.calibration_end < len(values):
            raise ValueError("calibration_end must split the series")
        if np.any(labels[: self.calibration_end]):
            raise ValueError("calibration region must be anomaly-free")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        for event in self.events:
            if event.end > len(values):
                raise ValueError("event exceeds series length")

    @property
    def evaluation_start(self) -> int:
        return self.calibration_end

    def evaluation_events(self) -> tuple[AnomalyEvent, ...]:
        offset = -self.calibration_end
        return tuple(event.shifted(offset) for event in self.events)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    detector: str
    scores: np.ndarray
    threshold: float
    predictions: np.ndarray

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores)
        predictions = np.asarray(self.predictions)
        if scores.ndim != 1 or predictions.ndim != 1:
            raise ValueError("scores and predictions must be one-dimensional")
        if len(scores) != len(predictions):
            raise ValueError("scores and predictions must have equal length")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
