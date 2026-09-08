from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

import numpy as np

from anomaly_lab.domain import AnomalyEvent


@dataclass(frozen=True, slots=True)
class PointMetrics:
    precision: float
    recall: float
    f1: float
    specificity: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EventMetrics:
    detected_events: int
    total_events: int
    event_recall: float
    false_positive_runs: int
    false_alarms_per_1000_normal_points: float
    mean_detection_delay: float | None
    max_detection_delay: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    point: PointMetrics
    event: EventMetrics

    def to_dict(self) -> dict[str, object]:
        return {"point": self.point.to_dict(), "event": self.event.to_dict()}


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> PointMetrics:
    truth = np.asarray(y_true, dtype=bool)
    prediction = np.asarray(y_pred, dtype=bool)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("truth and prediction must be equal-length one-dimensional arrays")

    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PointMetrics(precision, recall, f1, specificity, tp, fp, fn, tn)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def event_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    events: tuple[AnomalyEvent, ...],
    *,
    tolerance: int = 0,
) -> EventMetrics:
    truth = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted):
        raise ValueError("labels and predictions must be equal-length one-dimensional arrays")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    detected = 0
    delays: list[int] = []
    event_windows: list[tuple[int, int]] = []
    for event in events:
        start = max(0, event.start - tolerance)
        end = min(len(predicted), event.end + tolerance)
        if start >= len(predicted) or end <= 0:
            continue
        event_windows.append((start, end))
        hits = np.flatnonzero(predicted[start:end])
        if len(hits):
            detected += 1
            first_hit = start + int(hits[0])
            delays.append(max(0, first_hit - event.start))

    false_positive_runs = 0
    for run_start, run_end in _runs(predicted):
        overlaps_event = any(run_start < end and run_end > start for start, end in event_windows)
        if not overlaps_event:
            false_positive_runs += 1

    normal_points = int(np.sum(~truth))
    false_alarm_rate = (
        1000.0 * false_positive_runs / normal_points if normal_points else 0.0
    )
    total_events = len(events)
    return EventMetrics(
        detected_events=detected,
        total_events=total_events,
        event_recall=detected / total_events if total_events else 0.0,
        false_positive_runs=false_positive_runs,
        false_alarms_per_1000_normal_points=false_alarm_rate,
        mean_detection_delay=mean(delays) if delays else None,
        max_detection_delay=max(delays) if delays else None,
    )


def evaluate_detection(
    labels: np.ndarray,
    predictions: np.ndarray,
    events: tuple[AnomalyEvent, ...],
    *,
    tolerance: int = 3,
) -> EvaluationReport:
    return EvaluationReport(
        point=point_metrics(labels, predictions),
        event=event_metrics(labels, predictions, events, tolerance=tolerance),
    )
