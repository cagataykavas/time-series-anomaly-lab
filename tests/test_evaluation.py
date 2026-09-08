from __future__ import annotations

import numpy as np

from anomaly_lab.domain import AnomalyEvent
from anomaly_lab.evaluation import event_metrics, point_metrics


def test_point_metrics_match_confusion_matrix() -> None:
    truth = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    prediction = np.array([0, 1, 0, 1, 0, 1], dtype=bool)

    report = point_metrics(truth, prediction)

    assert (report.tp, report.fp, report.fn, report.tn) == (2, 1, 1, 2)
    assert report.precision == 2 / 3
    assert report.recall == 2 / 3
    assert report.f1 == 2 / 3
    assert report.specificity == 2 / 3


def test_event_metrics_measure_recall_delay_and_false_alarm_runs() -> None:
    labels = np.zeros(30, dtype=bool)
    labels[5:9] = True
    labels[18:22] = True
    events = (
        AnomalyEvent(5, 9, "shift"),
        AnomalyEvent(18, 22, "drift"),
    )
    predictions = np.zeros(30, dtype=bool)
    predictions[7:8] = True
    predictions[12:14] = True
    predictions[19:20] = True

    report = event_metrics(labels, predictions, events)

    assert report.detected_events == 2
    assert report.total_events == 2
    assert report.event_recall == 1.0
    assert report.false_positive_runs == 1
    assert report.mean_detection_delay == 1.5
    assert report.max_detection_delay == 2
    assert report.false_alarms_per_1000_normal_points > 0


def test_event_tolerance_can_credit_near_boundary_detection() -> None:
    labels = np.zeros(20, dtype=bool)
    labels[10:13] = True
    event = AnomalyEvent(10, 13, "spike")
    predictions = np.zeros(20, dtype=bool)
    predictions[9] = True

    strict = event_metrics(labels, predictions, (event,), tolerance=0)
    tolerant = event_metrics(labels, predictions, (event,), tolerance=1)

    assert strict.event_recall == 0.0
    assert tolerant.event_recall == 1.0
    assert tolerant.mean_detection_delay == 0
