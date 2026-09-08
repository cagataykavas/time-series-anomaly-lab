from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from anomaly_lab.calibration import CalibrationResult, QuantileCalibrator
from anomaly_lab.detectors import (
    AdaptiveEWMADetector,
    AnomalyDetector,
    CausalRollingMADDetector,
    PCAWindowReconstructionDetector,
)
from anomaly_lab.domain import DetectionResult, TimeSeriesDataset
from anomaly_lab.evaluation import EvaluationReport, evaluate_detection


class ScoreCalibrator(Protocol):
    def fit(self, scores: np.ndarray) -> CalibrationResult: ...


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    median: float
    p95: float
    p99: float
    maximum: float

    @classmethod
    def from_scores(cls, scores: np.ndarray) -> ScoreSummary:
        values = np.asarray(scores, dtype=np.float64)
        return cls(
            median=float(np.median(values)),
            p95=float(np.quantile(values, 0.95)),
            p99=float(np.quantile(values, 0.99)),
            maximum=float(np.max(values)),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    detector: str
    threshold: float
    calibration_method: str
    calibration_samples: int
    fit_samples: int
    evaluation_samples: int
    score_summary: ScoreSummary
    evaluation: EvaluationReport

    def to_dict(self) -> dict[str, object]:
        return {
            "detector": self.detector,
            "threshold": self.threshold,
            "calibration_method": self.calibration_method,
            "calibration_samples": self.calibration_samples,
            "fit_samples": self.fit_samples,
            "evaluation_samples": self.evaluation_samples,
            "score_summary": asdict(self.score_summary),
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    dataset: str
    total_samples: int
    calibration_end: int
    fit_end: int
    records: tuple[BenchmarkRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "total_samples": self.total_samples,
            "calibration_end": self.calibration_end,
            "fit_end": self.fit_end,
            "records": [record.to_dict() for record in self.records],
        }


def default_detectors() -> tuple[AnomalyDetector, ...]:
    return (
        CausalRollingMADDetector(window=101, min_history=25),
        AdaptiveEWMADetector(alpha=0.04, adaptation_clip=3.0),
        PCAWindowReconstructionDetector(window=48, components=5),
    )


def run_benchmark(
    dataset: TimeSeriesDataset,
    *,
    detectors: tuple[AnomalyDetector, ...] | None = None,
    calibrator: ScoreCalibrator | None = None,
    fit_fraction_of_calibration: float = 0.65,
    event_tolerance: int = 3,
) -> BenchmarkReport:
    if not 0.5 <= fit_fraction_of_calibration < 0.9:
        raise ValueError("fit_fraction_of_calibration must be in [0.5, 0.9)")

    active_detectors = detectors or default_detectors()
    if not active_detectors:
        raise ValueError("at least one detector is required")
    active_calibrator = calibrator or QuantileCalibrator(0.995)

    calibration_end = dataset.calibration_end
    fit_end = int(calibration_end * fit_fraction_of_calibration)
    if fit_end < 2 or fit_end >= calibration_end:
        raise ValueError("calibration split is too small")

    fit_values = dataset.values[:fit_end]
    threshold_values = dataset.values[fit_end:calibration_end]
    evaluation_values = dataset.values[calibration_end:]
    evaluation_labels = dataset.labels[calibration_end:]
    evaluation_events = dataset.evaluation_events()

    records: list[BenchmarkRecord] = []
    for detector in active_detectors:
        # Phase 1: learn detector state from an earlier normal-only segment and
        # estimate a threshold on a later normal-only holdout.
        detector.fit(fit_values)
        calibration_scores = detector.score(threshold_values)
        calibration = active_calibrator.fit(calibration_scores)

        # Phase 2: after the threshold is frozen, use all clean pre-evaluation
        # history to establish the best state available at deployment time.
        detector.fit(dataset.values[:calibration_end])
        evaluation_scores = detector.score(evaluation_values)
        predictions = evaluation_scores >= calibration.threshold
        detection = DetectionResult(
            detector=detector.name,
            scores=evaluation_scores,
            threshold=calibration.threshold,
            predictions=predictions,
        )
        report = evaluate_detection(
            evaluation_labels,
            detection.predictions,
            evaluation_events,
            tolerance=event_tolerance,
        )
        records.append(
            BenchmarkRecord(
                detector=detector.name,
                threshold=calibration.threshold,
                calibration_method=calibration.method,
                calibration_samples=calibration.samples,
                fit_samples=len(fit_values),
                evaluation_samples=len(evaluation_values),
                score_summary=ScoreSummary.from_scores(evaluation_scores),
                evaluation=report,
            )
        )

    return BenchmarkReport(
        dataset=dataset.name,
        total_samples=len(dataset.values),
        calibration_end=calibration_end,
        fit_end=fit_end,
        records=tuple(records),
    )
