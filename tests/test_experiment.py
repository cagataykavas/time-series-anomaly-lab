from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anomaly_lab.calibration import QuantileCalibrator
from anomaly_lab.experiment import run_benchmark
from anomaly_lab.synthetic import generate_synthetic_series


@dataclass
class SpyDetector:
    name: str = "spy"

    def __post_init__(self) -> None:
        self.fit_lengths: list[int] = []
        self.score_lengths: list[int] = []
        self._center = 0.0

    def fit(self, values: np.ndarray):
        self.fit_lengths.append(len(values))
        self._center = float(np.mean(values))
        return self

    def score(self, values: np.ndarray) -> np.ndarray:
        self.score_lengths.append(len(values))
        return np.abs(np.asarray(values, dtype=float) - self._center)


def test_synthetic_calibration_region_is_clean() -> None:
    dataset = generate_synthetic_series(n=1600, seed=17)

    assert not np.any(dataset.labels[: dataset.calibration_end])
    assert all(event.start >= dataset.calibration_end for event in dataset.events)
    assert np.any(dataset.labels[dataset.calibration_end :])


def test_benchmark_uses_normal_holdout_then_refits_before_evaluation() -> None:
    dataset = generate_synthetic_series(n=1600, seed=3)
    detector = SpyDetector()

    report = run_benchmark(
        dataset,
        detectors=(detector,),
        calibrator=QuantileCalibrator(0.99),
        fit_fraction_of_calibration=0.6,
    )

    expected_fit_end = int(dataset.calibration_end * 0.6)
    assert report.fit_end == expected_fit_end
    assert detector.fit_lengths == [expected_fit_end, dataset.calibration_end]
    assert detector.score_lengths == [
        dataset.calibration_end - expected_fit_end,
        len(dataset.values) - dataset.calibration_end,
    ]
    assert len(report.records) == 1
    assert report.records[0].calibration_samples == dataset.calibration_end - expected_fit_end


def test_default_benchmark_runs_multiple_detector_families() -> None:
    dataset = generate_synthetic_series(n=1600, seed=9)
    report = run_benchmark(dataset, calibrator=QuantileCalibrator(0.99))

    names = {record.detector for record in report.records}
    assert names == {
        "causal_rolling_mad",
        "adaptive_ewma",
        "pca_window_reconstruction",
    }
    for record in report.records:
        assert record.evaluation_samples == len(dataset.values) - dataset.calibration_end
        assert 0.0 <= record.evaluation.point.precision <= 1.0
        assert 0.0 <= record.evaluation.point.recall <= 1.0
        assert 0.0 <= record.evaluation.event.event_recall <= 1.0
        assert np.isfinite(record.threshold)
