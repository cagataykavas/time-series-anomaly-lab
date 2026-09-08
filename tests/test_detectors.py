from __future__ import annotations

import numpy as np
import pytest

from anomaly_lab.detectors import (
    AdaptiveEWMADetector,
    CausalRollingMADDetector,
    PCAWindowReconstructionDetector,
)
from anomaly_lab.synthetic import generate_synthetic_series


@pytest.mark.parametrize(
    "detector",
    [
        CausalRollingMADDetector(window=51, min_history=15),
        AdaptiveEWMADetector(alpha=0.05),
        PCAWindowReconstructionDetector(window=24, components=4),
    ],
)
def test_detectors_return_finite_aligned_scores(detector) -> None:
    dataset = generate_synthetic_series(n=1400, seed=11)
    fit = dataset.values[:300]
    evaluation = dataset.values[300:500]

    detector.fit(fit)
    scores = detector.score(evaluation)

    assert scores.shape == evaluation.shape
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CausalRollingMADDetector(window=51, min_history=15),
        lambda: AdaptiveEWMADetector(alpha=0.05),
        lambda: PCAWindowReconstructionDetector(window=24, components=4),
    ],
)
def test_scores_do_not_depend_on_future_evaluation_samples(factory) -> None:
    """A prefix must score identically whether or not future samples are supplied."""
    dataset = generate_synthetic_series(n=1400, seed=5)
    fit = dataset.values[:300]
    prefix = dataset.values[300:380]
    longer = dataset.values[300:460]

    detector_short = factory().fit(fit)
    detector_long = factory().fit(fit)

    short_scores = detector_short.score(prefix)
    long_scores = detector_long.score(longer)

    np.testing.assert_allclose(short_scores, long_scores[: len(prefix)], rtol=1e-10, atol=1e-10)


def test_pca_requires_enough_fit_history() -> None:
    detector = PCAWindowReconstructionDetector(window=32, components=3)
    with pytest.raises(ValueError, match="shorter"):
        detector.fit(np.arange(20, dtype=float))
