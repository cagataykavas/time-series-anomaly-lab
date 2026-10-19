from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from anomaly_lab.drift import DriftPolicy, audit_calibration_drift


def _audit(baseline: np.ndarray, current: np.ndarray, **kwargs):
    return audit_calibration_drift(
        baseline,
        current,
        threshold=kwargs.pop("threshold", 2.35),
        detector_revision="causal-mad:v2",
        baseline_window="calibration:2026-09",
        current_window="monitor:2026-10",
        policy=kwargs.pop("policy", None),
        **kwargs,
    )


def test_stable_window_is_accepted_with_deterministic_evidence() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(size=2_000)
    current = rng.normal(size=1_000)

    first = _audit(baseline, current)
    second = _audit(baseline[::-1], current[::-1])

    assert first.accepted
    assert first.reasons == ()
    assert first.evidence_sha256 == second.evidence_sha256
    assert first.baseline_scores_sha256 == second.baseline_scores_sha256
    assert first.current_scores_sha256 == second.current_scores_sha256


def test_location_shift_fails_distribution_and_threshold_gates() -> None:
    rng = np.random.default_rng(11)
    baseline = rng.normal(size=2_000)
    current = rng.normal(loc=1.5, size=1_000)

    report = _audit(baseline, current)

    assert not report.accepted
    assert "score_distribution_drift" in report.reasons
    assert "current_exceedance_rate_too_high" in report.reasons
    assert "exceedance_rate_increase_too_high" in report.reasons


def test_tail_inflation_can_fail_exceedance_gate_without_large_ks_shift() -> None:
    rng = np.random.default_rng(23)
    baseline = rng.normal(size=5_000)
    current = rng.normal(size=5_000)
    current[:150] = 3.0
    policy = DriftPolicy(
        max_ks_statistic=0.20,
        max_current_exceedance_rate=0.02,
        max_exceedance_rate_increase=0.01,
    )

    report = _audit(baseline, current, policy=policy)

    assert report.ks_statistic < policy.max_ks_statistic
    assert "score_distribution_drift" not in report.reasons
    assert "current_exceedance_rate_too_high" in report.reasons
    assert "exceedance_rate_increase_too_high" in report.reasons


@pytest.mark.parametrize(
    ("baseline", "current", "message"),
    [
        (np.ones((10, 10)), np.ones(100), "one-dimensional"),
        (np.ones(99), np.ones(100), "between 100 and"),
        (np.ones(100), np.r_[np.ones(99), np.nan], "finite"),
    ],
)
def test_malformed_windows_fail_closed(
    baseline: np.ndarray, current: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _audit(baseline, current)


def test_lineage_identifiers_and_distinct_windows_are_required() -> None:
    values = np.linspace(0.0, 1.0, 100)

    with pytest.raises(ValueError, match="detector_revision"):
        audit_calibration_drift(
            values,
            values,
            threshold=0.9,
            detector_revision="bad revision",
            baseline_window="baseline:v1",
            current_window="current:v1",
        )
    with pytest.raises(ValueError, match="must differ"):
        audit_calibration_drift(
            values,
            values,
            threshold=0.9,
            detector_revision="detector:v1",
            baseline_window="same:v1",
            current_window="same:v1",
        )


def test_policy_rejects_invalid_or_inconsistent_limits() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        DriftPolicy(min_current_samples=1)
    with pytest.raises(ValueError, match="smaller"):
        DriftPolicy(min_baseline_samples=200, max_samples_per_window=100)
    with pytest.raises(ValueError, match="max_ks_statistic"):
        replace(DriftPolicy(), max_ks_statistic=float("nan"))


def test_report_contains_no_raw_scores_and_changes_when_evidence_changes() -> None:
    baseline = np.linspace(-2.0, 2.0, 200)
    current = np.linspace(-2.0, 2.0, 100)
    changed = current.copy()
    changed[0] += 0.001

    report = _audit(baseline, current, threshold=1.9)
    changed_report = _audit(baseline, changed, threshold=1.9)
    payload = report.to_dict()

    assert "baseline_scores" not in payload
    assert "current_scores" not in payload
    assert len(report.evidence_sha256) == 64
    assert report.evidence_sha256 != changed_report.evidence_sha256
