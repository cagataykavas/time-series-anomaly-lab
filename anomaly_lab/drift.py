"""Fail-closed monitoring for anomaly-score calibration drift."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass

import numpy as np

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class DriftPolicy:
    """Release limits for a score-distribution monitoring window."""

    min_baseline_samples: int = 100
    min_current_samples: int = 50
    max_samples_per_window: int = 100_000
    max_ks_statistic: float = 0.20
    max_current_exceedance_rate: float = 0.05
    max_exceedance_rate_increase: float = 0.02

    def __post_init__(self) -> None:
        if self.min_baseline_samples < 2 or self.min_current_samples < 2:
            raise ValueError("minimum sample counts must be at least 2")
        if self.max_samples_per_window < max(self.min_baseline_samples, self.min_current_samples):
            raise ValueError("max_samples_per_window is smaller than a minimum sample count")
        for name, value in (
            ("max_ks_statistic", self.max_ks_statistic),
            ("max_current_exceedance_rate", self.max_current_exceedance_rate),
            ("max_exceedance_rate_increase", self.max_exceedance_rate_increase),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class DriftReport:
    decision: str
    reasons: tuple[str, ...]
    detector_revision: str
    baseline_window: str
    current_window: str
    threshold: float
    baseline_samples: int
    current_samples: int
    baseline_exceedance_rate: float
    current_exceedance_rate: float
    exceedance_rate_increase: float
    ks_statistic: float
    baseline_scores_sha256: str
    current_scores_sha256: str
    policy_sha256: str
    evidence_sha256: str

    @property
    def accepted(self) -> bool:
        return self.decision == "accepted"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_calibration_drift(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    threshold: float,
    detector_revision: str,
    baseline_window: str,
    current_window: str,
    policy: DriftPolicy | None = None,
) -> DriftReport:
    """Compare a reviewed-normal window with the original calibration distribution.

    The audit is intentionally fail closed. Callers should quarantine or review a
    rejected window; rejection is not permission to recalibrate automatically.
    """

    active_policy = policy or DriftPolicy()
    baseline = _validated_scores(
        baseline_scores,
        name="baseline_scores",
        minimum=active_policy.min_baseline_samples,
        maximum=active_policy.max_samples_per_window,
    )
    current = _validated_scores(
        current_scores,
        name="current_scores",
        minimum=active_policy.min_current_samples,
        maximum=active_policy.max_samples_per_window,
    )
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    for name, value in (
        ("detector_revision", detector_revision),
        ("baseline_window", baseline_window),
        ("current_window", current_window),
    ):
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"{name} must be a bounded stable identifier")
    if baseline_window == current_window:
        raise ValueError("baseline_window and current_window must differ")

    baseline_rate = float(np.mean(baseline >= threshold))
    current_rate = float(np.mean(current >= threshold))
    rate_increase = current_rate - baseline_rate
    ks_statistic = _two_sample_ks(baseline, current)

    reasons: list[str] = []
    if ks_statistic > active_policy.max_ks_statistic:
        reasons.append("score_distribution_drift")
    if current_rate > active_policy.max_current_exceedance_rate:
        reasons.append("current_exceedance_rate_too_high")
    if rate_increase > active_policy.max_exceedance_rate_increase:
        reasons.append("exceedance_rate_increase_too_high")

    policy_payload = asdict(active_policy)
    policy_digest = _json_digest(policy_payload)
    evidence = {
        "schema_version": 1,
        "decision": "rejected" if reasons else "accepted",
        "reasons": reasons,
        "detector_revision": detector_revision,
        "baseline_window": baseline_window,
        "current_window": current_window,
        "threshold": threshold,
        "baseline_samples": len(baseline),
        "current_samples": len(current),
        "baseline_exceedance_rate": baseline_rate,
        "current_exceedance_rate": current_rate,
        "exceedance_rate_increase": rate_increase,
        "ks_statistic": ks_statistic,
        "baseline_scores_sha256": _score_digest(baseline),
        "current_scores_sha256": _score_digest(current),
        "policy_sha256": policy_digest,
    }
    return DriftReport(
        decision=str(evidence["decision"]),
        reasons=tuple(reasons),
        detector_revision=detector_revision,
        baseline_window=baseline_window,
        current_window=current_window,
        threshold=threshold,
        baseline_samples=len(baseline),
        current_samples=len(current),
        baseline_exceedance_rate=baseline_rate,
        current_exceedance_rate=current_rate,
        exceedance_rate_increase=rate_increase,
        ks_statistic=ks_statistic,
        baseline_scores_sha256=str(evidence["baseline_scores_sha256"]),
        current_scores_sha256=str(evidence["current_scores_sha256"]),
        policy_sha256=policy_digest,
        evidence_sha256=_json_digest(evidence),
    )


def _validated_scores(scores: np.ndarray, *, name: str, minimum: int, maximum: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not minimum <= len(values) <= maximum:
        raise ValueError(f"{name} must contain between {minimum} and {maximum} samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _two_sample_ks(left: np.ndarray, right: np.ndarray) -> float:
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    support = np.unique(np.concatenate((left_sorted, right_sorted)))
    left_cdf = np.searchsorted(left_sorted, support, side="right") / len(left_sorted)
    right_cdf = np.searchsorted(right_sorted, support, side="right") / len(right_sorted)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _score_digest(scores: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack(">Q", len(scores)))
    # Distribution monitoring is order-insensitive, so the digest binds the score
    # multiset rather than incidental ingestion order.
    for value in np.sort(scores):
        digest.update(struct.pack(">d", float(value)))
    return digest.hexdigest()


def _json_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
