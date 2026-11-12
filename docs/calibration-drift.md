# Calibration-drift admission gate

An anomaly threshold is calibrated against a particular detector revision and score
distribution. Production data can move while the threshold remains numerically unchanged,
silently turning a useful detector into an alert flood or a blind spot.

`anomaly_lab.drift.audit_calibration_drift` compares the original normal-only calibration
scores with a later **reviewed-normal** monitoring window. It rejects the window when any
pre-registered policy limit is breached:

- two-sample Kolmogorov-Smirnov distance for broad distribution movement;
- absolute frozen-threshold exceedance rate;
- increase in exceedance rate relative to the calibration baseline.

The KS check and exceedance checks are complementary: a small tail can cross the operational
threshold without creating a large whole-distribution distance.

```python
from anomaly_lab.drift import DriftPolicy, audit_calibration_drift

report = audit_calibration_drift(
    baseline_scores,
    current_scores,
    threshold=frozen_threshold,
    detector_revision="pca-window:v4",
    baseline_window="calibration:2026-09",
    current_window="reviewed-normal:2026-10",
    policy=DriftPolicy(
        max_ks_statistic=0.15,
        max_current_exceedance_rate=0.03,
        max_exceedance_rate_increase=0.01,
    ),
)

if not report.accepted:
    quarantine_for_review(report.reasons)
```

## Evidence and privacy boundary

The report contains sample counts, aggregate rates, stable reason codes, policy identity, and
SHA-256 identities for the score multisets. Raw scores are not serialized. The evidence digest
is deterministic across ingestion-order changes and binds the detector revision, window IDs,
threshold, metrics, policy, and score identities.

The digests identify content; they do not authenticate the producer. Persist the report in a
trusted audit store or sign it when producer authenticity matters.

## Operational boundary

- Run this only on a window known or assumed to represent normal operation. Unreviewed true
  incidents can correctly trigger rejection but cannot distinguish concept drift from attack or
  failure.
- Choose policy limits before evaluating the monitoring window. The defaults are examples, not
  domain guarantees.
- A rejection should freeze promotion, quarantine the window, and trigger investigation. It is
  **not** permission to recalibrate automatically on possibly contaminated data.
- The gate detects distribution change; it does not prove causal model degradation, label
  quality, downstream alert usefulness, or safe retraining.

## Next integration step

Persist accepted calibration evidence alongside the benchmark artifact, then run this audit on
scheduled reviewed-normal windows before allowing threshold or detector promotion.
