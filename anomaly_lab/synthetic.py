from __future__ import annotations

import numpy as np

from anomaly_lab.domain import AnomalyEvent, TimeSeriesDataset


def _mark(labels: np.ndarray, event: AnomalyEvent) -> None:
    labels[event.start : event.end] = True


def generate_synthetic_series(
    *,
    n: int = 5000,
    seed: int = 42,
    calibration_fraction: float = 0.35,
) -> TimeSeriesDataset:
    """Generate a deterministic, public-safe benchmark with mixed anomaly types.

    The leading calibration region is deliberately anomaly-free. Spikes, a level
    shift, and a progressive drift are injected only after that boundary so an
    experiment can fit and calibrate thresholds without looking at evaluation labels.
    """
    if n < 1200:
        raise ValueError("n must be at least 1200")
    if not 0.2 <= calibration_fraction <= 0.6:
        raise ValueError("calibration_fraction must be between 0.2 and 0.6")

    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0.0, 48.0, n)
    baseline = (
        0.75 * np.sin(time_axis)
        + 0.22 * np.sin(3.3 * time_axis + 0.3)
        + 0.10 * np.cos(0.19 * time_axis)
    )
    noise_scale = 0.10 + 0.025 * (1.0 + np.sin(0.11 * time_axis))
    values = baseline + rng.normal(0.0, noise_scale, n)
    labels = np.zeros(n, dtype=bool)
    calibration_end = int(n * calibration_fraction)

    level_start = max(calibration_end + 100, int(n * 0.53))
    level_end = min(n, level_start + max(45, int(n * 0.045)))
    level_event = AnomalyEvent(level_start, level_end, "level_shift")
    values[level_start:level_end] += 1.25
    _mark(labels, level_event)

    drift_start = max(level_end + 150, int(n * 0.75))
    drift_end = min(n, drift_start + max(80, int(n * 0.08)))
    drift_values = np.linspace(0.0, 1.9, drift_end - drift_start)
    values[drift_start:drift_end] += drift_values
    drift_label_start = drift_start + max(1, (drift_end - drift_start) // 3)
    drift_event = AnomalyEvent(drift_label_start, drift_end, "slow_drift")
    _mark(labels, drift_event)

    blocked = np.zeros(n, dtype=bool)
    blocked[max(0, level_start - 30) : min(n, level_end + 30)] = True
    blocked[max(0, drift_start - 30) : min(n, drift_end + 30)] = True
    eligible = np.flatnonzero(
        (~blocked)
        & (np.arange(n) >= calibration_end + 60)
        & (np.arange(n) < n - 40)
    )
    spike_count = max(8, n // 300)
    spike_indices = np.sort(rng.choice(eligible, size=spike_count, replace=False))
    spike_events: list[AnomalyEvent] = []
    for index in spike_indices:
        width = int(rng.integers(1, 4))
        end = min(n, index + width)
        sign = float(rng.choice([-1.0, 1.0]))
        magnitude = float(rng.uniform(2.0, 3.4))
        values[index:end] += sign * magnitude
        event = AnomalyEvent(int(index), int(end), "spike")
        spike_events.append(event)
        _mark(labels, event)

    events = tuple(sorted((*spike_events, level_event, drift_event), key=lambda item: item.start))
    return TimeSeriesDataset(
        values=values.astype(np.float64),
        labels=labels,
        events=events,
        calibration_end=calibration_end,
        name="mixed-synthetic-v1",
    )
