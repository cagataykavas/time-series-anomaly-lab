from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    scores: np.ndarray
    threshold: float
    predictions: np.ndarray


def make_synthetic_series(n: int = 4000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40, n)
    signal = 0.8 * np.sin(t) + 0.25 * np.sin(3.1 * t) + rng.normal(0, 0.12, n)
    labels = np.zeros(n, dtype=bool)

    # Spikes
    spike_idx = rng.choice(np.arange(200, n - 200), size=18, replace=False)
    signal[spike_idx] += rng.choice([-1, 1], size=len(spike_idx)) * rng.uniform(2.0, 3.5, len(spike_idx))
    labels[spike_idx] = True

    # Level shift
    signal[1900:2050] += 1.4
    labels[1900:2050] = True

    # Slow drift
    drift = np.linspace(0, 1.8, 180)
    signal[3000:3180] += drift
    labels[3060:3180] = True

    return signal.astype(np.float32), labels


def rolling_robust_score(values: np.ndarray, window: int = 101) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scores = np.zeros_like(values)
    half = window // 2

    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        local = values[lo:hi]
        median = np.median(local)
        mad = np.median(np.abs(local - median))
        scale = 1.4826 * mad + 1e-8
        scores[i] = abs(values[i] - median) / scale

    return scores.astype(np.float32)


def calibrate_quantile(scores: np.ndarray, quantile: float = 0.995) -> DetectionResult:
    threshold = float(np.quantile(scores, quantile))
    predictions = scores >= threshold
    return DetectionResult(scores=scores, threshold=threshold, predictions=predictions)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    series, labels = make_synthetic_series()
    scores = rolling_robust_score(series)
    result = calibrate_quantile(scores)
    report = metrics(labels, result.predictions)

    print(f"threshold={result.threshold:.3f}")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}={value:.4f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
