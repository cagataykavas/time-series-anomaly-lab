"""Independently verify every published metric against saved score arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np

from anomaly_lab.evaluation import average_precision, point_metrics
from scripts.benchmark_vae_variants import count_runs, load_nab

root = Path(__file__).parent
_, _, events, _ = load_nab()
boundary = events[0][0]
checked = 0
for seed in (42, 43, 44):
    folder = root / f'seed{seed}'
    report = json.loads((folder / 'nab_vae_benchmark.json').read_text())
    for filename, expected in report['source_sha256'].items():
        payload = (root.parent / 'data/benchmark' / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
    for i, model in enumerate(report['models']):
        arrays = np.load(folder / f'scores_{i}.npz')
        for row in (r for r in report['results'] if r['model'] == model):
            scores, labels = arrays['evaluation'], arrays['labels']
            threshold = np.quantile(arrays['calibration'], row['threshold_quantile'])
            assert threshold == row['threshold']
            predicted = scores >= threshold
            assert average_precision(labels, scores) == row['average_precision']
            point = point_metrics(labels, predicted)
            for key in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'):
                assert getattr(point, key) == row[key]
            mask = np.zeros(len(scores), dtype=bool)
            hits, delays = 0, []
            for start, end in events:
                start, end = start-boundary, end-boundary
                lo, hi = max(0, start-3), min(len(scores), end+3)
                mask[lo:hi] = True
                hits += bool(predicted[lo:hi].any())
                positions = np.flatnonzero(predicted[start:end])
                if len(positions):
                    delays.append(int(positions[0]))
            assert hits == row['events_detected']
            assert count_runs(predicted & ~mask) == row['false_positive_runs']
            assert len(delays) == row['strict_events_detected']
            assert float(np.mean(delays)) == row['mean_strict_detection_delay_samples']
            checked += 1
print(f'Verified {checked} metric rows against 24 retained score files and input hashes.')
