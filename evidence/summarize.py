"""Recompute descriptive seed summaries from retained neural benchmark outputs."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
runs = [json.loads((root / f'seed{seed}/nab_vae_benchmark.json').read_text())
        for seed in (42, 43, 44)]
summary = []
for model in runs[0]['models']:
    rows = [next(x for x in run['results']
                 if x['model'] == model and x['threshold_quantile'] == .995) for run in runs]
    row = {'model': model, 'seeds': [42, 43, 44], 'epochs_per_fit': 30,
           'events_detected_with_tolerance': sum(x['events_detected'] for x in rows),
           'strict_events_detected': sum(x['strict_events_detected'] for x in rows),
           'repeated_event_opportunities': sum(x['labeled_events'] for x in rows)}
    for metric in ('average_precision', 'f1', 'false_positive_runs',
                   'mean_strict_detection_delay_samples'):
        vals = [x[metric] for x in rows]
        row[metric] = {'mean': float(np.mean(vals)), 'sample_sd': float(np.std(vals, ddof=1)),
                       'values': vals}
    summary.append(row)
(root / 'multiseed_summary.json').write_text(json.dumps({
    'note': 'Descriptive seed variability on ONE series; 15 opportunities are repeated evaluations '
            'of 5 events, not 15 independent events. No selection holdout.',
    'results': summary}, indent=2))
print(json.dumps(summary, indent=2))
