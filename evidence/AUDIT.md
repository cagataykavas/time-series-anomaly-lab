# Anomaly benchmark audit

Baseline: `265fcbea29d542894f61de6a0a34a9d04b951727` on
https://github.com/cagataykavas/time-series-anomaly-lab

## Implemented changes

- Kept fitted model parameters fixed after threshold calibration. The prior core
  benchmark refit on fit+calibration history, changing PCA normalization/basis
  without recalibrating the threshold. It now replays holdout context using the
  original fit checkpoint and takes only evaluation scores. Regression tests
  assert one fit, contiguous PCA scoring, and threshold invariance to future data.
- Added non-interpolated, tie-grouped average precision with regression tests.
- Neural output now retains every calibration/evaluation score and evaluation
  label in per-model NPZ files; hashes input files and records software versions.
- Added strict event detection and strict mean/max delay, alongside existing
  explicitly labeled three-sample tolerance metrics.

## Commands

From repository root, with the evidence environment active:

```
python -m pytest -q
ruff check .
anomaly-lab benchmark --samples 5000 --seed 42 --output evidence/synthetic_fixed.json
python scripts/benchmark_vae_variants.py --epochs 30 --seed 42 --output-dir evidence/seed42
python scripts/benchmark_vae_variants.py --epochs 30 --seed 43 --output-dir evidence/seed43
python scripts/benchmark_vae_variants.py --epochs 30 --seed 44 --output-dir evidence/seed44
python evidence/summarize.py
```

## Interpretation boundaries

- All NAB results concern one public NYC Taxi series, not TUSAŞ or flight-test data.
- Models are named-family reimplementations; Transformer baseline is inspired,
  not official TiTAD. No claim to reproduce original research implementations.
- Split assumes known clean historical data; the first labeled anomaly determines
  the fit/calibration boundary. Training normalization uses fit only. Neural
  checkpoint and quantile thresholds remain fixed during evaluation.
- There are five labeled event windows. Repeating three seeds yields fifteen
  event evaluation opportunities, not fifteen independent events.
- Delay is measured from NAB label-window start, not exact real incident onset.
  Each sample represents 30 minutes. Tolerated detection allows +/- 3 samples.
- False-positive runs in the neural script count normal-region alarm fragments;
  the core synthetic evaluator counts whole alarm runs that never overlap an
  event. These semantics differ and must not be combined as the same measure.
- Seed variability is descriptive. Ranking on the same test series does not
  establish unbiased model selection or cross-dataset generalization.
- Benchmark results do not demonstrate a live production deployment.
- Prior unverified AP/multi-seed claims are excluded. The evidence here comes
  from the recorded reproducible runs and retained score artifacts.

## Verified results

All 24 fits completed at 30 epochs for seeds 42, 43 and 44. Each uses 3,795
fit observations, 2,044 calibration observations and 4,481 evaluation observations
from 10,320 half-hour observations; fit creates 3,746 length-50 windows.

For the Transformer-inspired baseline at the frozen 99.5th percentile:

- AP: 0.478317 mean, 0.051544 sample standard deviation.
- Point F1: 0.399294 mean, 0.084052 sample standard deviation.
- Strict detections: 13/15 repeated opportunities over five unique event windows.
  The +/-3 sample tolerance count is also 13/15 for these runs.
- False-positive fragments/runs: 4.666667 mean, 1.527525 sample standard deviation.
- Mean of per-seed mean strict delays: 100.233333 samples (50.116667 hours).
  This is a long delay and precludes a claim of fast incident detection.

`python -m evidence.verify_scores` independently recomputed 48 metric rows from
24 retained score files and checked input hashes. `pytest -q`: 28 passed.
`ruff check .`: passed. Dependency consistency: passed.

Suggested CV bullet:

> Benchmarked eight neural reconstruction variants across 24 seeded training
> runs on 10,320 public taxi-demand observations, implementing chronological
> calibration, average-precision/event-delay evaluation and auditable score artifacts.

At the pre-publication validation snapshot, these local evidence and code changes
had not been pushed, merged or validated by remote CI. Validation branch: `codex/cv-anomaly-evidence-audit`.
