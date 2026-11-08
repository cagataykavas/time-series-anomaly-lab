# Pipeline engineering audit

Public technical audit, November 2026. Synthetic measurements only.

## Actual code defects and changes

- `SlidingWindowDataset` builds N-W windows, omitting the final valid window.
  The new default uses N-W+1. A compatibility mode retains N-W for speed-only
  comparison, with exact equality across every original window.
- The original eager Python loop copies all windows into a stacked tensor.
  A strided tensor view now retains just the original series storage. The
  reported storage metric counts tensor backing bytes, not process RSS.
- Original inference performs one model call per window. The new path batches
  256 windows and preallocates the result array.
- The original VAE `eval()` still samples latent noise. New evaluation uses
  posterior means while training remains stochastic. This is a documented
  semantic choice, not a demonstrated improvement in anomaly detection.
  Serial/batched speed and parity comparisons both use the same mean-mode model.
- Original GraphVAE's univariate mean broadcast expands reconstruction shape,
  and its returned target is a hidden vector instead of the input window.
  The repair returns a correctly shaped reconstruction and true window target.
- Original StackVAE passes window length 50 to a Linear expecting 1 and has no
  loss function. The repair converts to (batch,window,features), preserves the
  true input target, and supplies the reconstruction+KL objective. Repairs are
  explicitly restricted to the baseline univariate contract.
- Original overlap aggregation loops over every window. The new implementation
  loops over 50 offsets with vectorized accumulation. Floating-point order can
  differ; tests use an explicit 1e-12 bound. Uncovered positions become NaN rather
  than a fabricated zero. The corrected inclusive windowing covers the tail.
- Original workflow trains, sets its threshold and scores the same series.
  We provide a separate frozen-calibration threshold and window-end score helper.
  This is a safer primitive, not retrospective evidence of historical holdout use.

## Other limitations not silently repaired

`kl_warmup` is computed but not used in the baseline training objective.
Graph/Stack variants with a single feature do not demonstrate meaningful
multivariate graph learning. Loss values differ by model/objective and cannot be
ranked as interchangeable detection performance. Mean-centering reconstruction
using the input window's mean changes anomaly sensitivity. Naming variants after
papers does not establish faithful implementation of those papers.

## Reproduction

From this directory, using the captured environment:

```
MPLBACKEND=Agg python -m pytest test_pipeline.py -q
ruff check optimized.py benchmark.py test_pipeline.py
MPLBACKEND=Agg python benchmark.py
```

The benchmark warms both paths, alternates order, records each timing rather than
only a selected run, and asserts matching compatibility windows and numerical
inference/aggregation parity. Timing is from a shared CPU host with one Torch
thread. Training smoke tests use 512 synthetic samples, one epoch/eight batches:
finite loss and successful backward steps do not establish convergence.

## Completed correctness harness

Twenty-four tests pass. The integrated `run_chronological` reference uses a
60/20/20 chronological split, normalizes using only the fit partition, makes
windows independently within each partition, and freezes a calibration-only
99.5th-percentile threshold. A regression test changes only the test suffix and
checks identical normalization, training losses and threshold. All seven model
classes also complete this end-to-end reference smoke.

The inference timing model is randomly initialized: its weights are identical
across serial and batched calls, but it is not a baseline trained checkpoint.
The experiment therefore measures implementation throughput and numerical parity,
not the quality of a production model. Setup uses nine paired repetitions,
inference three, aggregation seven, after warming both paths. The legacy
comparison covers 72,885 windows; the corrected default includes 72,886 windows.
