# Retrospective pipeline engineering benchmark

**November 2026 retrospective experiment. Synthetic data, randomly initialized weights, CPU only. This is not historical production impact or anomaly-detection accuracy.**

This self-contained evidence package compares generic legacy univariate VAE pipeline code with a repaired and optimized implementation. It is separate from the repository's labeled anomaly-detection benchmark. The baseline and comparison source bytes are preserved to match the hashes in the recorded measurement. The `Private revision` docstring in `optimized.py` describes its preparation stage, not a restriction on this published package.

## Recorded measurements

`benchmark_results.json` is the original recorded run, not a forecast or a guarantee on other hardware:

- Serial versus batched posterior-mean inference: **5.851176 s → 0.164750 s**, ratio of medians **35.52×**, three paired repetitions after warmup.
- Window tensor backing storage: **14,868,740 → 291,740 bytes**, **98.04% reduction**. This is not total process RAM, peak RSS, or GPU memory.
- Window construction: 345.83×; overlap aggregation: 16.11× in this run.
- Same randomly initialized VanillaVAE weights, posterior-mean evaluation in both inference paths; maximum absolute output difference **2.3842e-7**. The legacy code normally samples latent noise even in evaluation, so this comparison explicitly changes that policy on both paths to isolate batching.
- Seven model smoke tests: baseline **5/7**, repaired **7/7**. This tests finite-loss training/backward execution, not convergence, correctness against published architectures, or predictive quality.
- **24 regression tests** cover compatibility windows, final-window inclusion, model execution, batched parity, overlap aggregation, frozen calibration and chronological split isolation.

The deterministic workload has 72,935 synthetic univariate samples, a 50-point window, NumPy seed 2026, Torch seed 42 and one Torch thread. Speed comparisons retain the baseline's 72,885 windows. The corrected default includes all 72,886 valid windows. Training smoke tests use 512 synthetic samples, one epoch/eight batches. No real-world dataset or trained checkpoint is bundled.

## Reproduce

Use Python 3.12 and a CPU environment. The exact directly used package versions are in `requirements.txt`; the full recorded package listing is in `environment.txt` (an unrelated local editable installation was omitted).

```sh
cd projects/retrospective-pipeline
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m pytest test_pipeline.py -q
ruff check optimized.py benchmark.py test_pipeline.py
MPLBACKEND=Agg python benchmark.py
```

`benchmark.py` overwrites `benchmark_results.json` with the new run and prints every measured repetition. Preserve a copy first if you want to compare with the recorded run. Timing varies on shared CPU hosts. `publication_rerun.json` is a separate pre-publication rerun, not a replacement for the original 35.52× record. SHA256.json records published file hashes; each benchmark result also records the exact three measured source hashes. No model download, cloud service or paid resource is needed.

## What changed and what remains limited

See [AUDIT.md](AUDIT.md) for defects, repairs, semantic changes, and limitations. The optimized chronological reference fits normalization on the first 60%, calibrates a frozen 99.5th percentile threshold on the next 20%, and evaluates the final 20%, making windows separately in each partition. This is a new tested reference, not evidence of a historical deployment's validation protocol.

Do not turn the throughput result into a claim about production savings, deployed flight data, anomaly accuracy/F1, or improved model quality. Graph/Stack repairs support only the documented univariate shape. Model names do not certify faithful reproductions of their namesake papers.

## Evidence mapping

- Implementation and defects: `original.py`, `optimized.py`, `AUDIT.md`.
- Experiment protocol and all raw timings: `benchmark.py`, `benchmark_results.json`, `publication_rerun.json`.
- Correctness checks: `test_pipeline.py`, `tests.log`, `lint.log`.
- Reproducibility and integrity: `requirements.txt`, `environment.txt`, `SHA256.json`.

Suitable CV wording: “In a November 2026 retrospective synthetic CPU benchmark, optimized a univariate VAE pipeline to achieve 35.52× batched posterior-mean inference throughput and 98.04% lower window tensor storage; validated with 24 regression tests.”

Core-only repository CI does not install the optional neural/plotting dependencies. The local `conftest.py` excludes this evidence suite when those dependencies are absent; it is not evidence that these 24 tests passed in core CI. Install this directory's requirements and run the explicit command above to execute them. The local Ruff configuration applies the audit's focused lint rules and exempts the byte-preserved baseline; baseline lint cleanliness is not claimed.
