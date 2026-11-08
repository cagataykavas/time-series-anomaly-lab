import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import optimized as new
import original as old

torch.set_num_threads(1)
ROOT = Path(__file__).parent
rng = np.random.default_rng(2026)
x = (
    np.sin(np.arange(72935, dtype=np.float64) / 50) + rng.normal(0, 0.05, 72935)
).astype(np.float32)
ws = 50


def elapsed(fn):
    gc.collect()
    begin = time.perf_counter()
    value = fn()
    return time.perf_counter() - begin, value


def paired(a, b, repetitions):
    # Both warmed once; alternate order to reduce order/load bias.
    a()
    b()
    aa = []
    bb = []
    for i in range(repetitions):
        if i % 2:
            tb, _ = elapsed(b)
            ta, _ = elapsed(a)
        else:
            ta, _ = elapsed(a)
            tb, _ = elapsed(b)
        aa.append(ta)
        bb.append(tb)
    return {
        "baseline_seconds": aa,
        "optimized_seconds": bb,
        "median_speedup": float(np.median(aa) / np.median(bb)),
        "baseline_median_seconds": float(np.median(aa)),
        "optimized_median_seconds": float(np.median(bb)),
    }


result = {
    "input": {
        "samples": len(x),
        "window": ws,
        "seed": 2026,
        "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "kind": "synthetic matched-shape signal, not company data",
    },
    "runtime": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": 1,
        "processor": platform.processor(),
    },
    "source_sha256": {
        f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        for f in ["original.py", "optimized.py", "benchmark.py"]
    },
}
result["window_setup"] = paired(
    lambda: old.SlidingWindowDataset(x, ws),
    lambda: new.SlidingWindowDataset(x, ws, legacy_drop_final=True),
    9,
)
a = old.SlidingWindowDataset(x, ws)
b = new.SlidingWindowDataset(x, ws, legacy_drop_final=True)
assert torch.equal(a.samples, b.samples)
result["window_memory"] = {
    "baseline_tensor_storage_bytes": a.series.untyped_storage().nbytes()
    + a.samples.untyped_storage().nbytes(),
    "optimized_tensor_storage_bytes": b.series.untyped_storage().nbytes(),
    "exact_legacy_window_parity": True,
    "windows": len(a),
}
print("window done", flush=True)
# Real original VanillaVAE architecture; explicit posterior-mean mode used on both
# paths to isolate batching speed from stochastic-policy changes.
torch.manual_seed(42)
model = new.VanillaVAE1D(window_size=50)
device = torch.device("cpu")
serial = old.get_reconstructions(model, b, device)
batch = new.get_reconstructions(model, b, device, 256)
np.testing.assert_allclose(serial, batch, atol=1e-6, rtol=1e-5)
result["inference"] = paired(
    lambda: old.get_reconstructions(model, b, device),
    lambda: new.get_reconstructions(model, b, device, 256),
    3,
)
result["inference"]["mode"] = (
    "same original VanillaVAE weights; posterior-mean evaluation in both serial and batched paths, no quality claim"
)
result["inference"]["max_absolute_output_difference"] = float(
    np.max(np.abs(serial - batch))
)
print("inference done", flush=True)
# Tail of legacy windowing remains intentionally uncovered in speed-only comparison.
legacy = old.aggregate_reconstructions(serial, ws, len(x))
fast = new.aggregate_reconstructions(serial, ws, len(x))
np.testing.assert_allclose(legacy[:-1], fast[:-1], atol=1e-12)
result["aggregation"] = paired(
    lambda: old.aggregate_reconstructions(serial, ws, len(x)),
    lambda: new.aggregate_reconstructions(serial, ws, len(x)),
    7,
)
result["aggregation"]["legacy_uncovered_tail"] = 1
print("aggregation done", flush=True)
old_classes = [
    old.VanillaVAE1D,
    old.BetaVAE1D,
    old.InfoVAE1D,
    old.WAVAE1D,
    old.GraphVAE1D,
    old.FCVAE1D,
    old.StackVAE_G1D,
]
result["model_smoke"] = []
for original, fixed in zip(old_classes, new.MODELS):
    row = {"model": original.__name__}
    for tag, cls in [("baseline", original), ("repaired", fixed)]:
        torch.manual_seed(42)
        m = cls(window_size=50)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=5e-5)
        try:
            d = new.SlidingWindowDataset(x[:512], 50)
            losses = []
            for batch in DataLoader(d, batch_size=64, shuffle=False):
                out = m(batch)
                loss = m.loss_function(*out)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            row[tag] = {
                "passed": True,
                "training_batches": len(losses),
                "finite_losses": True,
            }
        except (RuntimeError, ValueError, AttributeError, TypeError) as e:
            row[tag] = {"passed": False, "error": type(e).__name__ + ": " + str(e)}
    result["model_smoke"].append(row)
print("models done", flush=True)
result["limitations"] = [
    "Timing on shared CPU host, no GPU or production throughput claim.",
    "Synthetic size matches historical sample count only; no real flight-data performance inferred.",
    "Original windows drop final window. Speed/memory comparison preserves that count; inclusive fix tested separately.",
    "New default evaluation uses latent means, unlike original stochastic evaluation. Inference speed compares mean mode on both paths.",
    "One-epoch 512-sample model smoke verifies execution and finite losses, not convergence or anomaly accuracy.",
    "Original train/calibrate/score reuse same data; new frozen-threshold helpers do not establish a historical employer validation.",
]
(ROOT / "benchmark_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
