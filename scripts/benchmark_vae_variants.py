#!/usr/bin/env python3
"""Leakage-aware comparison of eight 1D VAE/sequence reconstruction variants on NAB.

The seven historical VAE names are implemented as explicit 1D adaptations from
the earlier lab. TiTAD-Lite is an eighth Transformer reconstruction baseline;
it is an inspired substitute, not an official TiTAD implementation.

Example:
    python scripts/benchmark_vae_variants.py --epochs 30 --seed 42

Only normal history before the first labeled anomaly is used for fitting and
threshold calibration. The threshold is frozen before the labeled evaluation
region is scored. Outputs are written to results/nab_vae_benchmark.csv/json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from anomaly_lab.evaluation import average_precision

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "benchmark"
RESULTS_DIR = ROOT / "results"
NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
WINDOW = 50
LATENT = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "time-series-anomaly-lab"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def load_nab() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[str]]:
    data_path = DATA_DIR / "nyc_taxi.csv"
    labels_path = DATA_DIR / "combined_windows.json"
    download(f"{NAB_BASE}/data/realKnownCause/nyc_taxi.csv", data_path)
    download(f"{NAB_BASE}/labels/combined_windows.json", labels_path)
    rows = list(csv.DictReader(data_path.open(encoding="utf-8")))
    timestamps = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    values = np.asarray([float(row["value"]) for row in rows], dtype=np.float32)
    label_map = json.loads(labels_path.read_text(encoding="utf-8"))
    raw_windows = label_map["realKnownCause/nyc_taxi.csv"]
    events: list[tuple[int, int]] = []
    labels = np.zeros(len(values), dtype=bool)
    for start_text, end_text in raw_windows:
        start = datetime.fromisoformat(start_text)
        end = datetime.fromisoformat(end_text)
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="right"))
        events.append((left, right))
        labels[left:right] = True
    return values, labels, events, [x.isoformat(sep=" ") for x in timestamps]


def make_windows(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(values) < window:
        raise ValueError(f"need at least {window} samples, got {len(values)}")
    return np.lib.stride_tricks.sliding_window_view(values, window).copy().astype(np.float32)


def mmd_rbf(x: Tensor, y: Tensor) -> Tensor:
    """Biased RBF-kernel MMD with a bandwidth estimated from the batch."""
    xx = torch.cdist(x, x).square()
    yy = torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()
    bandwidth = torch.cat((xx.flatten(), yy.flatten(), xy.flatten())).median().detach()
    bandwidth = bandwidth.clamp_min(1e-4)
    return (
        torch.exp(-xx / (2 * bandwidth)).mean()
        + torch.exp(-yy / (2 * bandwidth)).mean()
        - 2 * torch.exp(-xy / (2 * bandwidth)).mean()
    )


class DenseVAE(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (32, 16), latent: int = LATENT):
        super().__init__()
        layers: list[nn.Module] = []
        incoming = WINDOW
        for width in hidden:
            layers += [nn.Linear(incoming, width), nn.ReLU()]
            incoming = width
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(incoming, latent)
        self.logvar = nn.Linear(incoming, latent)
        decoder: list[nn.Module] = []
        incoming = latent
        for width in reversed(hidden):
            decoder += [nn.Linear(incoming, width), nn.ReLU()]
            incoming = width
        decoder.append(nn.Linear(incoming, WINDOW))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-12, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar, z


class WAE(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (32, 16), latent: int = LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(WINDOW, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], WINDOW)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z, torch.zeros_like(z), z


class GraphVAE(nn.Module):
    """Temporal-chain graph convolution followed by a variational bottleneck."""

    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.graph = nn.Conv1d(1, 8, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            filters = torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.25, 0.5, 0.25],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, -1.0],
                    [-0.5, 1.0, -0.5],
                    [1.0, -2.0, 1.0],
                ]
            ).unsqueeze(1)
            self.graph.weight.copy_(filters)
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(8 * WINDOW, 32), nn.ReLU())
        self.mu, self.logvar = nn.Linear(32, latent), nn.Linear(32, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, WINDOW))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.enc(self.graph(x.unsqueeze(1)))
        mu, logvar = self.mu(h), self.logvar(h).clamp(-12, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar, z


class StackedVAE(nn.Module):
    """Two-level Gaussian latent stack with reconstruction and both KL terms."""

    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Linear(WINDOW, 32), nn.ReLU())
        self.mu1, self.lv1 = nn.Linear(32, latent), nn.Linear(32, latent)
        self.enc2 = nn.Sequential(nn.Linear(latent, 16), nn.ReLU())
        self.mu2, self.lv2 = nn.Linear(16, latent), nn.Linear(16, latent)
        self.dec2 = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, latent))
        self.dec1 = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, WINDOW))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h1 = self.enc1(x)
        mu1, lv1 = self.mu1(h1), self.lv1(h1).clamp(-12, 8)
        z1 = mu1 + torch.randn_like(mu1) * torch.exp(0.5 * lv1)
        h2 = self.enc2(z1)
        mu2, lv2 = self.mu2(h2), self.lv2(h2).clamp(-12, 8)
        z2 = mu2 + torch.randn_like(mu2) * torch.exp(0.5 * lv2)
        recon = self.dec1(self.dec2(z2))
        # Concatenate both latent statistics for a single consistent loss API.
        mu, lv = torch.cat((mu1, mu2), 1), torch.cat((lv1, lv2), 1)
        return recon, mu, lv, z2


class TiTADLite(nn.Module):
    """Transformer sequence autoencoder, included as an explicitly inspired baseline."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(1, 16)
        layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.out_proj = nn.Linear(16, 1)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, 16))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.in_proj(x.unsqueeze(-1)) + self.pos
        h = self.encoder(h)
        recon = self.out_proj(h).squeeze(-1)
        z = h.mean(dim=1)
        return recon, z, torch.zeros_like(z), z


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[], nn.Module]
    loss_kind: str
    beta: float = 1.0


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("VanillaVAE1D", lambda: DenseVAE((32, 16)), "vae"),
        ModelSpec("BetaVAE1D", lambda: DenseVAE((32, 16)), "vae", 4.0),
        ModelSpec("InfoVAE1D", lambda: DenseVAE((32, 16)), "info"),
        ModelSpec("WAVAE1D", lambda: WAE((32, 16)), "wae"),
        ModelSpec("GraphVAE1D", GraphVAE, "vae"),
        ModelSpec("FCVAE1D", lambda: DenseVAE((64, 32, 16)), "vae"),
        ModelSpec("StackVAE_G1D", StackedVAE, "stack"),
        ModelSpec("TiTAD-Lite (inspired)", TiTADLite, "reconstruction"),
    ]


def objective(spec: ModelSpec, x: Tensor, out: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    recon, mu, logvar, z = out
    mse = F.mse_loss(recon, x)
    if spec.loss_kind == "reconstruction":
        return mse
    if spec.loss_kind == "wae":
        prior = torch.randn_like(z)
        return mse + 10.0 * mmd_rbf(z, prior)
    if spec.loss_kind == "info":
        kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
        return mse + 0.1 * kl + mmd_rbf(z, torch.randn_like(z))
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return mse + spec.beta * kl


def fit_model(spec: ModelSpec, windows: np.ndarray, *, epochs: int, seed: int) -> nn.Module:
    seed_everything(seed)
    model = spec.build().to(DEVICE)
    data = torch.from_numpy(windows)
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(spec, batch, model(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {spec.name}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model.eval()


@torch.no_grad()
def window_scores(
    model: nn.Module, history: np.ndarray, values: np.ndarray, mean: float, scale: float
) -> np.ndarray:
    context = history[-(WINDOW - 1) :] if len(history) else np.empty(0, dtype=np.float32)
    joined = np.concatenate((context, values)).astype(np.float32)
    normalized = (joined - mean) / scale
    windows = make_windows(normalized)
    result: list[np.ndarray] = []
    for start in range(0, len(windows), 512):
        batch = torch.from_numpy(windows[start : start + 512]).to(DEVICE)
        reconstruction = model(batch)[0]
        result.append(torch.mean((reconstruction - batch) ** 2, dim=1).cpu().numpy())
    scores = np.concatenate(result)
    if len(scores) != len(values):
        raise RuntimeError(f"score alignment error: expected {len(values)}, got {len(scores)}")
    return scores


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def count_runs(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    return int(np.sum(np.diff(padded) == 1))


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    values, labels, events, timestamps = load_nab()
    first_anomaly = events[0][0]
    fit_end = int(first_anomaly * 0.65)
    if fit_end <= WINDOW or fit_end >= first_anomaly:
        raise ValueError("not enough clean pre-anomaly history for fit/calibration")
    raw_fit = values[:fit_end]
    mean, scale = float(raw_fit.mean()), float(raw_fit.std())
    scale = max(scale, 1e-6)
    norm = (values - mean) / scale
    fit_windows = make_windows(norm[:fit_end])
    calibration = values[fit_end:first_anomaly]
    evaluation = values[first_anomaly:]
    y_eval = labels[first_anomaly:]
    event_eval = [
        (max(0, s - first_anomaly), min(len(evaluation), e - first_anomaly))
        for s, e in events
        if e > first_anomaly
    ]
    quantiles = (0.95, 0.995)
    rows: list[dict[str, object]] = []

    for i, spec in enumerate(model_specs()):
        print(
            f"Training {spec.name} on {len(fit_windows):,} normal windows ({args.epochs} epochs)",
            flush=True,
        )
        model = fit_model(spec, fit_windows, epochs=args.epochs, seed=args.seed + i)
        calibration_scores = window_scores(model, values[:fit_end], calibration, mean, scale)
        thresholds = {q: float(np.quantile(calibration_scores, q)) for q in quantiles}

        # Keep the fit checkpoint fixed after calibration so score scale does not
        # shift between the calibration and evaluation periods.
        test_scores = window_scores(model, values[:first_anomaly], evaluation, mean, scale)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_dir / f"scores_{i}.npz",
            calibration=calibration_scores, evaluation=test_scores, labels=y_eval,
        )
        ap = average_precision(y_eval, test_scores)
        for q, threshold in thresholds.items():
            predictions = test_scores >= threshold
            point = metrics(y_eval, predictions)
            detected = sum(
                bool(np.any(predictions[max(0, s - 3) : min(len(predictions), e + 3)]))
                for s, e in event_eval
            )
            event_mask = np.zeros(len(predictions), dtype=bool)
            for s, e in event_eval:
                event_mask[max(0, s - 3) : min(len(predictions), e + 3)] = True
            false_runs = count_runs(predictions & ~event_mask)
            delays = []
            for start, end in event_eval:
                hits = np.flatnonzero(predictions[start:end])
                if len(hits):
                    delays.append(int(hits[0]))
            normal_count = int((~y_eval).sum())
            rows.append(
                {
                    "model": spec.name,
                    "average_precision": ap,
                    "seed": args.seed,
                    "event_tolerance_samples": 3,
                    "strict_events_detected": len(delays),
                    "mean_strict_detection_delay_samples": (
                        float(np.mean(delays)) if delays else None
                    ),
                    "max_strict_detection_delay_samples": max(delays) if delays else None,
                    "threshold_quantile": q,
                    "threshold": threshold,
                    "samples_total": len(values),
                    "normal_fit_samples": fit_end,
                    "normal_calibration_samples": first_anomaly - fit_end,
                    "evaluation_samples": len(evaluation),
                    "labeled_events": len(event_eval),
                    "events_detected": detected,
                    "event_recall": detected / len(event_eval) if event_eval else 0.0,
                    "false_positive_runs": false_runs,
                    "false_alarms_per_1000_normal_points": 1000 * false_runs / max(normal_count, 1),
                    **point,
                }
            )
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    seed_everything(args.seed)
    rows = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "nab_vae_benchmark.csv"
    json_path = args.output_dir / "nab_vae_benchmark.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "dataset": "NAB realKnownCause/nyc_taxi",
                "source_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(DATA_DIR.glob("*")) if path.is_file()
                },
                "environment": {
                    "python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__,
                },
                "limitations": [
                    "Single public series; not employer or flight-test data.",
                    "Split uses first labeled anomaly; clean-history availability is assumed.",
                    "AP/model ranking are descriptive, not a separate selection/test protocol.",
                    "Neural scoring uses stochastic latent samples; seed and environment matter.",
                    "NAB label windows are not exact incident onset times.",
                ],
                "protocol": (
                    "chronological normal-only fit/calibration; frozen thresholds; "
                    "post-boundary evaluation"
                ),
                "window": WINDOW,
                "seed": args.seed,
                "epochs_per_fit": args.epochs,
                "device": str(DEVICE),
                "models": [spec.name for spec in model_specs()],
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {csv_path}\nWrote {json_path}")
    print("model | q | event recall | false alarm runs | point F1")
    for row in rows:
        print(
            f"{row['model']} | {row['threshold_quantile']} | "
            f"{row['events_detected']}/{row['labeled_events']} "
            f"({row['event_recall']:.3f}) | {row['false_positive_runs']} | {row['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
