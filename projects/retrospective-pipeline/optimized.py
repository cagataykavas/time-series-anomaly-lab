"""Private revision of recovered generic anomaly code; no employer data bundled."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import original


class SlidingWindowDataset(Dataset):
    def __init__(self, series, window_size, *, legacy_drop_final=False):
        a = np.asarray(series)
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError("expected finite univariate series")
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size < 1
            or len(a) < window_size
        ):
            raise ValueError("invalid window size")
        self.ws = window_size
        self.series = torch.tensor(a, dtype=torch.float32)
        self.samples = self.series.unfold(0, window_size, 1)
        if legacy_drop_final:
            self.samples = self.samples[:-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i].unsqueeze(0)


class MeanEvaluation:
    def reparam(self, mu, lv):
        return super().reparam(mu, lv) if self.training else mu


class VanillaVAE1D(MeanEvaluation, original.VanillaVAE1D):
    pass


class BetaVAE1D(MeanEvaluation, original.BetaVAE1D):
    pass


class InfoVAE1D(MeanEvaluation, original.InfoVAE1D):
    pass


class WAVAE1D(MeanEvaluation, original.WAVAE1D):
    pass


class FCVAE1D(MeanEvaluation, original.FCVAE1D):
    pass


class GraphVAE1D(MeanEvaluation, original.GraphVAE1D):
    def forward(self, x):
        if self.nf != 1 or x.ndim != 3 or x.shape[1:] != (1, self.ws):
            raise ValueError(
                "reference repair supports univariate (batch,1,window) only"
            )
        target = x.reshape(x.size(0), -1)
        hidden = self.enc_gc(target)
        mu, lv = self.fc_mu(hidden), self.fc_lv(hidden).clamp(-10, 10)
        recon = self.dec_gc(self.reparam(mu, lv))
        recon = recon - recon.mean(1, keepdim=True) + target.mean(1, keepdim=True)
        return recon, target, mu, lv


class StackVAE_G1D(MeanEvaluation, original.StackVAE_G1D):
    def forward(self, x):
        if self.nf != 1 or x.ndim != 3 or x.shape[1:] != (1, self.ws):
            raise ValueError(
                "reference repair supports univariate (batch,1,window) only"
            )
        target = x.reshape(x.size(0), -1)
        h = torch.relu(self.gc1(x.transpose(1, 2)))
        h = torch.relu(self.gc2(h)).reshape(x.size(0), -1)
        stats = self.encoder(h)
        d = stats.size(1) // 2
        mu, lv = stats[:, :d], stats[:, d:].clamp(-10, 10)
        recon = self.decoder(self.reparam(mu, lv))
        recon = recon - recon.mean(1, keepdim=True) + target.mean(1, keepdim=True)
        return recon, target, mu, lv

    def loss_function(self, recon, target, mu, lv):
        return nn.functional.mse_loss(
            recon, target, reduction="sum"
        ) - self.beta * 0.5 * torch.sum(1 + lv - mu.square() - lv.exp())


def get_reconstructions(model, dataset, device, batch_size=256):
    model.eval()
    output = np.empty((len(dataset), dataset.ws), dtype=np.float32)
    start = 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            recon = model(batch.to(device))[0].cpu().numpy()
            output[start : start + len(recon)] = recon
            start += len(recon)
    return output


def aggregate_reconstructions(recons, ws, T):
    r = np.asarray(recons)
    if r.ndim != 2 or r.shape[1] != ws or len(r) + ws - 1 > T:
        raise ValueError("inconsistent reconstruction dimensions")
    out = np.zeros(T, dtype=np.float64)
    counts = np.zeros(T, dtype=np.int64)
    for offset in range(ws):
        out[offset : offset + len(r)] += r[:, offset]
        counts[offset : offset + len(r)] += 1
    return np.divide(out, counts, out=np.full(T, np.nan), where=counts > 0)


def frozen_threshold(calibration_errors, quantile=0.995):
    errors = np.asarray(calibration_errors)
    if (
        errors.ndim != 1
        or not len(errors)
        or not np.isfinite(errors).all()
        or not 0 < quantile < 1
    ):
        raise ValueError("invalid calibration")
    return float(np.quantile(errors, quantile))


def score_windows(reconstructions, dataset, threshold):
    # Window-end attribution, avoiding future windows in online point scoring.
    recons = np.asarray(reconstructions)
    target = dataset.samples.numpy()
    if (
        recons.shape != target.shape
        or not np.isfinite(recons).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("invalid scoring input")
    errors = np.mean((recons - target) ** 2, axis=1)
    return errors, errors > threshold


MODELS = [
    VanillaVAE1D,
    BetaVAE1D,
    InfoVAE1D,
    WAVAE1D,
    GraphVAE1D,
    FCVAE1D,
    StackVAE_G1D,
]


def run_chronological(
    values, model_class=VanillaVAE1D, window_size=50, epochs=1, seed=42
):
    """Train-only normalization, disjoint windows and frozen calibration.

    This offline reference does not claim the original employer used this split.
    """
    series = np.asarray(values, dtype=np.float32)
    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError("expected finite univariate series")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    split1, split2 = int(len(series) * 0.6), int(len(series) * 0.8)
    if min(split1, split2 - split1, len(series) - split2) < window_size:
        raise ValueError("each disjoint split must contain a full window")
    mean, scale = float(series[:split1].mean()), float(series[:split1].std())
    scale = max(scale, 1e-6)
    normalized = (series - mean) / scale
    datasets = [
        SlidingWindowDataset(part, window_size)
        for part in [
            normalized[:split1],
            normalized[split1:split2],
            normalized[split2:],
        ]
    ]
    torch.manual_seed(seed)
    model = model_class(window_size=window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    losses = []
    for _ in range(epochs):
        model.train()
        for batch in DataLoader(datasets[0], batch_size=64, shuffle=True):
            outputs = model(batch)
            loss = model.loss_function(*outputs)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    cal_recon = get_reconstructions(model, datasets[1], torch.device("cpu"))
    cal_errors = np.mean((cal_recon - datasets[1].samples.numpy()) ** 2, axis=1)
    threshold = frozen_threshold(cal_errors)
    test_recon = get_reconstructions(model, datasets[2], torch.device("cpu"))
    errors, flags = score_windows(test_recon, datasets[2], threshold)
    return {
        "model": model,
        "normalization": {"mean": mean, "scale": scale},
        "split_boundaries": [0, split1, split2, len(series)],
        "window_counts": [len(d) for d in datasets],
        "threshold": threshold,
        "test_errors": errors,
        "test_flags": flags,
        "losses": losses,
        "test_first_window_end": split2 + window_size - 1,
    }
