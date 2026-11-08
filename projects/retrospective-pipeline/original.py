import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from typing import List, Tuple, Type

# =========================
# === COMMON UTILITIES ===
# =========================

class SlidingWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, window_size: int):
        self.ws = window_size
        self.series = torch.tensor(series, dtype=torch.float32)
        self.samples = torch.stack([
            self.series[i : i + window_size]
            for i in range(len(self.series) - window_size)
        ])
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> torch.Tensor:
        # returns shape (1, ws)
        return self.samples[idx].unsqueeze(0)

def train_model_generic(
    model: nn.Module,
    dataset: Dataset,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 5e-5,
    kl_warmup: int = 10,
    device: torch.device = None
) -> nn.Module:
    torch.manual_seed(0); np.random.seed(0)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        kl_wt = min(1.0, epoch / kl_warmup)
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            # unpack based on model type
            if isinstance(model, InfoVAE1D):
                recon, xflat, mu, lv, z = out
                loss = model.loss_function(recon, xflat, mu, lv, z)
            elif isinstance(model, WAVAE1D):
                recon, xflat, mu, lv, z, z2 = out
                loss = model.loss_function(recon, xflat, mu, lv, z, z2)
            else:
                recon, xflat, mu, lv = out
                if isinstance(model, BetaVAE1D):
                    loss = model.loss_function(recon, xflat, mu, lv)
                else:
                    # VanillaVAE, GraphVAE, FCVAE, StackVAE
                    loss = model.loss_function(recon, xflat, mu, lv)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        print(f"[{model.__class__.__name__}] Epoch {epoch}/{epochs} — loss {avg:.4f}")
    # plot training loss
    plt.figure(figsize=(6,3))
    plt.plot(losses, marker='o')
    plt.title(f"{model.__class__.__name__} Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.show()
    return model

def get_reconstructions(model: nn.Module, dataset: Dataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    recons = []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            recon = model(b)[0]  # always first element
            recons.append(recon.cpu().numpy().reshape(-1))
    return np.stack(recons, axis=0)

def aggregate_reconstructions(recons: np.ndarray, ws: int, T: int):
    recon_series = np.zeros(T)
    counts = np.zeros(T)
    for i in range(len(recons)):
        recon_series[i:i+ws] += recons[i]
        counts[i:i+ws] += 1
    return recon_series / np.maximum(counts, 1)

def compute_threshold(errors: np.ndarray, method: str = 'meanstd', quantile: float = 0.995):
    if method == 'meanstd':
        return errors.mean() + 3 * errors.std()
    return np.quantile(errors, quantile)

def score_and_flag(values: np.ndarray, recon: np.ndarray, ws: int, method='meanstd'):
    errors = (recon[ws:] - values[ws:]) ** 2
    thr = compute_threshold(errors, method=method)
    flags = errors > thr
    return errors, flags, thr

def summary(flags: np.ndarray):
    cnt = int(flags.sum())
    pct = 100.0 * cnt / flags.shape[0]
    return cnt, pct

def plot_static(values: np.ndarray, recon: np.ndarray, flags: np.ndarray, ws: int, title: str):
    T = len(values)
    t = np.arange(T)
    plt.figure(figsize=(10,3))
    plt.plot(t, values, label='Original', alpha=0.6)
    plt.plot(t, recon, label='Reconstructed', alpha=0.8)
    at = t[ws:][flags]; av = values[ws:][flags]
    plt.scatter(at, av, color='red', label='Anomaly')
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

def plot_interactive(values: np.ndarray, recon: np.ndarray, flags: np.ndarray, ws: int, title: str):
    df = pd.DataFrame({
        'Time': np.arange(len(values)),
        'Original': values,
        'Reconstructed': recon
    })
    df['Anomaly'] = False
    df.loc[ws + np.where(flags)[0], 'Anomaly'] = True
    fig = px.line(df, x='Time', y=['Original','Reconstructed'], title=title)
    fig.add_scatter(
        x=df[df['Anomaly']]['Time'],
        y=df[df['Anomaly']]['Original'],
        mode='markers',
        marker=dict(color='red', size=6),
        name='Anomaly'
    )
    fig.show()

# =========================
# === MODEL CLASSES =======
# =========================

class VanillaVAE1D(nn.Module):
    def __init__(self, latent_dim=8, window_size=50):
        super().__init__()
        self.ws = window_size
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size, 64), nn.ReLU(),
            nn.Linear(64, latent_dim*2)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, window_size)
        )
    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = self.enc(x_flat)
        D = h.size(1)//2
        mu, lv = h[:,:D], h[:,D:]
        z = self.reparam(mu, lv)
        recon = self.dec(z)
        xm = x_flat.mean(1, keepdim=True)
        recon = recon - recon.mean(1,keepdim=True) + xm
        return recon, x_flat, mu, lv
    def loss_function(self, recon, xflat, mu, lv):
        recon_l = F.mse_loss(recon, xflat, reduction='sum')
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        return recon_l + kl

class BetaVAE1D(VanillaVAE1D):
    def __init__(self, latent_dim=8, beta=4.0, window_size=50):
        super().__init__(latent_dim, window_size)
        self.beta = beta
    def loss_function(self, recon, xflat, mu, lv):
        recon_l = F.mse_loss(recon, xflat, reduction='sum')
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        return recon_l + self.beta * kl

class InfoVAE1D(VanillaVAE1D):
    def __init__(self, latent_dim=8, alpha=1.0, lambda_mmd=1.0, window_size=50):
        super().__init__(latent_dim, window_size)
        self.alpha = alpha
        self.lambda_mmd = lambda_mmd
    def forward(self, x):
        recon, xflat, mu, lv = super().forward(x)
        lv = torch.clamp(lv, -10, 10)
        z = self.reparam(mu, lv)
        return recon, xflat, mu, lv, z
    def loss_function(self, recon, xflat, mu, lv, z):
        recon_l = F.mse_loss(recon, xflat, reduction='sum')
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        zp = torch.randn_like(z)
        def mmd(a,b):
            d=(a.unsqueeze(1)-b.unsqueeze(0)).pow(2).sum(-1)
            s=0
            for σ in [1,2,4,8,16]:
                s+=torch.exp(-d/(2*σ*σ)).mean()
            return s
        m = mmd(z,z)+mmd(zp,zp)-2*mmd(z,zp)
        m = m / z.size(0)
        return recon_l + self.alpha*kl + self.lambda_mmd*m

class WAVAE1D(VanillaVAE1D):
    def __init__(self, latent_dim=8, beta=1.0, contrastive_weight=1.0,
                 temperature=0.5, augmentation_noise_std=0.05, window_size=50):
        super().__init__(latent_dim, window_size)
        self.beta = beta
        self.cw = contrastive_weight
        self.temp = temperature
        self.noise = augmentation_noise_std
    def augment(self, x):
        return x + self.noise * torch.randn_like(x)
    def forward(self, x):
        recon, xflat, mu, lv = super().forward(x)
        z = self.reparam(mu, lv)
        x2 = self.augment(x).view(x.size(0), -1)
        h2 = self.enc(x2)
        D = h2.size(1)//2
        mu2, lv2 = h2[:,:D], h2[:,D:]
        z2 = self.reparam(mu2, lv2)
        return recon, xflat, mu, lv, z, z2
    def loss_function(self, recon, xflat, mu, lv, z, z2):
        recon_l = F.mse_loss(recon, xflat, reduction='sum')
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        zn, z2n = F.normalize(z,1), F.normalize(z2,1)
        sim = torch.mm(zn, z2n.t()) / self.temp
        pos = torch.exp(torch.diag(sim))
        neg = torch.exp(sim).sum(1)
        cont = -torch.log(pos / neg).mean()
        return recon_l + self.beta*kl + self.cw*cont

class GraphVAE1D(VanillaVAE1D):
    def __init__(self, window_size=50, num_features=1,
                 latent_dim=8, beta=1.0, hidden_dim=64):
        super().__init__(latent_dim, window_size)
        self.ws, self.nf, self.beta = window_size, num_features, beta
        self.register_buffer('adj', torch.eye(num_features))
        self.enc_gc = nn.Sequential(
            nn.Linear(window_size*num_features, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_lv = nn.Linear(hidden_dim, latent_dim)
        self.dec_gc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, window_size*num_features)
        )
    def set_adjacency(self, data: np.ndarray):
        arr = np.array(data)
        if arr.ndim == 1 or arr.shape[1] == 1:
            self.adj = torch.eye(1); return
        corr = np.corrcoef(arr.T); corr = np.nan_to_num(corr)
        a = np.abs(corr); np.fill_diagonal(a,0)
        a += np.eye(a.shape[0]); a /= a.sum(1,keepdims=True)
        self.adj = torch.from_numpy(a.astype(np.float32))
    def graph_conv(self, x: torch.Tensor):
        x_t = x.transpose(1,2)
        out = torch.einsum('ij,bjk->bik', self.adj.to(x.device), x_t)
        return out.transpose(1,2)
    def forward(self, x):
        xgc = self.graph_conv(x) if self.nf>1 else x
        flat = self.enc_gc(xgc.reshape(x.size(0),-1))
        mu, lv = self.fc_mu(flat), self.fc_lv(flat)
        lv = torch.clamp(lv, -10, 10)
        z = self.reparam(mu, lv)
        recon = self.dec_gc(z).view(x.size(0), -1)
        xm = x.mean(1, keepdim=True)
        recon = recon.view(x.size(0), self.ws, self.nf)
        recon = recon - recon.mean(1, keepdim=True) + xm
        return recon.view(x.size(0), -1), flat, mu, lv

class FCVAE1D(nn.Module):
    def __init__(self, latent_dim=8, window_size=50, hidden_dim=64):
        super().__init__()
        self.ws = window_size
        self.freq_proj = nn.Sequential(
            nn.Linear(window_size//2+1, hidden_dim), nn.ReLU()
        )
        self.encoder = nn.Sequential(
            nn.Linear(window_size+hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, window_size)
        )
    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std
    def forward(self, x):
        xf = x.view(x.size(0),-1)
        Xf = torch.fft.rfft(xf, dim=1).abs()
        ff = self.freq_proj(Xf)
        stats = self.encoder(torch.cat([xf,ff],1))
        D = stats.size(1)//2
        mu, lv = stats[:,:D], stats[:,D:]
        lv = torch.clamp(lv, -10, 10)
        z = self.reparam(mu, lv)
        recon = self.decoder(z)
        xm = xf.mean(1, keepdim=True)
        recon = recon - recon.mean(1, keepdim=True) + xm
        return recon, xf, mu, lv
    def loss_function(self, recon, xf, mu, lv):
        recon_l = F.mse_loss(recon, xf, reduction='sum')
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        return recon_l + kl

class StackVAE_G1D(nn.Module):
    def __init__(self, window_size=50, num_features=1,
                 latent_dim=8, hidden_dim=64, beta=1.0):
        super().__init__()
        self.ws, self.nf, self.beta = window_size, num_features, beta
        self.register_buffer('adjacency', torch.eye(num_features))
        self.gc1 = nn.Linear(num_features, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(window_size*hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, window_size*num_features)
        )
    def set_adjacency(self, data: np.ndarray):
        arr = np.array(data)
        if arr.ndim == 1 or arr.shape[1] == 1:
            self.adjacency = torch.eye(1); return
        corr = np.corrcoef(arr.T); corr = np.nan_to_num(corr)
        a = np.abs(corr); np.fill_diagonal(a,0)
        a += np.eye(a.shape[0]); a /= a.sum(1,keepdims=True)
        self.adjacency = torch.from_numpy(a.astype(np.float32))
    def graph_conv(self, x):
        x_t = x.transpose(1,2)
        out = torch.einsum('ij,bjk->bik', self.adjacency.to(x.device), x_t)
        return out.transpose(1,2)
    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std
    def forward(self, x):
        xgc = self.graph_conv(x) if self.nf>1 else x
        h1 = F.relu(self.gc1(xgc.squeeze(1)))
        h2 = F.relu(self.gc2(h1).unsqueeze(2)).squeeze(2)
        flat = h2.reshape(x.size(0), -1)
        stats = self.encoder(flat)
        D = stats.size(1)//2
        mu, lv = stats[:,:D], stats[:,D:]
        lv = torch.clamp(lv, -10, 10)
        z = self.reparam(mu, lv)
        recon = self.decoder(z).view(x.size(0), -1)
        xm = x.mean(1, keepdim=True)
        recon = recon - recon.mean(1, keepdim=True) + xm
        return recon, flat, mu, lv

# =========================
# === RUN ALL MODELS =====
# =========================

def run_models(values: np.ndarray, window_size=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = len(values)
    ds = SlidingWindowDataset(values, window_size)

    configs: List[Tuple[str, Type[nn.Module], dict]] = [
        ("VanillaVAE",   VanillaVAE1D, {"latent_dim":8, "window_size":window_size}),
        ("BetaVAE",      BetaVAE1D,    {"latent_dim":8, "beta":4.0,    "window_size":window_size}),
        ("InfoVAE",      InfoVAE1D,    {"latent_dim":8, "alpha":1.0, "lambda_mmd":1.0, "window_size":window_size}),
        ("WAVAE",        WAVAE1D,      {"latent_dim":8, "beta":1.0, "contrastive_weight":1.0,
                                        "temperature":0.5, "augmentation_noise_std":0.05,
                                        "window_size":window_size}),
        ("GraphVAE",     GraphVAE1D,   {"window_size":window_size, "num_features":1,
                                        "latent_dim":8, "beta":1.0, "hidden_dim":64}),
        ("FCVAE",        FCVAE1D,      {"latent_dim":8, "window_size":window_size, "hidden_dim":64}),
        ("StackVAE-G",   StackVAE_G1D, {"window_size":window_size, "num_features":1,
                                        "latent_dim":8, "hidden_dim":64, "beta":1.0}),
    ]

    for name, cls, hp in configs:
        print(f"\n=== {name} ===")
        model = cls(**hp).to(device)
        if name in ("GraphVAE", "StackVAE-G"):
            model.set_adjacency(values.reshape(-1, hp.get("num_features",1)))
        model = train_model_generic(model, ds, epochs=30, batch_size=64, lr=5e-5, kl_warmup=10, device=device)
        recons = get_reconstructions(model, ds, device)
        recon_series = aggregate_reconstructions(recons, window_size, T)
        errors, flags, thr = score_and_flag(values, recon_series, window_size, method='meanstd')
        cnt, pct = summary(flags)
        print(f"Anomalies: {cnt}/{len(errors)} ({pct:.2f}%), threshold={thr:.4e}")
        plot_static(values, recon_series, flags, window_size, f"{name} Static")
        plot_interactive(values, recon_series, flags, window_size, f"{name} Interactive")

if __name__ == "__main__":
    t = np.linspace(0, 20 * math.pi, 2000)
    values = np.sin(0.2 * t) + 0.05 * np.random.randn(len(t))
    run_models(values, window_size=50)

