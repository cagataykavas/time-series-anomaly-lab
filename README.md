# Time-Series Anomaly Lab

A public benchmark harness for **unsupervised and weakly supervised time-series anomaly detection**.

The project is designed around a common real-world constraint: multivariate or univariate sensor data may have little or no reliable anomaly labeling. The repository therefore emphasizes reproducible preprocessing, windowing, reconstruction/error scoring, threshold calibration and model comparison rather than a single magic detector.

## Scope

Planned / supported experiment families include:

- statistical baselines
- rolling z-score and robust MAD thresholds
- PCA reconstruction error
- dense autoencoders
- convolutional autoencoders
- LSTM / CNN-LSTM autoencoders
- temporal convolutional autoencoders
- VAE-style models

## Public-data boundary

This repository uses **synthetic and public datasets only**. It contains no employer sensor data, flight-test data, proprietary labels or internal thresholds.

## Pipeline

```text
Raw series
   |
   v
Cleaning / scaling
   |
   v
Sliding windows
   |
   +--> baseline features
   |
   +--> reconstruction models
   |
   v
Anomaly score
   |
   v
Threshold calibration
   |
   v
Event-level evaluation + plots
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python demo.py
```

`demo.py` generates a synthetic sensor series with injected spikes, drifts and level shifts, then compares robust statistical scores using the same evaluation interface that later neural models can plug into.

## Why this repository exists

Anomaly-detection work often becomes difficult to reproduce because preprocessing, window alignment and thresholding are mixed into model code. This project keeps those concerns explicit so models can be compared fairly.

## Evaluation philosophy

Pointwise accuracy is often misleading under heavy class imbalance. The lab therefore favors:

- precision / recall / F1 on injected or labeled anomalies
- event-level detection rate
- false alarms per unit time
- detection delay
- threshold sensitivity
- score distributions and time-aligned plots

## Roadmap

1. robust statistical baseline
2. PCA reconstruction baseline
3. PyTorch dense autoencoder
4. CNN-LSTM and causal TCN-LSTM autoencoders
5. VAE-family benchmark interface
6. config-driven experiment runner
7. saved metrics and plots
8. CI smoke tests

## Portfolio context

The point of this repository is not to advertise one benchmark number. It demonstrates the full engineering workflow around anomaly detection: **data preparation, windowing, modeling, score calibration, evaluation and failure analysis**.
