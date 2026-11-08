import numpy as np
import pytest
import torch

import optimized as new
import original


def test_compatibility_windows_exact():
    x = np.random.default_rng(42).normal(size=72935)
    old = original.SlidingWindowDataset(x, 50)
    fixed = new.SlidingWindowDataset(x, 50, legacy_drop_final=True)
    assert torch.equal(old.samples, fixed.samples)
    assert (
        fixed.samples.untyped_storage().data_ptr()
        == fixed.series.untyped_storage().data_ptr()
    )


def test_last_window_and_no_uncovered_tail():
    x = np.arange(100, dtype=np.float32)
    d = new.SlidingWindowDataset(x, 50)
    assert len(d) == 51
    np.testing.assert_array_equal(d.samples[-1], x[-50:])
    np.testing.assert_array_equal(
        new.aggregate_reconstructions(d.samples.numpy(), 50, 100), x
    )


@pytest.mark.parametrize(
    "x,ws",
    [
        (np.arange(4), 5),
        (np.arange(5), 0),
        (np.array([1, np.nan]), 1),
        (np.ones((4, 2)), 2),
    ],
)
def test_invalid_windows(x, ws):
    with pytest.raises(ValueError):
        new.SlidingWindowDataset(x, ws)


@pytest.mark.parametrize("cls", new.MODELS)
def test_model_train_infer(cls):
    torch.manual_seed(42)
    model = cls(window_size=50)
    x = torch.randn(8, 1, 50)
    model.train()
    out = model(x)
    assert out[0].shape == out[1].shape == (8, 50)
    loss = model.loss_function(*out)
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    assert torch.equal(model(x)[0], model(x)[0])


def test_batched_mean_inference_close():
    d = new.SlidingWindowDataset(np.sin(np.arange(400)), 50)
    m = new.VanillaVAE1D(window_size=50)
    serial = original.get_reconstructions(m, d, torch.device("cpu"))
    batched = new.get_reconstructions(m, d, torch.device("cpu"), 64)
    np.testing.assert_allclose(serial, batched, atol=1e-6, rtol=1e-5)


def test_aggregation_close():
    r = np.random.default_rng(42).normal(size=(1000, 50)).astype(np.float32)
    np.testing.assert_allclose(
        new.aggregate_reconstructions(r, 50, 1049),
        original.aggregate_reconstructions(r, 50, 1049),
        atol=1e-12,
    )


def test_frozen_threshold_future_invariance():
    cal = np.linspace(0, 1, 100)
    t = new.frozen_threshold(cal)
    d = new.SlidingWindowDataset(np.zeros(100), 50)
    _, flags = new.score_windows(np.zeros((51, 50)), d, t)
    _, future = new.score_windows(np.ones((51, 50)) * 100, d, t)
    assert t == new.frozen_threshold(cal) and not flags.any() and future.all()


def test_chronological_fit_and_threshold_ignore_future():
    x = np.sin(np.arange(1000, dtype=np.float32) / 20)
    altered = x.copy()
    altered[800:] += 100
    a = new.run_chronological(x)
    b = new.run_chronological(altered)
    assert a["normalization"] == b["normalization"]
    assert a["threshold"] == b["threshold"]
    assert a["losses"] == b["losses"]
    assert a["split_boundaries"] == [0, 600, 800, 1000]
    assert a["window_counts"] == [551, 151, 151]
    assert a["test_first_window_end"] == 849
    assert not np.array_equal(a["test_errors"], b["test_errors"])


@pytest.mark.parametrize("cls", new.MODELS)
def test_chronological_all_models(cls):
    x = np.sin(np.arange(1000, dtype=np.float32) / 20)
    result = new.run_chronological(x, model_class=cls)
    assert np.isfinite(result["threshold"])
    assert result["test_errors"].shape == (151,)
    assert len(result["losses"]) == 9
