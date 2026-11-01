from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.benchmark_vae_variants import (  # noqa: E402
    WINDOW,
    count_runs,
    make_windows,
    model_specs,
    objective,
)


def test_expected_model_set_is_explicit() -> None:
    names = [spec.name for spec in model_specs()]
    assert names == [
        "VanillaVAE1D",
        "BetaVAE1D",
        "InfoVAE1D",
        "WAVAE1D",
        "GraphVAE1D",
        "FCVAE1D",
        "StackVAE_G1D",
        "TiTAD-Lite (inspired)",
    ]


def test_windowing_preserves_endpoint_alignment() -> None:
    windows = make_windows(np.arange(WINDOW + 2, dtype=np.float32))
    assert windows.shape == (3, WINDOW)
    np.testing.assert_array_equal(windows[0], np.arange(WINDOW))
    np.testing.assert_array_equal(windows[-1], np.arange(2, WINDOW + 2))


@pytest.mark.parametrize("spec_index", range(8))
def test_each_model_forward_and_objective_are_finite(spec_index: int) -> None:
    torch.manual_seed(11)
    spec = model_specs()[spec_index]
    model = spec.build()
    batch = torch.randn(3, WINDOW)
    output = model(batch)
    assert output[0].shape == batch.shape
    loss = objective(spec, batch, output)
    assert torch.isfinite(loss)


def test_count_runs_counts_disjoint_alarm_bursts() -> None:
    assert count_runs(np.array([False, True, True, False, True, False])) == 2
