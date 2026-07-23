from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import distribution_entropy

if os.environ.get("ABFE_RUN_REFERENCE") != "1":
    pytest.skip("set ABFE_RUN_REFERENCE=1 to run external references", allow_module_level=True)

dihc_kernels = pytest.importorskip("dihc_feature_manager._numba_kernels")

pytestmark = pytest.mark.reference


def required_signals() -> dict[str, NDArray[np.float64]]:
    periodic_time = np.linspace(0.0, 6.0 * np.pi, 128, endpoint=False)
    chirp_time = np.linspace(0.0, 1.0, 128, endpoint=False)
    return {
        "periodic": np.sin(periodic_time),
        "white_noise": np.random.default_rng(321).normal(size=128),
        "chirp": np.sin(2.0 * np.pi * (2.0 * chirp_time + 10.0 * chirp_time**2)),
    }


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("variant", ["original", "2x", "5x", "10x", "zscore"])
@pytest.mark.parametrize("order,n_bins", [(1, 16), (2, 64), (3, 500)])
def test_matches_pinned_dihc_for_ordinary_inputs(
    name: str, variant: str, order: int, n_bins: int
) -> None:
    signal = required_signals()[name]
    if variant == "2x":
        signal = 2.0 * signal
    elif variant == "5x":
        signal = 5.0 * signal
    elif variant == "10x":
        signal = 10.0 * signal
    elif variant == "zscore":
        signal = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    actual = distribution_entropy(signal, order=order, delay=1, n_bins=n_bins)
    expected = dihc_kernels.distribution_entropy(signal, m=order, n_bins=n_bins)
    assert actual == pytest.approx(expected, abs=6e-15)


def test_constant_input_corrects_dihc_zero_fallback() -> None:
    signal = np.ones(20)
    assert np.isnan(distribution_entropy(signal))
    assert dihc_kernels.distribution_entropy(signal) == 0.0
