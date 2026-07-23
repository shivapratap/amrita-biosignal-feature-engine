from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import approximate_entropy

if os.environ.get("ABFE_RUN_REFERENCE") != "1":
    pytest.skip("set ABFE_RUN_REFERENCE=1 to run external references", allow_module_level=True)

antropy = pytest.importorskip("antropy")

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
def test_required_matrix_matches_antropy(name: str, variant: str) -> None:
    signal = required_signals()[name]
    if variant == "2x":
        signal = 2.0 * signal
    elif variant == "5x":
        signal = 5.0 * signal
    elif variant == "10x":
        signal = 10.0 * signal
    elif variant == "zscore":
        signal = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    actual = approximate_entropy(signal, order=2)
    expected = antropy.app_entropy(signal, order=2, metric="chebyshev")
    assert actual == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("seed", [1, 42, 999])
def test_randomized_matrix_matches_antropy(order: int, seed: int) -> None:
    signal = np.random.default_rng(seed).normal(size=160)
    actual = approximate_entropy(signal, order=order)
    expected = antropy.app_entropy(signal, order=order, metric="chebyshev")
    assert actual == pytest.approx(expected, abs=2e-15)
