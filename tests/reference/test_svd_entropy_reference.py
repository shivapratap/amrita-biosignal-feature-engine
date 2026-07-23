from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import svd_entropy

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
@pytest.mark.parametrize(
    "variant", ["original", "2x", "5x", "10x", "negative", "offset", "zscore"]
)
@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (4, 2)])
def test_required_matrix_matches_antropy(
    name: str, variant: str, order: int, delay: int
) -> None:
    signal = required_signals()[name]
    if variant == "2x":
        signal = 2.0 * signal
    elif variant == "5x":
        signal = 5.0 * signal
    elif variant == "10x":
        signal = 10.0 * signal
    elif variant == "negative":
        signal = -2.0 * signal
    elif variant == "offset":
        signal = signal + 10.0
    elif variant == "zscore":
        signal = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    actual = svd_entropy(signal, order=order, delay=delay)
    centered = signal - np.mean(signal)
    expected_bits = antropy.svd_entropy(centered, order=order, delay=delay, normalize=False)
    assert actual == pytest.approx(expected_bits * np.log(2.0), abs=2e-15)


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize("delay", [1, 2, 3])
def test_randomized_normalized_matrix_matches_antropy(order: int, delay: int) -> None:
    signal = np.random.default_rng(order * 100 + delay).normal(size=180)
    actual = svd_entropy(signal, order=order, delay=delay, normalize=True)
    centered = signal - np.mean(signal)
    expected = antropy.svd_entropy(centered, order=order, delay=delay, normalize=True)
    assert actual == pytest.approx(expected, abs=2e-15)


def test_constant_input_is_an_intentional_degeneracy_exception() -> None:
    signal = np.ones(20)
    assert np.isnan(svd_entropy(signal))
    assert np.isfinite(antropy.svd_entropy(signal))
