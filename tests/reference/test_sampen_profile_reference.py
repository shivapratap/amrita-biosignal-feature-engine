from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import sample_entropy_profile

if os.environ.get("ABFE_RUN_REFERENCE") != "1":
    pytest.skip("set ABFE_RUN_REFERENCE=1 to run external references", allow_module_level=True)

sampen_profile_package = pytest.importorskip("sampen_profile")

pytestmark = pytest.mark.reference


def required_signals() -> dict[str, NDArray[np.float64]]:
    periodic_time = np.linspace(0.0, 4.0 * np.pi, 96, endpoint=False)
    chirp_time = np.linspace(0.0, 1.0, 96, endpoint=False)
    return {
        "periodic": np.sin(periodic_time),
        "white_noise": np.random.default_rng(123).normal(size=96),
        "chirp": np.sin(2.0 * np.pi * (2.0 * chirp_time + 8.0 * chirp_time**2)),
    }


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("variant", ["original", "2x", "5x", "10x", "zscore"])
def test_required_profile_matrix_matches_authority_exactly(name: str, variant: str) -> None:
    signal = required_signals()[name]
    if variant == "2x":
        signal = 2.0 * signal
    elif variant == "5x":
        signal = 5.0 * signal
    elif variant == "10x":
        signal = 10.0 * signal
    elif variant == "zscore":
        signal = (signal - np.mean(signal)) / np.std(signal)

    actual = sample_entropy_profile(signal, order=2)
    expected = sampen_profile_package.sample_entropy_profile(signal, m=2)
    np.testing.assert_array_equal(actual.r_values, expected["r_values"])
    np.testing.assert_array_equal(actual.se_profile, expected["se_profile"])


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("length", [20, 50, 150])
@pytest.mark.parametrize("seed", [1, 42, 999])
def test_randomized_profile_matrix_matches_authority_exactly(
    order: int, length: int, seed: int
) -> None:
    signal = np.random.default_rng(seed).normal(size=length)
    actual = sample_entropy_profile(signal, order=order)
    expected = sampen_profile_package.sample_entropy_profile(signal, m=order)
    np.testing.assert_array_equal(actual.r_values, expected["r_values"])
    np.testing.assert_array_equal(actual.se_profile, expected["se_profile"])
