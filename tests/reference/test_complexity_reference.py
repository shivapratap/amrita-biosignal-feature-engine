from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.complexity import (
    hjorth_complexity,
    hjorth_mobility,
    katz_fractal_dimension,
    petrosian_fractal_dimension,
)

if os.environ.get("ABFE_RUN_REFERENCE") != "1":
    pytest.skip("set ABFE_RUN_REFERENCE=1 to run external references", allow_module_level=True)

antropy = pytest.importorskip("antropy")

pytestmark = pytest.mark.reference


def required_signals() -> dict[str, NDArray[np.float64]]:
    time = np.linspace(0.0, 1.0, 128, endpoint=False)
    return {
        "periodic": np.sin(2.0 * np.pi * 7.0 * time),
        "white_noise": np.random.default_rng(321).normal(size=128),
        "chirp": np.sin(2.0 * np.pi * (2.0 * time + 10.0 * time**2)),
    }


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (3.0, 10.0), (-2.0, -4.0)])
def test_hjorth_and_katz_match_antropy(
    name: str, scale: float, offset: float
) -> None:
    signal = scale * required_signals()[name] + offset
    expected_mobility, expected_complexity = antropy.hjorth_params(signal)
    assert hjorth_mobility(signal) == pytest.approx(expected_mobility, abs=2e-15)
    assert hjorth_complexity(signal) == pytest.approx(expected_complexity, abs=2e-15)
    assert katz_fractal_dimension(signal) == pytest.approx(
        antropy.katz_fd(signal), abs=2e-15
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_petrosian_matches_antropy_without_zero_derivative_plateaus(
    name: str,
) -> None:
    signal = required_signals()[name]
    assert np.all(np.diff(signal) != 0.0)
    assert petrosian_fractal_dimension(signal) == pytest.approx(
        antropy.petrosian_fd(signal), abs=2e-15
    )


def test_petrosian_plateau_policy_is_an_intentional_reference_difference() -> None:
    signal = np.array([0.0, -1.0, -1.0, -2.0])
    assert petrosian_fractal_dimension(signal) == 1.0
    assert antropy.petrosian_fd(signal) > 1.0
