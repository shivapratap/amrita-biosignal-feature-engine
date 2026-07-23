from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import fuzzy_entropy

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
def test_records_corrected_paper_definition_against_dihc(
    name: str, variant: str
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
    actual = fuzzy_entropy(signal, order=2, delay=1)
    legacy = dihc_kernels.fuzzy_entropy(signal, m=2, tau=1, r_factor=0.2)
    assert np.isfinite(actual)
    assert np.isfinite(legacy)
    assert not np.isclose(actual, legacy, rtol=1e-10, atol=1e-12)


def test_regular_ramp_corrects_dihc_negative_entropy() -> None:
    signal = np.arange(64, dtype=float)
    assert fuzzy_entropy(signal) == pytest.approx(0.0, abs=1e-15)
    assert dihc_kernels.fuzzy_entropy(signal) == pytest.approx(
        -0.015497710010711062, abs=1e-15
    )


def test_constant_input_corrects_dihc_degeneracy_failure() -> None:
    signal = np.ones(20)
    assert np.isnan(fuzzy_entropy(signal))
    with pytest.raises(ZeroDivisionError):
        dihc_kernels.fuzzy_entropy(signal)
