from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.frequency_domain import spectral_entropy
from amrita_biosignal_feature_engine.psd import WelchPSDConfig, compute_psd

if os.environ.get("ABFE_RUN_REFERENCE") != "1":
    pytest.skip("set ABFE_RUN_REFERENCE=1 to run external references", allow_module_level=True)

antropy = pytest.importorskip("antropy")

pytestmark = pytest.mark.reference

SAMPLING_FREQUENCY = 128.0
WINDOW_SAMPLES = 64


def required_signals() -> dict[str, NDArray[np.float64]]:
    periodic_time = np.arange(128, dtype=np.float64) / SAMPLING_FREQUENCY
    chirp_time = periodic_time
    return {
        "periodic": np.sin(2.0 * np.pi * 8.0 * periodic_time),
        "white_noise": np.random.default_rng(321).normal(size=128),
        "chirp": np.sin(2.0 * np.pi * (3.0 * chirp_time + 20.0 * chirp_time**2)),
    }


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize(
    "variant", ["original", "2x", "5x", "10x", "negative", "offset", "zscore"]
)
@pytest.mark.parametrize("scaling", ["density", "bin_power"])
@pytest.mark.parametrize("normalize", [False, True])
def test_required_welch_matrix_matches_antropy(
    name: str,
    variant: str,
    scaling: str,
    normalize: bool,
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

    config = WelchPSDConfig(
        window_length=WINDOW_SAMPLES / SAMPLING_FREQUENCY,
        overlap=(WINDOW_SAMPLES // 2) / SAMPLING_FREQUENCY,
        window="hann",
        detrend="constant",
        scaling=scaling,  # type: ignore[arg-type]
    )
    actual = spectral_entropy(
        compute_psd(signal, SAMPLING_FREQUENCY, config), normalize=normalize
    )
    expected = antropy.spectral_entropy(
        signal,
        sf=SAMPLING_FREQUENCY,
        method="welch",
        nperseg=WINDOW_SAMPLES,
        normalize=normalize,
    )
    assert actual == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("seed", [1, 42, 999])
@pytest.mark.parametrize("nperseg", [32, 64, 128])
def test_randomized_welch_matrix_matches_antropy(seed: int, nperseg: int) -> None:
    signal = np.random.default_rng(seed).normal(size=256)
    config = WelchPSDConfig(
        window_length=nperseg / SAMPLING_FREQUENCY,
        overlap=(nperseg // 2) / SAMPLING_FREQUENCY,
    )
    actual = spectral_entropy(
        compute_psd(signal, SAMPLING_FREQUENCY, config), normalize=True
    )
    expected = antropy.spectral_entropy(
        signal,
        sf=SAMPLING_FREQUENCY,
        method="welch",
        nperseg=nperseg,
        normalize=True,
    )
    assert actual == pytest.approx(expected, abs=2e-15)


def test_constant_input_corrects_antropy_negative_zero_fallback() -> None:
    signal = np.ones(128)
    config = WelchPSDConfig(window_length=0.5, overlap=0.25)
    assert np.isnan(spectral_entropy(compute_psd(signal, SAMPLING_FREQUENCY, config)))
    with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
        expected = antropy.spectral_entropy(
            signal,
            sf=SAMPLING_FREQUENCY,
            method="welch",
            nperseg=WINDOW_SAMPLES,
            normalize=True,
        )
    assert expected == pytest.approx(-0.0, abs=0.0)
