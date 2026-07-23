from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.signal.windows import dpss as scipy_dpss

from amrita_biosignal_feature_engine.psd import (
    MultitaperPSDConfig,
    PSDConfig,
    PSDResult,
    WelchPSDConfig,
    compute_psd,
)


def test_welch_matches_explicit_scipy_reference() -> None:
    rng = np.random.default_rng(42)
    signal = rng.normal(size=1000)
    config = WelchPSDConfig(
        window_length=2.0,
        overlap=1.0,
        detrend="constant",
        scaling="density",
        window="hann",
    )
    actual = compute_psd(signal, 100.0, config)
    frequencies, values = welch(
        signal,
        fs=100.0,
        window="hann",
        nperseg=200,
        noverlap=100,
        nfft=200,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )
    np.testing.assert_array_equal(actual.frequencies, frequencies)
    np.testing.assert_array_equal(actual.values, values)


def test_spectrum_scaling_is_density_times_resolution() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=800)
    density = compute_psd(signal, 100.0, WelchPSDConfig(scaling="density"))
    bin_power = compute_psd(signal, 100.0, WelchPSDConfig(scaling="bin_power"))
    np.testing.assert_allclose(
        bin_power.values, density.values * density.bin_spacing, rtol=0, atol=0
    )


def test_psd_result_records_complete_provenance() -> None:
    config = WelchPSDConfig(
        window_length=2.0,
        overlap=0.5,
        detrend="linear",
        scaling="density",
        window="boxcar",
    )
    result = compute_psd(np.arange(1000, dtype=float), 100.0, config)
    assert result.config == config
    assert result.sampling_frequency == 100.0
    assert result.signal_length == 1000
    assert result.samples_per_window == 200
    assert result.samples_overlap == 50
    assert result.segment_count == 6
    assert result.bin_spacing == 0.5
    assert result.effective_bandwidth is None


def test_psd_configuration_and_arrays_are_immutable() -> None:
    config = WelchPSDConfig()
    result = compute_psd(np.arange(400, dtype=float), 100.0, config)
    with pytest.raises(FrozenInstanceError):
        config.overlap = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        result.values[0] = 2.0


def test_method_specific_configs_reject_irrelevant_parameters() -> None:
    with pytest.raises(TypeError):
        WelchPSDConfig(bandwidth=4.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        MultitaperPSDConfig(window="hann")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("seconds", "expected_samples"),
    [(0.025, 2), (0.035, 4), (0.045, 4), (0.055, 6)],
)
def test_seconds_to_samples_uses_round_half_to_even(
    seconds: float, expected_samples: int
) -> None:
    result = compute_psd(
        np.arange(20, dtype=float),
        100.0,
        WelchPSDConfig(window_length=seconds, overlap=0.0, window="boxcar"),
    )
    assert result.samples_per_window == expected_samples


def test_psd_result_owns_and_protects_input_arrays() -> None:
    frequencies = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 1.0, 2.0, 0.0])
    result = PSDResult(
        frequencies=frequencies,
        values=values,
        bin_spacing=1.0,
        effective_bandwidth=None,
        sampling_frequency=6.0,
        signal_length=6,
        samples_per_window=6,
        samples_overlap=0,
        segment_count=1,
        config=WelchPSDConfig(window_length=1.0, overlap=0.0),
    )
    assert not np.shares_memory(result.frequencies, frequencies)
    assert not np.shares_memory(result.values, values)
    frequencies[0] = 99.0
    values[1] = 99.0
    np.testing.assert_array_equal(result.frequencies, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.values, [0.0, 1.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="read-only"):
        result.frequencies[0] = 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"frequencies": np.array([0.0, 2.0, 1.0, 3.0])},
        {"values": np.array([0.0, -1.0, 2.0, 0.0])},
        {"bin_spacing": 2.0},
        {"sampling_frequency": -6.0},
        {"samples_overlap": 6},
        {"segment_count": 2},
        {"effective_bandwidth": 2.0},
    ],
)
def test_psd_result_rejects_inconsistent_invariants(overrides: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "frequencies": np.array([0.0, 1.0, 2.0, 3.0]),
        "values": np.array([0.0, 1.0, 2.0, 0.0]),
        "bin_spacing": 1.0,
        "effective_bandwidth": None,
        "sampling_frequency": 6.0,
        "signal_length": 6,
        "samples_per_window": 6,
        "samples_overlap": 0,
        "segment_count": 1,
        "config": WelchPSDConfig(window_length=1.0, overlap=0.0),
    }
    arguments.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        PSDResult(**arguments)  # type: ignore[arg-type]


def test_multitaper_result_records_effective_bandwidth() -> None:
    result = compute_psd(
        np.random.default_rng(8).normal(size=1000),
        100.0,
        MultitaperPSDConfig(window_length=2.0, overlap=1.0, bandwidth=3.0),
    )
    assert result.bin_spacing == 0.5
    assert result.effective_bandwidth == 3.0


def test_multitaper_requests_floor_two_nw_candidate_tapers() -> None:
    observed_kmax: list[int] = []

    def recording_dpss(*args: Any, **kwargs: Any) -> Any:
        observed_kmax.append(int(kwargs["Kmax"]))
        return scipy_dpss(*args, **kwargs)

    with patch("amrita_biosignal_feature_engine.psd.dpss", side_effect=recording_dpss):
        compute_psd(
            np.random.default_rng(9).normal(size=1600),
            200.0,
            MultitaperPSDConfig(
                window_length=4.0,
                overlap=2.0,
                bandwidth=2.0,
                low_bias=False,
            ),
        )
    # NW = window_length * bandwidth / 2 = 4; floor(2*NW) = 8.
    assert observed_kmax == [8]


@pytest.mark.parametrize(("samples", "includes_nyquist"), [(199, False), (200, True)])
def test_odd_and_even_window_frequency_axes(samples: int, includes_nyquist: bool) -> None:
    sampling_frequency = 100.0
    result = compute_psd(
        np.random.default_rng(samples).normal(size=1000),
        sampling_frequency,
        WelchPSDConfig(
            window_length=samples / sampling_frequency,
            overlap=0.0,
            window="boxcar",
        ),
    )
    assert result.samples_per_window == samples
    assert bool(result.frequencies[-1] == sampling_frequency / 2.0) is includes_nyquist


@pytest.mark.parametrize("weighting", ["uniform", "eigen"])
def test_multitaper_detects_known_sinusoid(weighting: str) -> None:
    fs = 200.0
    time = np.arange(0.0, 8.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * 23.0 * time)
    config = MultitaperPSDConfig(
        window_length=2.0,
        overlap=1.0,
        detrend="constant",
        scaling="density",
        bandwidth=3.0,
        low_bias=True,
        multitaper_weighting=weighting,  # type: ignore[arg-type]
    )
    result = compute_psd(signal, fs, config)
    assert result.frequencies[np.argmax(result.values)] == pytest.approx(23.0, abs=1.0)
    assert np.all(result.values >= 0)
    assert np.all(np.isfinite(result.values))


def test_multitaper_density_integrates_to_signal_variance() -> None:
    rng = np.random.default_rng(123)
    signal = rng.normal(size=4000)
    result = compute_psd(
        signal,
        200.0,
        MultitaperPSDConfig(
            window_length=4.0,
            overlap=2.0,
            bandwidth=2.0,
        ),
    )
    integrated = trapezoid(result.values, result.frequencies)
    assert integrated == pytest.approx(np.var(signal), rel=0.08)


@pytest.mark.parametrize(
    "config",
    [
        WelchPSDConfig(window_length=0.0),
        WelchPSDConfig(overlap=2.0, window_length=2.0),
        MultitaperPSDConfig(bandwidth=0.1, window_length=2.0),
    ],
)
def test_invalid_psd_configuration_raises(config: PSDConfig) -> None:
    with pytest.raises(ValueError):
        compute_psd(np.arange(1000, dtype=float), 100.0, config)


def test_window_longer_than_signal_raises_instead_of_scipy_fallback() -> None:
    with pytest.raises(ValueError, match="requires 200 samples"):
        compute_psd(np.ones(100), 100.0, WelchPSDConfig(window_length=2.0, overlap=0.0))


def test_psd_rejects_nonfinite_signal() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_psd(
            [1.0, np.nan, 2.0],
            100.0,
            WelchPSDConfig(window_length=0.02, overlap=0.0),
        )
