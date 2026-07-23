from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from amrita_biosignal_feature_engine.diagnostics import FrequencyResolutionWarning
from amrita_biosignal_feature_engine.frequency_domain import (
    band_power,
    band_power_ratio,
    mean_frequency,
    median_frequency,
    peak_frequency,
    spectral_edge_frequency,
    spectral_entropy,
)
from amrita_biosignal_feature_engine.psd import (
    MultitaperPSDConfig,
    PSDResult,
    WelchPSDConfig,
    compute_psd,
)


def sinusoid(
    frequency: float, *, amplitude: float = 1.0
) -> tuple[NDArray[np.float64], float]:
    fs = 200.0
    time = np.arange(0.0, 10.0, 1.0 / fs)
    return amplitude * np.sin(2 * np.pi * frequency * time), fs


def test_frequency_locations_for_bin_centered_sinusoid() -> None:
    signal, fs = sinusoid(20.0)
    psd = compute_psd(
        signal,
        fs,
        WelchPSDConfig(window_length=2.0, overlap=1.0, window="hann", detrend="constant"),
    )
    assert peak_frequency(psd) == 20.0
    assert mean_frequency(psd) == pytest.approx(20.0, abs=0.05)
    assert median_frequency(psd) == pytest.approx(20.0, abs=0.5)
    assert spectral_edge_frequency(psd) == pytest.approx(20.5, abs=0.5)


def test_mean_frequency_matches_direct_trapezoidal_definition() -> None:
    signal, fs = sinusoid(12.0)
    psd = compute_psd(signal, fs, WelchPSDConfig())
    expected = trapezoid(psd.frequencies * psd.values, psd.frequencies) / trapezoid(
        psd.values, psd.frequencies
    )
    assert mean_frequency(psd) == pytest.approx(expected, rel=1e-15)


def test_band_power_and_explicit_ratio_on_two_tones() -> None:
    low, fs = sinusoid(10.0, amplitude=2.0)
    high, _ = sinusoid(30.0, amplitude=1.0)
    psd = compute_psd(low + high, fs, WelchPSDConfig(window_length=2.0, overlap=1.0))
    low_power = band_power(psd, band=(8.0, 12.0))
    high_power = band_power(psd, band=(28.0, 32.0))
    assert low_power == pytest.approx(2.0, rel=0.02)
    assert high_power == pytest.approx(0.5, rel=0.02)
    assert band_power_ratio(
        psd,
        numerator_band=(8.0, 12.0),
        denominator_band=(28.0, 32.0),
    ) == pytest.approx(4.0, rel=0.03)


def test_relative_band_power_is_fraction_of_total_integrated_power() -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig())
    absolute = band_power(psd, band=(8.0, 12.0))
    relative = band_power(psd, band=(8.0, 12.0), relative=True)
    total = trapezoid(psd.values, psd.frequencies)
    assert relative == pytest.approx(absolute / total)
    assert relative == pytest.approx(1.0, abs=1e-10)


def test_narrow_band_emits_structured_resolution_warning() -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig(window_length=1.0, overlap=0.5))
    assert psd.bin_spacing == 1.0
    with pytest.warns(FrequencyResolutionWarning, match="narrower than PSD effective resolution"):
        value = band_power(psd, band=(9.8, 10.2))
    assert value > 0


def test_multitaper_band_narrower_than_effective_bandwidth_warns() -> None:
    """Regression: multitaper resolution is bandwidth, not raw FFT spacing."""
    signal, fs = sinusoid(10.0)
    psd = compute_psd(
        signal,
        fs,
        MultitaperPSDConfig(
            window_length=2.0,
            overlap=1.0,
            bandwidth=4.0,
        ),
    )
    assert psd.bin_spacing == 0.5
    assert psd.effective_bandwidth == 4.0
    band = (9.0, 11.0)  # 2 Hz: wider than bin spacing, narrower than 4 Hz bandwidth.
    with pytest.warns(FrequencyResolutionWarning, match="narrower than PSD effective resolution"):
        band_power(psd, band=band)


def test_resolved_band_does_not_emit_resolution_warning() -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig(window_length=1.0, overlap=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FrequencyResolutionWarning)
        assert band_power(psd, band=(8.0, 12.0)) > 0


def test_resolution_warning_points_to_public_caller() -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig(window_length=1.0, overlap=0.5))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
        band_power(psd, band=(9.8, 10.2))
    assert len(captured) == 1
    assert Path(captured[0].filename).resolve() == Path(__file__).resolve()
    assert captured[0].lineno == expected_line


def test_ratio_warns_once_for_each_underresolved_named_band() -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig(window_length=1.0, overlap=0.5))
    with pytest.warns(FrequencyResolutionWarning) as captured:
        band_power_ratio(
            psd,
            numerator_band=(9.8, 10.2),
            denominator_band=(20.0, 22.0),
        )
    assert len(captured) == 1


def test_spectral_entropy_matches_hand_computed_bin_probabilities() -> None:
    frequencies = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, 1.0, 2.0, 0.0])
    config = WelchPSDConfig(window_length=1.0, overlap=0.0)
    psd = PSDResult(frequencies, values, 1.0, None, 6.0, 6, 6, 0, 1, config)
    probabilities = np.array([0.25, 0.25, 0.5])
    expected = -np.sum(probabilities * np.log2(probabilities))
    assert spectral_entropy(psd, normalize=False) == pytest.approx(expected)
    assert spectral_entropy(psd, normalize=True) == pytest.approx(expected / np.log2(4))


def test_white_noise_has_higher_spectral_entropy_than_periodic_signal() -> None:
    periodic, fs = sinusoid(15.0)
    noise = np.random.default_rng(5).normal(size=periodic.size)
    config = WelchPSDConfig(window_length=2.0, overlap=1.0)
    assert spectral_entropy(compute_psd(noise, fs, config)) > spectral_entropy(
        compute_psd(periodic, fs, config)
    )


def test_chirp_spectral_edge_is_above_median() -> None:
    fs = 200.0
    time = np.arange(0.0, 10.0, 1.0 / fs)
    phase = 2 * np.pi * (5.0 * time + 0.5 * 4.0 * time**2)
    psd = compute_psd(
        np.sin(phase), fs, WelchPSDConfig(window_length=2.0, overlap=1.0)
    )
    assert spectral_edge_frequency(psd, fraction=0.95) > median_frequency(psd)


def test_frequency_features_obey_amplitude_scaling_laws() -> None:
    signal, fs = sinusoid(10.0)
    config = WelchPSDConfig()
    reference = compute_psd(signal, fs, config)
    for scale in (2.0, 5.0, 10.0):
        scaled = compute_psd(scale * signal, fs, config)
        assert peak_frequency(scaled) == peak_frequency(reference)
        assert mean_frequency(scaled) == pytest.approx(mean_frequency(reference), abs=1e-13)
        assert median_frequency(scaled) == pytest.approx(median_frequency(reference), abs=1e-13)
        assert spectral_edge_frequency(scaled) == pytest.approx(
            spectral_edge_frequency(reference), abs=1e-13
        )
        assert spectral_entropy(scaled) == pytest.approx(spectral_entropy(reference), abs=1e-13)
        assert band_power(scaled, band=(8.0, 12.0)) == pytest.approx(
            scale**2 * band_power(reference, band=(8.0, 12.0)), rel=1e-13
        )
        assert band_power(scaled, band=(8.0, 12.0), relative=True) == pytest.approx(
            band_power(reference, band=(8.0, 12.0), relative=True), abs=1e-13
        )


@pytest.mark.parametrize("scale", [-2.0, -5.0, -10.0])
def test_spectral_entropy_obeys_negative_amplitude_scaling(scale: float) -> None:
    signal, fs = sinusoid(10.0)
    config = WelchPSDConfig()
    assert spectral_entropy(compute_psd(scale * signal, fs, config)) == pytest.approx(
        spectral_entropy(compute_psd(signal, fs, config)), abs=1e-13
    )


def test_spectral_entropy_is_z_score_invariant_with_constant_detrending() -> None:
    signal = np.random.default_rng(77).normal(loc=4.0, scale=3.0, size=1000)
    z_scored = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    config = WelchPSDConfig(detrend="constant")
    assert spectral_entropy(compute_psd(z_scored, 100.0, config)) == pytest.approx(
        spectral_entropy(compute_psd(signal, 100.0, config)), abs=2e-15
    )


def test_constant_detrended_signal_has_nan_frequency_features() -> None:
    psd = compute_psd(np.ones(1000), 100.0, WelchPSDConfig())
    assert np.all(psd.values == 0)
    assert np.isnan(peak_frequency(psd))
    assert np.isnan(mean_frequency(psd))
    assert np.isnan(median_frequency(psd))
    assert np.isnan(spectral_edge_frequency(psd))
    assert np.isnan(spectral_entropy(psd))
    assert band_power(psd, band=(1.0, 10.0)) == 0.0
    assert np.isnan(band_power(psd, band=(1.0, 10.0), relative=True))
    assert np.isnan(
        band_power_ratio(psd, numerator_band=(1.0, 10.0), denominator_band=(10.0, 20.0))
    )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_spectral_edge_rejects_invalid_fraction(fraction: float) -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig())
    with pytest.raises(ValueError, match="fraction"):
        spectral_edge_frequency(psd, fraction=fraction)


@pytest.mark.parametrize(
    "band",
    [(-1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (0.0, 101.0), (np.nan, 2.0)],
)
def test_band_power_rejects_invalid_band(band: tuple[float, float]) -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig())
    with pytest.raises(ValueError):
        band_power(psd, band=band)


def test_bands_at_dc_and_nyquist_are_valid() -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig(window_length=2.0, overlap=1.0))
    assert band_power(psd, band=(0.0, 2.0)) >= 0.0
    assert band_power(psd, band=(98.0, 100.0)) >= 0.0


def test_band_exactly_one_bin_wide_does_not_warn() -> None:
    signal, fs = sinusoid(10.0)
    psd = compute_psd(signal, fs, WelchPSDConfig(window_length=2.0, overlap=1.0))
    assert psd.bin_spacing == 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", FrequencyResolutionWarning)
        assert band_power(psd, band=(9.75, 10.25)) > 0.0
