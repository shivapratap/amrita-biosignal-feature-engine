"""Frequency-domain features computed from an explicit shared PSD."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid, trapezoid

from .diagnostics import FrequencyResolutionWarning
from .psd import PSDResult

FloatArray = NDArray[np.float64]


def _validate_psd(psd: PSDResult) -> tuple[FloatArray, FloatArray]:
    if not isinstance(psd, PSDResult):
        raise TypeError("psd must be a PSDResult")
    frequencies = psd.frequencies
    values = psd.values
    if frequencies.ndim != 1 or values.ndim != 1 or frequencies.shape != values.shape:
        raise ValueError("PSD frequencies and values must be aligned one-dimensional arrays")
    if frequencies.size < 2 or not np.all(np.isfinite(frequencies)):
        raise ValueError("PSD must contain at least two finite frequency bins")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("PSD values must be finite and nonnegative")
    if not np.all(np.diff(frequencies) > 0):
        raise ValueError("PSD frequencies must be strictly increasing")
    return frequencies, values


def _total_power(frequencies: FloatArray, values: FloatArray) -> float:
    return float(trapezoid(values, frequencies))


def peak_frequency(psd: PSDResult) -> float:
    """Return the frequency of maximum PSD, or ``NaN`` for zero spectral mass."""
    frequencies, values = _validate_psd(psd)
    if _total_power(frequencies, values) <= 0:
        return float("nan")
    return float(frequencies[int(np.argmax(values))])


def mean_frequency(psd: PSDResult) -> float:
    """Return the power-weighted mean frequency."""
    frequencies, values = _validate_psd(psd)
    total = _total_power(frequencies, values)
    if total <= 0:
        return float("nan")
    return float(trapezoid(frequencies * values, frequencies) / total)


def _frequency_at_power_fraction(psd: PSDResult, fraction: float) -> float:
    frequencies, values = _validate_psd(psd)
    try:
        fraction = float(fraction)
    except (TypeError, ValueError) as exc:
        raise TypeError("fraction must be a real number") from exc
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be finite and in (0, 1]")
    cumulative = cumulative_trapezoid(values, frequencies, initial=0.0)
    total = float(cumulative[-1])
    if total <= 0:
        return float("nan")
    return float(np.interp(fraction * total, cumulative, frequencies))


def median_frequency(psd: PSDResult) -> float:
    """Return the frequency below which half of integrated power lies."""
    return _frequency_at_power_fraction(psd, 0.5)


def spectral_edge_frequency(psd: PSDResult, *, fraction: float = 0.95) -> float:
    """Return the frequency below which ``fraction`` of integrated power lies."""
    return _frequency_at_power_fraction(psd, fraction)


def spectral_entropy(psd: PSDResult, *, normalize: bool = True) -> float:
    """Return Shannon entropy of normalized PSD-bin powers in bits."""
    _, values = _validate_psd(psd)
    total = float(np.sum(values))
    if total <= 0:
        return float("nan")
    probabilities = values / total
    positive = probabilities > 0
    entropy = float(-np.sum(probabilities[positive] * np.log2(probabilities[positive])))
    if normalize:
        if values.size <= 1:
            return float("nan")
        entropy /= float(np.log2(values.size))
    return entropy


def _validate_band(psd: PSDResult, band: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(band, tuple) or len(band) != 2:
        raise TypeError("band must be a (low_hz, high_hz) tuple")
    try:
        low, high = float(band[0]), float(band[1])
    except (TypeError, ValueError) as exc:
        raise TypeError("band limits must be real numbers") from exc
    if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high <= low:
        raise ValueError("band must satisfy 0 <= low_hz < high_hz")
    nyquist = psd.sampling_frequency / 2.0
    if high > nyquist:
        raise ValueError(f"band high edge {high} exceeds Nyquist frequency {nyquist}")
    return low, high


def _effective_resolution(psd: PSDResult) -> float:
    """Resolution relevant to resolving band power for the configured method."""
    if psd.effective_bandwidth is not None:
        return max(psd.bin_spacing, psd.effective_bandwidth)
    return psd.bin_spacing


def _warn_if_underresolved(psd: PSDResult, low: float, high: float) -> None:
    resolution = _effective_resolution(psd)
    if high - low < resolution:
        warnings.warn(
            f"band width {high - low:g} Hz is narrower than PSD effective resolution "
            f"{resolution:g} Hz",
            FrequencyResolutionWarning,
            stacklevel=3,
        )


def _band_arrays(psd: PSDResult, band: tuple[float, float]) -> tuple[FloatArray, FloatArray]:
    frequencies, values = _validate_psd(psd)
    low, high = _validate_band(psd, band)
    interior = (frequencies > low) & (frequencies < high)
    band_frequencies = np.concatenate(([low], frequencies[interior], [high]))
    band_values = np.interp(band_frequencies, frequencies, values)
    return (
        np.asarray(band_frequencies, dtype=np.float64),
        np.asarray(band_values, dtype=np.float64),
    )


def _integrated_band_power(psd: PSDResult, band: tuple[float, float]) -> float:
    frequencies, values = _band_arrays(psd, band)
    return float(trapezoid(values, frequencies))


def band_power(
    psd: PSDResult,
    *,
    band: tuple[float, float],
    relative: bool = False,
) -> float:
    """Integrate absolute or relative power over an explicit closed band."""
    low, high = _validate_band(psd, band)
    _warn_if_underresolved(psd, low, high)
    power = _integrated_band_power(psd, (low, high))
    if not relative:
        return power
    all_frequencies, all_values = _validate_psd(psd)
    total = _total_power(all_frequencies, all_values)
    return power / total if total > 0 else float("nan")


def band_power_ratio(
    psd: PSDResult,
    *,
    numerator_band: tuple[float, float],
    denominator_band: tuple[float, float],
) -> float:
    """Return an explicitly defined numerator-band/denominator-band ratio."""
    numerator_low, numerator_high = _validate_band(psd, numerator_band)
    denominator_low, denominator_high = _validate_band(psd, denominator_band)
    _warn_if_underresolved(psd, numerator_low, numerator_high)
    _warn_if_underresolved(psd, denominator_low, denominator_high)
    numerator = _integrated_band_power(psd, (numerator_low, numerator_high))
    denominator = _integrated_band_power(psd, (denominator_low, denominator_high))
    return numerator / denominator if denominator > 0 else float("nan")


__all__ = [
    "band_power",
    "band_power_ratio",
    "mean_frequency",
    "median_frequency",
    "peak_frequency",
    "spectral_edge_frequency",
    "spectral_entropy",
]
