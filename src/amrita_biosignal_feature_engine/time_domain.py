"""Time-domain features for pre-windowed one-dimensional signals.

Every public function validates its input, performs no display rounding, and
returns a Python ``float``. A constant signal is valid: features that remain
mathematically defined return their genuine values, while skewness and
kurtosis return ``NaN`` because standardized higher moments are undefined at
zero variance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .validation import validate_nonnegative_threshold, validate_signal

__all__ = [
    "integrated_absolute_value",
    "kurtosis",
    "maximum",
    "mean",
    "mean_absolute_value",
    "median",
    "minimum",
    "peak_to_peak",
    "root_mean_square",
    "skewness",
    "slope_sign_change_count",
    "standard_deviation",
    "sum_value",
    "variance",
    "waveform_length",
    "zero_crossing_count",
]


def minimum(signal: ArrayLike) -> float:
    """Return the minimum sample value."""
    return float(np.min(validate_signal(signal)))


def maximum(signal: ArrayLike) -> float:
    """Return the maximum sample value."""
    return float(np.max(validate_signal(signal)))


def sum_value(signal: ArrayLike) -> float:
    """Return the sample sum using the canonical API name ``sum_value``.

    No separate ``total`` or ``summation`` alias is exposed; this avoids
    representing one calculation as multiple distinct features.
    """
    return float(np.sum(validate_signal(signal)))


def mean(signal: ArrayLike) -> float:
    """Return the arithmetic mean."""
    return float(np.mean(validate_signal(signal)))


def median(signal: ArrayLike) -> float:
    """Return the median sample value."""
    return float(np.median(validate_signal(signal)))


def standard_deviation(signal: ArrayLike) -> float:
    """Return population standard deviation (``ddof=0``)."""
    return float(np.std(validate_signal(signal), ddof=0))


def variance(signal: ArrayLike) -> float:
    """Return population variance (``ddof=0``)."""
    return float(np.var(validate_signal(signal), ddof=0))


def kurtosis(signal: ArrayLike) -> float:
    """Return Fisher excess kurtosis using SciPy's bias-corrected estimator.

    Returns ``NaN`` for constant signals and for signals shorter than four
    samples, where this estimator is not meaningful.
    """
    data = validate_signal(signal)
    if data.size < 4 or np.ptp(data) == 0:
        return float("nan")
    return float(stats.kurtosis(data, fisher=True, bias=False))


def skewness(signal: ArrayLike) -> float:
    """Return skewness using SciPy's bias-corrected estimator.

    Returns ``NaN`` for constant signals and for signals shorter than three
    samples, where this estimator is not meaningful.
    """
    data = validate_signal(signal)
    if data.size < 3 or np.ptp(data) == 0:
        return float("nan")
    return float(stats.skew(data, bias=False))


def mean_absolute_value(signal: ArrayLike) -> float:
    """Return the mean absolute sample amplitude."""
    return float(np.mean(np.abs(validate_signal(signal))))


def root_mean_square(signal: ArrayLike) -> float:
    """Return root mean square amplitude."""
    data = validate_signal(signal)
    return float(np.sqrt(np.mean(np.square(data))))


def peak_to_peak(signal: ArrayLike) -> float:
    """Return maximum minus minimum amplitude."""
    return float(np.ptp(validate_signal(signal)))


def integrated_absolute_value(signal: ArrayLike) -> float:
    """Return the discrete integrated absolute value, ``sum(abs(x))``.

    This sample-domain definition intentionally has no sampling-frequency
    factor. Callers needing a time integral should divide by their sampling
    frequency explicitly.
    """
    return float(np.sum(np.abs(validate_signal(signal))))


def waveform_length(signal: ArrayLike) -> float:
    """Return ``sum(abs(diff(x)))``, the cumulative sample-to-sample change."""
    data = validate_signal(signal)
    return float(np.sum(np.abs(np.diff(data))))


def zero_crossing_count(signal: ArrayLike) -> float:
    """Count sign transitions after removing samples exactly equal to zero.

    Removing zeros treats ``[-1, 0, 1]`` as one crossing and prevents a run of
    zeros from creating multiple crossings. An all-zero signal has zero
    crossings.
    """
    data = validate_signal(signal)
    nonzero = data[data != 0]
    if nonzero.size < 2:
        return 0.0
    return float(np.count_nonzero(np.signbit(nonzero[1:]) != np.signbit(nonzero[:-1])))


def slope_sign_change_count(signal: ArrayLike, *, threshold: float = 0.0) -> float:
    """Count interior slope reversals whose adjacent slopes differ sufficiently.

    At sample ``i``, a change is counted when the slopes on either side have
    opposite signs and ``abs(slope_left - slope_right) >= threshold``. Flat
    slopes are not sign changes. ``threshold`` is expressed in signal units
    and defaults to zero; it is never inferred from a hidden scale constant.
    """
    data = validate_signal(signal)
    threshold = validate_nonnegative_threshold(threshold)
    if data.size < 3:
        return 0.0
    slopes = np.diff(data)
    reversals = slopes[:-1] * slopes[1:] < 0
    large_enough = np.abs(slopes[:-1] - slopes[1:]) >= threshold
    return float(np.count_nonzero(reversals & large_enough))
