"""Fractal-dimension and complexity features for one-dimensional signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .validation import validate_signal


def hjorth_mobility(signal: ArrayLike) -> float:
    """Return Hjorth mobility in samples⁻¹.

    Mobility is ``sqrt(var(diff(x)) / var(x))`` using population variances.
    A constant signal has undefined mobility and returns ``NaN``.
    """
    data = validate_signal(signal, minimum_length=2)
    signal_variance = float(np.var(data, ddof=0))
    if signal_variance == 0.0:
        return float("nan")
    derivative_variance = float(np.var(np.diff(data), ddof=0))
    return float(np.sqrt(derivative_variance / signal_variance))


def hjorth_complexity(signal: ArrayLike) -> float:
    """Return dimensionless Hjorth complexity.

    Complexity is the mobility of the first difference divided by the
    mobility of the signal. A constant signal or constant first difference
    has undefined complexity and returns ``NaN``.
    """
    data = validate_signal(signal, minimum_length=3)
    first_difference = np.diff(data)
    signal_variance = float(np.var(data, ddof=0))
    derivative_variance = float(np.var(first_difference, ddof=0))
    if signal_variance == 0.0 or derivative_variance == 0.0:
        return float("nan")
    second_derivative_variance = float(np.var(np.diff(first_difference), ddof=0))
    mobility = np.sqrt(derivative_variance / signal_variance)
    derivative_mobility = np.sqrt(second_derivative_variance / derivative_variance)
    return float(derivative_mobility / mobility)


def petrosian_fractal_dimension(signal: ArrayLike) -> float:
    """Return Petrosian fractal dimension.

    Zero first differences are removed before derivative sign transitions are
    counted, so a flat run does not create artificial transitions.
    """
    data = validate_signal(signal, minimum_length=3)
    differences = np.diff(data)
    nonzero_differences = differences[differences != 0.0]
    if nonzero_differences.size < 2:
        sign_changes = 0
    else:
        sign_changes = int(
            np.count_nonzero(
                np.signbit(nonzero_differences[1:])
                != np.signbit(nonzero_differences[:-1])
            )
        )
    sample_count = int(data.size)
    log_length = np.log10(sample_count)
    denominator = log_length + np.log10(
        sample_count / (sample_count + 0.4 * sign_changes)
    )
    return float(log_length / denominator)


def katz_fractal_dimension(signal: ArrayLike) -> float:
    """Return Katz fractal dimension.

    Constant signals and geometries with an indeterminate logarithmic ratio
    return ``NaN``.
    """
    data = validate_signal(signal, minimum_length=2)
    distances = np.abs(np.diff(data))
    path_length = float(np.sum(distances))
    mean_step = float(np.mean(distances))
    maximum_displacement = float(np.max(np.abs(data - data[0])))
    if path_length == 0.0 or mean_step == 0.0 or maximum_displacement == 0.0:
        return float("nan")
    numerator = float(np.log10(path_length / mean_step))
    denominator = float(np.log10(maximum_displacement / mean_step))
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


__all__ = [
    "hjorth_complexity",
    "hjorth_mobility",
    "katz_fractal_dimension",
    "petrosian_fractal_dimension",
]
