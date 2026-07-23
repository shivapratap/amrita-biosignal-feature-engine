"""Fractal-dimension and complexity features for one-dimensional signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import validate_signal


def _lz76_phrase_count(sequence: NDArray[np.uint8]) -> int:
    """Return the LZ76 exhaustive-history phrase count for a binary array."""
    complexity = 1
    prefix_length = 1
    substring_length = 1
    maximum_substring_length = 1
    pointer = 0

    while prefix_length + substring_length <= sequence.size:
        if (
            sequence[pointer + substring_length - 1]
            == sequence[prefix_length + substring_length - 1]
        ):
            substring_length += 1
        else:
            maximum_substring_length = max(
                substring_length, maximum_substring_length
            )
            pointer += 1
            if pointer == prefix_length:
                complexity += 1
                prefix_length += maximum_substring_length
                pointer = 0
                maximum_substring_length = 1
            substring_length = 1

    if substring_length != 1:
        complexity += 1
    return complexity


def lempel_ziv_complexity(signal: ArrayLike, *, normalize: bool = True) -> float:
    """Return LZ76 complexity after a deterministic median binary split.

    Samples greater than or equal to the signal median map to one; all others
    map to zero. The raw exhaustive-history phrase count is returned when
    ``normalize`` is false. Otherwise the count is multiplied by
    ``log2(n) / n``. Constant signals return ``NaN``.
    """
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    data = validate_signal(signal, minimum_length=2)
    if float(np.ptp(data)) == 0.0:
        return float("nan")
    binary = np.asarray(data >= np.median(data), dtype=np.uint8)
    phrase_count = _lz76_phrase_count(binary)
    if not normalize:
        return float(phrase_count)
    return float(phrase_count * np.log2(data.size) / data.size)


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
    "lempel_ziv_complexity",
    "petrosian_fractal_dimension",
]
