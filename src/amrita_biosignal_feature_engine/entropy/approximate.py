"""Approximate entropy using the standard self-match-inclusive definition."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

from ..validation import FloatArray, validate_signal


def _validate_order(order: int) -> int:
    if isinstance(order, bool | np.bool_) or not isinstance(order, int | np.integer):
        raise TypeError("order must be an integer")
    result = int(order)
    if result < 1:
        raise ValueError("order must be at least 1")
    return result


def _resolve_tolerance(data: FloatArray, tolerance: float | None) -> float:
    if tolerance is None:
        standard_deviation = float(np.std(data, ddof=0))
        if standard_deviation == 0.0:
            return float("nan")
        return 0.2 * standard_deviation
    if isinstance(tolerance, bool | np.bool_) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number")
    result = float(tolerance)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("tolerance must be finite and positive")
    return result


def _embed(data: FloatArray, dimension: int) -> FloatArray:
    template_count = data.size - dimension + 1
    return np.column_stack(
        [data[offset : offset + template_count] for offset in range(dimension)]
    )


def _phi(data: FloatArray, dimension: int, tolerance: float) -> float:
    templates = _embed(data, dimension)
    distances = cdist(templates, templates, metric="chebyshev")
    counts = np.count_nonzero(distances <= tolerance, axis=1)
    probabilities = counts / templates.shape[0]
    return float(np.mean(np.log(probabilities)))


def approximate_entropy(
    signal: ArrayLike,
    *,
    order: int = 2,
    tolerance: float | None = None,
) -> float:
    """Return Pincus approximate entropy using natural logarithms.

    Chebyshev-distance matches include each template's self-match. For
    dimension ``m``, all ``N - m + 1`` templates are used; for ``m + 1``, all
    ``N - m`` templates are used. Each correlation sum is normalized by its
    own template count before taking the mean natural logarithm.

    When ``tolerance`` is ``None``, it is ``0.2`` times the population standard
    deviation (``ddof=0``), making the default calculation invariant to any
    nonzero affine rescaling. Constant input returns ``NaN`` under ABFE's
    entropy degeneracy policy. An explicit tolerance must be finite and
    positive.
    """
    order = _validate_order(order)
    data = validate_signal(signal, minimum_length=order + 2)
    resolved_tolerance = _resolve_tolerance(data, tolerance)
    if np.isnan(resolved_tolerance):
        return float("nan")
    return _phi(data, order, resolved_tolerance) - _phi(
        data, order + 1, resolved_tolerance
    )


__all__ = ["approximate_entropy"]
