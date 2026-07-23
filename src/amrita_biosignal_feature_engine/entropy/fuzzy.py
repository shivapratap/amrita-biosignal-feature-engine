"""Shape-based fuzzy entropy following Chen et al. (2007)."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist

from ..validation import FloatArray, validate_signal


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_positive_real(value: float, *, name: str) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _centered_embedding(
    data: FloatArray, dimension: int, delay: int, template_count: int
) -> FloatArray:
    embedded = np.column_stack(
        [data[offset * delay : offset * delay + template_count] for offset in range(dimension)]
    )
    return np.asarray(
        embedded - np.mean(embedded, axis=1, keepdims=True), dtype=np.float64
    )


def _similarity_mean(
    data: FloatArray,
    dimension: int,
    delay: int,
    template_count: int,
    tolerance: float,
    exponent: float,
) -> float:
    templates = _centered_embedding(data, dimension, delay, template_count)
    distances = pdist(templates, metric="chebyshev")
    with np.errstate(over="ignore"):
        similarities = np.exp(
            -np.log(2.0) * np.power(distances / tolerance, exponent)
        )
    return float(np.mean(similarities))


def fuzzy_entropy(
    signal: ArrayLike,
    *,
    order: int = 2,
    delay: int = 1,
    tolerance_factor: float = 0.2,
    exponent: float = 2.0,
) -> float:
    """Return shape-based fuzzy entropy using natural logarithms.

    Following Chen et al. (2007), every delayed template has its local mean
    removed before pairwise Chebyshev distances are measured. Self-matches
    are excluded, and the same ``N - order * delay`` templates are used at
    dimensions ``order`` and ``order + 1``. Similarity is
    ``exp(-log(2) * (distance / tolerance) ** exponent)``, where ``tolerance``
    is ``tolerance_factor * std(signal, ddof=0)``. Thus a distance equal to
    the tolerance has membership one half.

    The default definition is invariant to nonzero affine rescaling and
    z-scoring. Constant input or complete similarity underflow returns
    ``NaN`` under ABFE's entropy degeneracy policy. A numeric zero remains a
    meaningful result for a perfectly regular nonconstant signal.
    """
    order = _validate_integer(order, name="order", minimum=1)
    delay = _validate_integer(delay, name="delay", minimum=1)
    tolerance_factor = _validate_positive_real(
        tolerance_factor, name="tolerance_factor"
    )
    exponent = _validate_positive_real(exponent, name="exponent")
    minimum_length = order * delay + 2
    data = validate_signal(signal, minimum_length=minimum_length)

    standard_deviation = float(np.std(data, ddof=0))
    if standard_deviation == 0.0:
        return float("nan")
    tolerance = tolerance_factor * standard_deviation
    template_count = data.size - order * delay
    phi_order = _similarity_mean(
        data, order, delay, template_count, tolerance, exponent
    )
    phi_next = _similarity_mean(
        data, order + 1, delay, template_count, tolerance, exponent
    )
    if phi_order == 0.0 or phi_next == 0.0:
        return float("nan")
    return float(np.log(phi_order) - np.log(phi_next))


__all__ = ["fuzzy_entropy"]
