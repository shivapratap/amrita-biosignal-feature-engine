"""Distribution entropy of delayed-embedding distances."""

from __future__ import annotations

import math

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


def _embed(data: FloatArray, order: int, delay: int) -> FloatArray:
    template_count = data.size - (order - 1) * delay
    return np.column_stack(
        [data[offset * delay : offset * delay + template_count] for offset in range(order)]
    )


def distribution_entropy(
    signal: ArrayLike,
    *,
    order: int = 2,
    delay: int = 1,
    n_bins: int = 500,
    normalize: bool = True,
) -> float:
    """Return entropy of the embedding-distance distribution.

    All unordered, off-diagonal Chebyshev distances between delayed
    embeddings are placed into ``n_bins`` equal-width bins spanning the
    observed minimum and maximum. The maximum belongs to the final bin.
    Shannon entropy is computed with natural logarithms and, by default,
    divided by ``log(n_bins)`` to produce a value in ``[0, 1]``.

    Nonzero affine rescaling and z-scoring preserve the result because the
    histogram spans the observed distance range. If every pairwise distance
    is identical, the empirical distance distribution has no estimable range
    and ABFE returns ``NaN`` under its entropy degeneracy policy.
    """
    order = _validate_integer(order, name="order", minimum=1)
    delay = _validate_integer(delay, name="delay", minimum=1)
    n_bins = _validate_integer(n_bins, name="n_bins", minimum=2)
    minimum_length = (order - 1) * delay + 2
    data = validate_signal(signal, minimum_length=minimum_length)

    distances = pdist(_embed(data, order, delay), metric="chebyshev")
    minimum_distance = float(np.min(distances))
    maximum_distance = float(np.max(distances))
    if minimum_distance == maximum_distance:
        return float("nan")

    counts, _ = np.histogram(
        distances, bins=n_bins, range=(minimum_distance, maximum_distance)
    )
    probabilities = counts[counts > 0].astype(np.float64) / distances.size
    result = -float(np.sum(probabilities * np.log(probabilities)))
    if normalize:
        result /= math.log(n_bins)
    return result


__all__ = ["distribution_entropy"]
