"""Permutation entropy from ordinal-pattern probabilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..validation import FloatArray, validate_signal


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _ordinal_patterns(data: FloatArray, order: int, delay: int) -> NDArray[np.intp]:
    pattern_count = data.size - (order - 1) * delay
    embedded = np.column_stack(
        [data[offset * delay : offset * delay + pattern_count] for offset in range(order)]
    )
    # Stable sorting gives tied values a deterministic left-to-right ordering.
    return np.argsort(embedded, axis=1, kind="stable")


def permutation_entropy(
    signal: ArrayLike,
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> float:
    """Return Bandt-Pompe permutation entropy using natural logarithms.

    Each delayed embedding is replaced by its ordinal pattern. Pattern
    probabilities are estimated from their observed relative frequencies and
    entropy is ``-sum(p * log(p))``. When ``normalize`` is true, the result is
    divided by ``log(order!)`` and therefore lies in ``[0, 1]``.

    Ties are resolved deterministically by stable left-to-right index order.
    Input that produces fewer than two distinct ordinal patterns returns
    ``NaN`` under ABFE's entropy degeneracy policy. ``order`` must be at least
    2, ``delay`` must be at least 1, and the signal must contain at least two
    delayed embeddings.
    """
    order = _validate_integer(order, name="order", minimum=2)
    delay = _validate_integer(delay, name="delay", minimum=1)
    minimum_length = (order - 1) * delay + 2
    data = validate_signal(signal, minimum_length=minimum_length)

    patterns = _ordinal_patterns(data, order, delay)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    if counts.size < 2:
        return float("nan")

    probabilities = counts / counts.sum()
    result = -float(np.sum(probabilities * np.log(probabilities)))
    if normalize:
        result /= math.lgamma(order + 1)
    return result


__all__ = ["permutation_entropy"]
