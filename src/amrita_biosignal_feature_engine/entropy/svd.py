"""Singular-value-decomposition entropy of a delayed embedding."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ..validation import FloatArray, validate_signal


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _embed(data: FloatArray, order: int, delay: int) -> FloatArray:
    embedding_count = data.size - (order - 1) * delay
    return np.column_stack(
        [data[offset * delay : offset * delay + embedding_count] for offset in range(order)]
    )


def svd_entropy(
    signal: ArrayLike,
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> float:
    """Return the natural-log entropy of normalized embedding singular values.

    The signal is mean-centered, then a delayed embedding with
    ``N - (order - 1) * delay`` rows is decomposed into singular values. After
    division by their sum, the result is
    ``-sum(p * log(p))`` over positive probabilities. When ``normalize`` is
    true, the entropy is divided by ``log(order)`` and lies in ``[0, 1]``.

    An embedding with numerical rank zero or one returns ``NaN`` under ABFE's
    entropy degeneracy policy. ``order`` must be at least 2, ``delay`` must be
    at least 1, and the signal must provide at least two embedding rows.
    """
    order = _validate_integer(order, name="order", minimum=2)
    delay = _validate_integer(delay, name="delay", minimum=1)
    minimum_length = (order - 1) * delay + 2
    data = validate_signal(signal, minimum_length=minimum_length)

    centered = data - np.mean(data)
    embedded = _embed(centered, order, delay)
    if np.linalg.matrix_rank(embedded) <= 1:
        return float("nan")

    singular_values = np.linalg.svd(embedded, compute_uv=False)
    probabilities = singular_values / np.sum(singular_values)
    positive = probabilities > 0.0
    result = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    if normalize:
        result /= math.log(order)
    return result


__all__ = ["svd_entropy"]
