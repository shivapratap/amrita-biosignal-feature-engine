"""Shared validation for pre-windowed biosignal arrays."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def validate_signal(signal: ArrayLike, *, minimum_length: int = 1) -> FloatArray:
    """Return an owned, read-only, finite 1-D ``float64`` signal.

    Parameters
    ----------
    signal:
        Real numeric one-dimensional signal. Boolean, string, complex, and
        object arrays are rejected rather than silently coerced. Two-dimensional
        row or column arrays are rejected rather than flattened because ABFE
        operates on explicitly pre-windowed 1-D arrays.
    minimum_length:
        Minimum number of samples required by the calling computation.

    Raises
    ------
    TypeError
        If the input is not a real numeric array or sequence, or if
        ``minimum_length`` is not an integer.
    ValueError
        If the input is not one-dimensional, is too short, or contains a NaN
        or infinity.
    """
    if isinstance(minimum_length, bool) or not isinstance(minimum_length, int):
        raise TypeError("minimum_length must be an integer")
    if minimum_length < 1:
        raise ValueError("minimum_length must be at least 1")

    try:
        raw = np.asarray(signal)
    except (TypeError, ValueError) as exc:
        raise TypeError("signal must be a homogeneous real numeric sequence") from exc
    if raw.ndim != 1:
        raise ValueError(f"signal must be one-dimensional; got shape {raw.shape}")
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
        raise TypeError("signal must contain real numeric values, not booleans or strings")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError("signal must contain real numeric values, not complex values")

    array = np.array(raw, dtype=np.float64, copy=True)
    if array.size < minimum_length:
        raise ValueError(
            f"signal must contain at least {minimum_length} sample(s); got {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("signal must contain only finite values")
    array.setflags(write=False)
    return array


def validate_nonnegative_threshold(threshold: float) -> float:
    """Return a finite, nonnegative threshold as ``float``."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError("threshold must be a real number") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return value
