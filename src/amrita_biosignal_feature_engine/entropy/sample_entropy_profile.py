"""Authoritative-compatible sample-entropy profiling.

This module implements the Cumulative Histogram Method used by
``sampen-profile``. Compatibility intentionally includes NumPy half-to-even
rounding of Chebyshev distances to three decimal places and the ``1e-12``
profile regularizer/filter. The fixed distance quantization is a documented
compatibility exception to ABFE's general prohibition on scale-coupled
rounding: removing it would change the reference algorithm's tolerance axis
and numerical output.

Only the profile itself is exposed. ABFE deliberately does not implement the
legacy ``TotalSampEn`` or ``AvgSampEn`` profile-sum construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

from ..validation import FloatArray, validate_signal

_PROFILE_EPSILON = 1e-12
_DISTANCE_DECIMALS = 3


@dataclass(frozen=True, slots=True)
class SampleEntropyProfile:
    """Valid tolerance values and their corresponding sample entropy."""

    r_values: FloatArray
    se_profile: FloatArray
    order: int
    distance_decimals: int = _DISTANCE_DECIMALS

    @property
    def point_count(self) -> int:
        """Number of valid tolerance/profile pairs."""
        return int(self.se_profile.size)


def _validate_order(order: int) -> int:
    if isinstance(order, bool | np.bool_) or not isinstance(order, int | np.integer):
        raise TypeError("order must be an integer")
    result = int(order)
    if result < 1:
        raise ValueError("order must be at least 1")
    return result


def _templates(signal: FloatArray, dimension: int, template_count: int) -> FloatArray:
    """Return ``template_count`` sliding templates of the given dimension."""
    return np.column_stack(
        [signal[index : template_count + index] for index in range(dimension)]
    )


def _distance_matrix(templates: FloatArray) -> FloatArray:
    """Return quantized Chebyshev distances with self-pairs excluded."""
    distances = cdist(templates, templates, metric="chebyshev")
    np.round(distances, _DISTANCE_DECIMALS, out=distances)
    np.fill_diagonal(distances, np.inf)
    return np.asarray(distances, dtype=np.float64)


def _mean_cdf(distances: FloatArray, tolerances: FloatArray) -> FloatArray:
    """Average per-template empirical cumulative distance distributions."""
    count = distances.shape[0]
    cumulative = np.zeros(tolerances.size, dtype=np.float64)
    for row in distances:
        sorted_row = np.sort(row)
        matches = np.searchsorted(sorted_row, tolerances, side="right")
        cumulative += matches / (count - 1)
    return np.asarray(cumulative / count, dtype=np.float64)


def _chm(signal: FloatArray, order: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    template_count = signal.size - order
    templates_m = _templates(signal, order, template_count)
    templates_m1 = _templates(signal, order + 1, template_count)
    distances_m = _distance_matrix(templates_m)
    distances_m1 = _distance_matrix(templates_m1)
    tolerances = np.union1d(
        np.unique(distances_m[np.isfinite(distances_m)]),
        np.unique(distances_m1[np.isfinite(distances_m1)]),
    )
    b = _mean_cdf(distances_m, tolerances)
    a = _mean_cdf(distances_m1, tolerances)
    return b, a, np.asarray(tolerances, dtype=np.float64)


def sample_entropy_profile(signal: ArrayLike, *, order: int = 2) -> SampleEntropyProfile:
    """Compute the authoritative CHM sample-entropy profile.

    Parameters
    ----------
    signal:
        Finite, numeric, one-dimensional pre-windowed signal.
    order:
        Embedding dimension. At least ``order + 2`` samples are required.

    Returns
    -------
    SampleEntropyProfile
        Read-only arrays containing only points where both cumulative match
        probabilities exceed ``1e-12``.

    Notes
    -----
    The tolerance axis is amplitude-dependent. Profile entropy values are
    theoretically scale-invariant, but the reference algorithm's fixed
    three-decimal distance quantization can cause small scale-dependent
    differences. This behavior is retained solely for exact compatibility.
    """
    order = _validate_order(order)
    data = validate_signal(signal, minimum_length=order + 2)
    b, a, tolerances = _chm(data, order)
    profile = np.log((b + _PROFILE_EPSILON) / (a + _PROFILE_EPSILON))
    valid = np.isfinite(profile) & (b > _PROFILE_EPSILON) & (a > _PROFILE_EPSILON)
    r_values = np.asarray(tolerances[valid], dtype=np.float64)
    se_profile = np.asarray(profile[valid], dtype=np.float64)
    if se_profile.size == 0:
        raise ValueError("no valid sample entropy profile points were produced")
    r_values.setflags(write=False)
    se_profile.setflags(write=False)
    return SampleEntropyProfile(r_values=r_values, se_profile=se_profile, order=order)


__all__ = ["SampleEntropyProfile", "sample_entropy_profile"]
