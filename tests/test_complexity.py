from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

from amrita_biosignal_feature_engine.complexity import (
    hjorth_complexity,
    hjorth_mobility,
    katz_fractal_dimension,
    petrosian_fractal_dimension,
)


def _manual_hjorth(signal: np.ndarray) -> tuple[float, float]:
    first = np.diff(signal)
    mobility = np.sqrt(np.var(first) / np.var(signal))
    derivative_mobility = np.sqrt(np.var(np.diff(first)) / np.var(first))
    return float(mobility), float(derivative_mobility / mobility)


def _manual_katz(signal: np.ndarray) -> float:
    steps = np.abs(np.diff(signal))
    path_length = np.sum(steps)
    mean_step = np.mean(steps)
    displacement = np.max(np.abs(signal - signal[0]))
    return float(np.log10(path_length / mean_step) / np.log10(displacement / mean_step))


def test_hjorth_features_match_independent_population_variance_oracle() -> None:
    signal = np.array([0.0, 1.0, -0.5, 2.0, 0.25, -1.0])
    expected_mobility, expected_complexity = _manual_hjorth(signal)
    assert hjorth_mobility(signal) == pytest.approx(expected_mobility)
    assert hjorth_complexity(signal) == pytest.approx(expected_complexity)


@pytest.mark.parametrize("scale", [0.5, 3.0, -2.0])
def test_hjorth_features_are_affine_invariant(scale: float) -> None:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 80, endpoint=False))
    transformed = scale * signal + 17.0
    assert hjorth_mobility(transformed) == pytest.approx(hjorth_mobility(signal))
    assert hjorth_complexity(transformed) == pytest.approx(hjorth_complexity(signal))


def test_hjorth_degeneracy_policy_distinguishes_zero_from_undefined() -> None:
    assert np.isnan(hjorth_mobility(np.ones(4)))
    assert np.isnan(hjorth_complexity(np.ones(4)))
    ramp = np.arange(5.0)
    assert hjorth_mobility(ramp) == 0.0
    assert np.isnan(hjorth_complexity(ramp))


def test_petrosian_counts_transitions_after_removing_zero_differences() -> None:
    signal = np.array([0.0, -1.0, -1.0, -2.0])
    assert petrosian_fractal_dimension(signal) == 1.0

    alternating = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    n = alternating.size
    changes = 3
    expected = np.log10(n) / (
        np.log10(n) + np.log10(n / (n + 0.4 * changes))
    )
    assert petrosian_fractal_dimension(alternating) == pytest.approx(expected)
    assert petrosian_fractal_dimension(np.ones(5)) == 1.0


@pytest.mark.parametrize("scale", [0.5, 3.0, -2.0])
def test_fractal_dimensions_are_affine_invariant(scale: float) -> None:
    signal = np.array([0.0, 1.0, -0.5, 2.0, 0.25, -1.0])
    transformed = scale * signal + 17.0
    assert petrosian_fractal_dimension(transformed) == pytest.approx(
        petrosian_fractal_dimension(signal)
    )
    assert katz_fractal_dimension(transformed) == pytest.approx(
        katz_fractal_dimension(signal)
    )


def test_katz_matches_independent_geometry_oracle() -> None:
    signal = np.array([0.0, 2.0, 1.0, 4.0, -1.0, 3.0])
    assert katz_fractal_dimension(signal) == pytest.approx(_manual_katz(signal))
    assert katz_fractal_dimension(np.arange(5.0)) == pytest.approx(1.0)
    assert np.isnan(katz_fractal_dimension(np.ones(5)))
    assert np.isnan(katz_fractal_dimension(np.array([0.0, 1.0])))


@pytest.mark.parametrize(
    "function,minimum_length",
    [
        (hjorth_mobility, 2),
        (hjorth_complexity, 3),
        (petrosian_fractal_dimension, 3),
        (katz_fractal_dimension, 2),
    ],
)
def test_complexity_features_enforce_minimum_length(
    function: Callable[[ArrayLike], float], minimum_length: int
) -> None:
    with pytest.raises(ValueError, match="at least"):
        function(np.arange(minimum_length - 1.0))
    result = function(np.arange(float(minimum_length)))
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "bad_signal",
    [
        [],
        [True, False, True],
        ["a", "b", "c"],
        [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j],
        np.array([[1.0, 2.0, 3.0]]),
        [0.0, np.nan, 1.0],
        [0.0, np.inf, 1.0],
    ],
)
def test_complexity_features_preserve_strict_signal_validation(
    bad_signal: object,
) -> None:
    for function in (
        hjorth_mobility,
        hjorth_complexity,
        petrosian_fractal_dimension,
        katz_fractal_dimension,
    ):
        with pytest.raises((TypeError, ValueError)):
            function(bad_signal)  # type: ignore[arg-type]
