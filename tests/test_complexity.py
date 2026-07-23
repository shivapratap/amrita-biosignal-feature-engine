from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

from amrita_biosignal_feature_engine.complexity import (
    fisher_information,
    higuchi_fractal_dimension,
    hjorth_complexity,
    hjorth_mobility,
    katz_fractal_dimension,
    lempel_ziv_complexity,
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


def test_lempel_ziv_matches_published_hand_parsed_sequence() -> None:
    # LZ76 phrases: 1 / 0 / 01 / 1110 / 1100 / 0010
    binary = np.array([int(symbol) for symbol in "1001111011000010"])
    signal = np.where(binary == 1, 1.0, -1.0)
    assert np.median(signal) == 0.0
    assert lempel_ziv_complexity(signal, normalize=False) == 6.0
    assert lempel_ziv_complexity(signal) == pytest.approx(1.5)


def test_lempel_ziv_median_ties_map_to_one() -> None:
    signal = np.array([-2.0, 0.0, 0.0, 3.0])
    binary = signal >= np.median(signal)
    np.testing.assert_array_equal(binary, [False, True, True, True])
    assert lempel_ziv_complexity(signal, normalize=False) == 3.0


def test_lempel_ziv_validation_degeneracy_and_invariance() -> None:
    with pytest.raises(TypeError, match="normalize"):
        lempel_ziv_complexity([0.0, 1.0], normalize=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="normalize"):
        lempel_ziv_complexity([0.0, 1.0], normalize=np.bool_(True))  # type: ignore[arg-type]
    assert np.isnan(lempel_ziv_complexity(np.ones(8)))

    signal = np.array([-4.0, -1.0, 0.5, 3.0, -2.0, 2.0])
    expected = lempel_ziv_complexity(signal)
    assert lempel_ziv_complexity(4.0 * signal + 11.0) == expected
    assert lempel_ziv_complexity(-signal) == expected


def test_hjorth_features_match_independent_population_variance_oracle() -> None:
    signal = np.array([0.0, 1.0, -0.5, 2.0, 0.25, -1.0])
    expected_mobility, expected_complexity = _manual_hjorth(signal)
    assert hjorth_mobility(signal) == pytest.approx(expected_mobility)
    assert hjorth_complexity(signal) == pytest.approx(expected_complexity)


def _eigenvalue_fisher_oracle(
    signal: np.ndarray, *, order: int, delay: int
) -> float:
    centered = signal - np.mean(signal)
    rows = centered.size - (order - 1) * delay
    embedded = np.column_stack(
        [
            centered[offset * delay : offset * delay + rows]
            for offset in range(order)
        ]
    )
    eigenvalues = np.linalg.eigvalsh(embedded.T @ embedded)
    singular_values = np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
    probabilities = singular_values / np.sum(singular_values)
    return float(
        sum(
            (probabilities[index + 1] - probabilities[index]) ** 2
            / probabilities[index]
            for index in range(order - 1)
            if probabilities[index] > 0.0
        )
    )


def test_fisher_information_matches_hand_calculated_two_value_spectrum() -> None:
    signal = np.array([1.0, -1.0, -1.0, 1.0])
    first = 2.0 / (2.0 + np.sqrt(2.0))
    second = np.sqrt(2.0) / (2.0 + np.sqrt(2.0))
    expected = (second - first) ** 2 / first
    assert fisher_information(signal) == pytest.approx(expected)


@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (3, 2), (4, 2)])
def test_fisher_matches_independent_gram_eigenvalue_oracle(
    order: int, delay: int
) -> None:
    signal = np.random.default_rng(order * 100 + delay).normal(size=80)
    assert fisher_information(signal, order=order, delay=delay) == pytest.approx(
        _eigenvalue_fisher_oracle(signal, order=order, delay=delay),
        abs=2e-15,
    )


@pytest.mark.parametrize("scale,offset", [(0.5, 0.0), (3.0, 10.0), (-2.0, -4.0)])
def test_fisher_is_affine_invariant(scale: float, offset: float) -> None:
    signal = np.random.default_rng(2468).normal(size=80)
    assert fisher_information(scale * signal + offset, order=3, delay=2) == pytest.approx(
        fisher_information(signal, order=3, delay=2), abs=2e-15
    )


def test_fisher_validation_and_rank_degeneracy() -> None:
    for name, arguments in (
        ("order", {"order": 1}),
        ("delay", {"delay": 0}),
    ):
        with pytest.raises(ValueError, match=name):
            fisher_information(np.arange(10.0), **arguments)
    with pytest.raises(TypeError, match="integer"):
        fisher_information(np.arange(10.0), order=True)
    with pytest.raises(TypeError, match="integer"):
        fisher_information(np.arange(10.0), delay=1.5)  # type: ignore[arg-type]
    assert np.isnan(fisher_information(np.ones(8)))
    assert np.isnan(fisher_information(np.tile([-1.0, 1.0], 8)))


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


def _literal_higuchi_oracle(signal: np.ndarray, *, k_max: int) -> float:
    length = signal.size
    scale_lengths: list[float] = []
    for scale in range(1, k_max + 1):
        offset_lengths: list[float] = []
        for offset in range(scale):
            intervals = (length - offset - 1) // scale
            total = 0.0
            for interval in range(1, intervals + 1):
                right = offset + interval * scale
                left = offset + (interval - 1) * scale
                total += abs(signal[right] - signal[left])
            offset_lengths.append(
                total * (length - 1) / (scale**2 * intervals)
            )
        scale_lengths.append(sum(offset_lengths) / scale)
    predictor = [np.log(1.0 / scale) for scale in range(1, k_max + 1)]
    response = [np.log(value) for value in scale_lengths]
    predictor_mean = sum(predictor) / len(predictor)
    response_mean = sum(response) / len(response)
    numerator = sum(
        (x_value - predictor_mean) * (y_value - response_mean)
        for x_value, y_value in zip(predictor, response, strict=True)
    )
    denominator = sum((value - predictor_mean) ** 2 for value in predictor)
    return float(numerator / denominator)


@pytest.mark.parametrize("k_max", [2, 3, 5, 10])
def test_higuchi_matches_independent_literal_loop_oracle(k_max: int) -> None:
    signal = np.random.default_rng(1000 + k_max).normal(size=80)
    assert higuchi_fractal_dimension(signal, k_max=k_max) == pytest.approx(
        _literal_higuchi_oracle(signal, k_max=k_max), abs=2e-15
    )


def test_higuchi_known_ramp_degeneracy_and_validation() -> None:
    assert higuchi_fractal_dimension(np.arange(80.0)) == pytest.approx(1.0)
    assert np.isnan(higuchi_fractal_dimension(np.ones(80)))
    with pytest.raises(TypeError, match="k_max"):
        higuchi_fractal_dimension(np.arange(80.0), k_max=True)
    with pytest.raises(TypeError, match="k_max"):
        higuchi_fractal_dimension(np.arange(80.0), k_max=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_max"):
        higuchi_fractal_dimension(np.arange(80.0), k_max=1)
    with pytest.raises(ValueError, match="at least 21"):
        higuchi_fractal_dimension(np.arange(20.0))
    assert isinstance(higuchi_fractal_dimension(np.arange(21.0)), float)


@pytest.mark.parametrize("scale,offset", [(0.5, 0.0), (3.0, 10.0), (-2.0, -4.0)])
def test_higuchi_is_affine_invariant(scale: float, offset: float) -> None:
    signal = np.random.default_rng(9753).normal(size=100)
    assert higuchi_fractal_dimension(scale * signal + offset, k_max=8) == pytest.approx(
        higuchi_fractal_dimension(signal, k_max=8), abs=2e-15
    )


@pytest.mark.parametrize(
    "function,minimum_length",
    [
        (lempel_ziv_complexity, 2),
        (hjorth_mobility, 2),
        (hjorth_complexity, 3),
        (fisher_information, 3),
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
        fisher_information,
        higuchi_fractal_dimension,
        petrosian_fractal_dimension,
        katz_fractal_dimension,
        lempel_ziv_complexity,
    ):
        with pytest.raises((TypeError, ValueError)):
            function(bad_signal)  # type: ignore[arg-type]
