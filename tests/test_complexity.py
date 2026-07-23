from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import pytest
from numpy.typing import ArrayLike

from amrita_biosignal_feature_engine.complexity import (
    _resolve_dfa_scales,
    detrended_fluctuation_analysis,
    fisher_information,
    higuchi_fractal_dimension,
    hjorth_complexity,
    hjorth_mobility,
    katz_fractal_dimension,
    largest_lyapunov_exponent,
    lempel_ziv_complexity,
    petrosian_fractal_dimension,
)


class _LyapunovParameters(TypedDict):
    sampling_frequency: float
    embedding_dimension: int
    delay_samples: int
    minimum_separation_samples: int
    fit_start: int
    fit_end: int


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


def _literal_dfa_oracle(
    signal: np.ndarray, *, scales: tuple[int, ...], detrend_order: int
) -> float:
    integrated = np.cumsum(signal - np.mean(signal))
    fluctuations: list[float] = []
    for scale in scales:
        residuals: list[float] = []
        positions = np.arange(scale, dtype=np.float64)
        design = np.vander(positions, N=detrend_order + 1, increasing=True)
        segment_count = signal.size // scale
        for segment_index in range(segment_count):
            start = segment_index * scale
            segment = integrated[start : start + scale]
            coefficients, _, _, _ = np.linalg.lstsq(design, segment, rcond=None)
            fitted = design @ coefficients
            residuals.extend(float(value) for value in segment - fitted)
        fluctuations.append(float(np.sqrt(np.mean(np.square(residuals)))))
    predictor = np.log(np.asarray(scales, dtype=np.float64))
    response = np.log(np.asarray(fluctuations))
    design = np.column_stack((np.ones(predictor.size), predictor))
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    return float(coefficients[1])


def test_dfa_automatic_scale_generation_is_frozen() -> None:
    assert _resolve_dfa_scales(
        50,
        scales=None,
        minimum_scale=4,
        maximum_scale_fraction=0.1,
        scale_ratio=1.2,
        detrend_order=1,
    ) == (4, 5)
    assert _resolve_dfa_scales(
        128,
        scales=None,
        minimum_scale=4,
        maximum_scale_fraction=0.1,
        scale_ratio=1.2,
        detrend_order=1,
    ) == (4, 5, 6, 8, 9, 11)


@pytest.mark.parametrize(
    "scales,detrend_order",
    [((4, 8, 16), 0), ((4, 8, 16), 1), ((5, 10, 20), 2)],
)
def test_dfa_matches_independent_lstsq_oracle(
    scales: tuple[int, ...], detrend_order: int
) -> None:
    signal = np.random.default_rng(8642 + detrend_order).normal(size=120)
    assert detrended_fluctuation_analysis(
        signal, scales=scales, detrend_order=detrend_order
    ) == pytest.approx(
        _literal_dfa_oracle(
            signal, scales=scales, detrend_order=detrend_order
        ),
        abs=2e-14,
    )


def test_dfa_signal_classes_and_degeneracy() -> None:
    noise = np.random.default_rng(24601).normal(size=1000)
    random_walk = np.cumsum(noise)
    noise_exponent = detrended_fluctuation_analysis(noise)
    walk_exponent = detrended_fluctuation_analysis(random_walk)
    assert 0.3 < noise_exponent < 0.8
    assert 1.2 < walk_exponent < 1.8
    assert np.isfinite(
        detrended_fluctuation_analysis(
            np.sin(np.linspace(0.0, 20.0 * np.pi, 1000))
        )
    )
    assert np.isnan(detrended_fluctuation_analysis(np.ones(100)))


@pytest.mark.parametrize("scale,offset", [(0.5, 0.0), (3.0, 10.0), (-2.0, -4.0)])
def test_dfa_is_affine_invariant(scale: float, offset: float) -> None:
    signal = np.random.default_rng(13579).normal(size=500)
    assert detrended_fluctuation_analysis(scale * signal + offset) == pytest.approx(
        detrended_fluctuation_analysis(signal), abs=2e-14
    )


@pytest.mark.parametrize(
    "scales",
    [
        [4, 8],
        (4,),
        (4, 4),
        (8, 4),
        (True, 8),
        (4.0, 8),
        (2, 4),
        (4, 51),
    ],
)
def test_dfa_rejects_invalid_explicit_scales(scales: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        detrended_fluctuation_analysis(
            np.arange(100.0), scales=scales  # type: ignore[arg-type]
        )


def test_dfa_rejects_invalid_generator_and_detrending_parameters() -> None:
    with pytest.raises(ValueError, match="minimum_scale"):
        detrended_fluctuation_analysis(np.arange(100.0), minimum_scale=2)
    with pytest.raises(TypeError, match="detrend_order"):
        detrended_fluctuation_analysis(np.arange(100.0), detrend_order=True)
    with pytest.raises(ValueError, match="detrend_order"):
        detrended_fluctuation_analysis(np.arange(100.0), detrend_order=-1)
    for value in (0.0, 0.6, np.inf):
        with pytest.raises(ValueError, match="maximum_scale_fraction"):
            detrended_fluctuation_analysis(
                np.arange(100.0), maximum_scale_fraction=value
            )
    with pytest.raises(ValueError, match="scale_ratio"):
        detrended_fluctuation_analysis(np.arange(100.0), scale_ratio=1.0)
    with pytest.raises(ValueError, match="at least 50"):
        detrended_fluctuation_analysis(np.arange(49.0))
    assert isinstance(detrended_fluctuation_analysis(np.arange(50.0)), float)


def _literal_lyapunov_oracle(
    signal: np.ndarray,
    *,
    sampling_frequency: float,
    embedding_dimension: int,
    delay_samples: int,
    minimum_separation_samples: int,
    fit_start: int,
    fit_end: int,
) -> float:
    centered = signal - np.mean(signal)
    vector_count = signal.size - (embedding_dimension - 1) * delay_samples
    embedded = np.asarray(
        [
            [
                centered[index + coordinate * delay_samples]
                for coordinate in range(embedding_dimension)
            ]
            for index in range(vector_count)
        ]
    )
    distance_matrix = np.sqrt(
        np.sum((embedded[:, None, :] - embedded[None, :, :]) ** 2, axis=2)
    )
    pairs: list[tuple[int, int]] = []
    for reference in range(vector_count):
        candidates = [
            index
            for index in range(vector_count)
            if abs(index - reference) > minimum_separation_samples
            and distance_matrix[reference, index] > 0.0
        ]
        if candidates:
            neighbor = min(
                candidates, key=lambda index: distance_matrix[reference, index]
            )
            pairs.append((reference, neighbor))
    times: list[float] = []
    divergence: list[float] = []
    for step in range(fit_start, fit_end):
        logs = [
            np.log(distance_matrix[left + step, right + step])
            for left, right in pairs
            if left + step < vector_count
            and right + step < vector_count
            and distance_matrix[left + step, right + step] > 0.0
        ]
        if len(logs) >= 2:
            times.append(step / sampling_frequency)
            divergence.append(float(np.mean(logs)))
    design = np.column_stack((np.ones(len(times)), times))
    coefficients, _, _, _ = np.linalg.lstsq(
        design, np.asarray(divergence), rcond=None
    )
    return float(coefficients[1])


def _logistic_signal(length: int, *, parameter: float = 4.0) -> np.ndarray:
    values = np.empty(length + 500)
    values[0] = 0.123456789
    for index in range(values.size - 1):
        values[index + 1] = parameter * values[index] * (1.0 - values[index])
    return values[500:]


def _lorenz_x_signal(length: int, *, step_seconds: float = 0.01) -> np.ndarray:
    states = np.empty((length + 1000, 3))
    states[0] = (1.0, 1.0, 1.0)

    def derivative(state: np.ndarray) -> np.ndarray:
        x_value, y_value, z_value = state
        return np.array(
            (
                10.0 * (y_value - x_value),
                x_value * (28.0 - z_value) - y_value,
                x_value * y_value - (8.0 / 3.0) * z_value,
            )
        )

    for index in range(states.shape[0] - 1):
        state = states[index]
        first = derivative(state)
        second = derivative(state + step_seconds * first / 2.0)
        third = derivative(state + step_seconds * second / 2.0)
        fourth = derivative(state + step_seconds * third)
        states[index + 1] = state + step_seconds * (
            first + 2.0 * second + 2.0 * third + fourth
        ) / 6.0
    return states[1000:, 0]


def test_largest_lyapunov_matches_independent_distance_matrix_oracle() -> None:
    signal = _logistic_signal(300)
    parameters: _LyapunovParameters = {
        "sampling_frequency": 1.0,
        "embedding_dimension": 3,
        "delay_samples": 1,
        "minimum_separation_samples": 10,
        "fit_start": 0,
        "fit_end": 6,
    }
    assert largest_lyapunov_exponent(signal, **parameters) == pytest.approx(
        _literal_lyapunov_oracle(signal, **parameters), abs=2e-14
    )


def test_largest_lyapunov_skips_duplicate_embeddings_before_neighbor_selection() -> None:
    signal = np.tile(np.array([0.0, 1.0, 3.0, 2.0, -1.0, 0.5]), 8)
    parameters: _LyapunovParameters = {
        "sampling_frequency": 1.0,
        "embedding_dimension": 2,
        "delay_samples": 1,
        "minimum_separation_samples": 1,
        "fit_start": 0,
        "fit_end": 4,
    }
    expected = _literal_lyapunov_oracle(signal, **parameters)
    assert expected == pytest.approx(0.23969930756318522, abs=2e-15)
    assert largest_lyapunov_exponent(signal, **parameters) == pytest.approx(
        expected, abs=2e-14
    )


def test_largest_lyapunov_synthetic_behavior_and_invariances() -> None:
    chaotic = _logistic_signal(1000)
    parameters: _LyapunovParameters = {
        "sampling_frequency": 1.0,
        "embedding_dimension": 3,
        "delay_samples": 1,
        "minimum_separation_samples": 20,
        "fit_start": 0,
        "fit_end": 6,
    }
    estimate = largest_lyapunov_exponent(chaotic, **parameters)
    assert 0.4 < estimate < 0.9
    assert largest_lyapunov_exponent(3.0 * chaotic + 10.0, **parameters) == pytest.approx(
        estimate, abs=2e-13
    )
    assert largest_lyapunov_exponent(-2.0 * chaotic, **parameters) == pytest.approx(
        estimate, abs=2e-13
    )
    periodic = np.sin(np.linspace(0.0, 80.0 * np.pi, 1000, endpoint=False))
    assert np.isfinite(largest_lyapunov_exponent(periodic, **parameters))
    time = np.arange(1000)
    quasiperiodic = np.sin(0.11 * time) + np.sin(np.sqrt(2.0) * 0.11 * time)
    assert np.isfinite(largest_lyapunov_exponent(quasiperiodic, **parameters))


def test_largest_lyapunov_scales_with_sampling_frequency_in_inverse_seconds() -> None:
    signal = _logistic_signal(1000)
    parameters: _LyapunovParameters = {
        "sampling_frequency": 1.0,
        "embedding_dimension": 3,
        "delay_samples": 1,
        "minimum_separation_samples": 20,
        "fit_start": 0,
        "fit_end": 6,
    }
    estimate_per_second = largest_lyapunov_exponent(signal, **parameters)
    doubled_sampling_frequency: _LyapunovParameters = {
        **parameters,
        "sampling_frequency": 2.0,
    }
    assert largest_lyapunov_exponent(
        signal, **doubled_sampling_frequency
    ) == pytest.approx(2.0 * estimate_per_second, abs=2e-13)


def test_largest_lyapunov_is_positive_for_seeded_lorenz_x_trajectory() -> None:
    signal = _lorenz_x_signal(3000)
    estimate = largest_lyapunov_exponent(
        signal,
        sampling_frequency=100.0,
        embedding_dimension=6,
        delay_samples=10,
        minimum_separation_samples=100,
        fit_start=0,
        fit_end=10,
    )
    assert 0.3 < estimate < 2.0


def test_largest_lyapunov_validation_minimum_length_and_degeneracy() -> None:
    parameters: _LyapunovParameters = {
        "sampling_frequency": 100.0,
        "embedding_dimension": 3,
        "delay_samples": 2,
        "minimum_separation_samples": 4,
        "fit_start": 1,
        "fit_end": 6,
    }
    minimum_length = 20
    with pytest.raises(ValueError, match="at least 20"):
        largest_lyapunov_exponent(np.arange(minimum_length - 1.0), **parameters)
    assert isinstance(
        largest_lyapunov_exponent(np.arange(float(minimum_length)), **parameters),
        float,
    )
    assert np.isnan(largest_lyapunov_exponent(np.ones(minimum_length), **parameters))
    with pytest.raises(ValueError, match="fit_end"):
        largest_lyapunov_exponent(
            np.arange(100.0),
            sampling_frequency=100.0,
            embedding_dimension=3,
            delay_samples=2,
            minimum_separation_samples=4,
            fit_start=1,
            fit_end=3,
        )
    with pytest.raises(ValueError, match="sampling_frequency"):
        largest_lyapunov_exponent(
            np.arange(100.0), sampling_frequency=0.0,
            embedding_dimension=3, delay_samples=2,
            minimum_separation_samples=4, fit_start=1, fit_end=6,
        )
    with pytest.raises(ValueError, match="embedding_dimension"):
        largest_lyapunov_exponent(
            np.arange(100.0), sampling_frequency=100.0,
            embedding_dimension=1, delay_samples=2,
            minimum_separation_samples=4, fit_start=1, fit_end=6,
        )
    with pytest.raises(ValueError, match="delay_samples"):
        largest_lyapunov_exponent(
            np.arange(100.0), sampling_frequency=100.0,
            embedding_dimension=3, delay_samples=0,
            minimum_separation_samples=4, fit_start=1, fit_end=6,
        )
    with pytest.raises(ValueError, match="minimum_separation_samples"):
        largest_lyapunov_exponent(
            np.arange(100.0), sampling_frequency=100.0,
            embedding_dimension=3, delay_samples=2,
            minimum_separation_samples=-1, fit_start=1, fit_end=6,
        )
    with pytest.raises(ValueError, match="fit_start"):
        largest_lyapunov_exponent(
            np.arange(100.0), sampling_frequency=100.0,
            embedding_dimension=3, delay_samples=2,
            minimum_separation_samples=4, fit_start=-1, fit_end=6,
        )
    with pytest.raises(TypeError, match="embedding_dimension"):
        largest_lyapunov_exponent(
            np.arange(100.0), sampling_frequency=100.0,
            embedding_dimension=True, delay_samples=2,
            minimum_separation_samples=4, fit_start=1, fit_end=6,
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
        detrended_fluctuation_analysis,
        petrosian_fractal_dimension,
        katz_fractal_dimension,
        lempel_ziv_complexity,
    ):
        with pytest.raises((TypeError, ValueError)):
            function(bad_signal)  # type: ignore[arg-type]
