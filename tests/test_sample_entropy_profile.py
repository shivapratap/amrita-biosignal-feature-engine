from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import sample_entropy_profile
from amrita_biosignal_feature_engine.entropy.sample_entropy_profile import _chm

FloatArray = NDArray[np.float64]


def brute_force_profile(signal: FloatArray, order: int) -> tuple[FloatArray, FloatArray]:
    """Independent unordered-pair oracle for the authoritative CHM definition."""
    signal = np.asarray(signal, dtype=float)
    template_count = signal.size - order
    templates_m = [signal[index : index + order] for index in range(template_count)]
    templates_m1 = [signal[index : index + order + 1] for index in range(template_count)]

    def distances(templates: list[FloatArray]) -> FloatArray:
        return np.array(
            [
                np.round(np.max(np.abs(templates[left] - templates[right])), 3)
                for left, right in combinations(range(template_count), 2)
            ]
        )

    distances_m = distances(templates_m)
    distances_m1 = distances(templates_m1)
    tolerances = np.unique(np.concatenate((distances_m, distances_m1)))
    pair_count = template_count * (template_count - 1) / 2
    b = np.array([(distances_m <= value).sum() / pair_count for value in tolerances])
    a = np.array([(distances_m1 <= value).sum() / pair_count for value in tolerances])
    profile = np.log((b + 1e-12) / (a + 1e-12))
    valid = np.isfinite(profile) & (b > 1e-12) & (a > 1e-12)
    return tolerances[valid], profile[valid]


@pytest.fixture(scope="module")
def synthetic_signals() -> dict[str, FloatArray]:
    periodic_time = np.linspace(0.0, 4.0 * np.pi, 96, endpoint=False)
    chirp_time = np.linspace(0.0, 1.0, 96, endpoint=False)
    return {
        "periodic": np.sin(periodic_time),
        "white_noise": np.random.default_rng(123).normal(size=96),
        "chirp": np.sin(2.0 * np.pi * (2.0 * chirp_time + 8.0 * chirp_time**2)),
    }


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_matches_independent_oracle_for_required_signals(
    name: str, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    expected_r, expected_profile = brute_force_profile(signal, 2)
    actual = sample_entropy_profile(signal, order=2)
    np.testing.assert_array_equal(actual.r_values, expected_r)
    # The unordered-pair oracle normalizes in a different floating-point
    # operation order than the authoritative per-row CDF. Two ULPs is the
    # measured worst case across these signals.
    np.testing.assert_allclose(actual.se_profile, expected_profile, rtol=0, atol=5e-15)


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("scale", [2.0, 5.0, 10.0])
def test_scaled_signals_preserve_authoritative_compatibility(
    name: str, scale: float, synthetic_signals: dict[str, FloatArray]
) -> None:
    """Quantized profiles need not be invariant, but must still match the authority."""
    signal = scale * synthetic_signals[name]
    expected_r, expected_profile = brute_force_profile(signal, 2)
    actual = sample_entropy_profile(signal, order=2)
    np.testing.assert_array_equal(actual.r_values, expected_r)
    np.testing.assert_allclose(actual.se_profile, expected_profile, rtol=0, atol=5e-15)


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_z_scored_signals_preserve_authoritative_compatibility(
    name: str, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    z_scored = (signal - np.mean(signal)) / np.std(signal)
    expected_r, expected_profile = brute_force_profile(z_scored, 2)
    actual = sample_entropy_profile(z_scored, order=2)
    np.testing.assert_array_equal(actual.r_values, expected_r)
    np.testing.assert_allclose(actual.se_profile, expected_profile, rtol=0, atol=5e-15)


def test_constant_signal_hand_calculation() -> None:
    result = sample_entropy_profile([0.0, 0.0, 0.0, 0.0], order=2)
    np.testing.assert_array_equal(result.r_values, [0.0])
    np.testing.assert_array_equal(result.se_profile, [0.0])


def test_known_profile_filters_zero_match_tolerance() -> None:
    result = sample_entropy_profile([0.0, 0.0, 1.0, 0.0], order=1)
    np.testing.assert_array_equal(result.r_values, [1.0])
    np.testing.assert_allclose(result.se_profile, [0.0], atol=1e-12)


def test_half_even_distance_quantization_is_preserved() -> None:
    signal = np.array([0.0, 0.0005, 0.0015, 0.0])
    _, _, tolerances = _chm(signal, 1)
    np.testing.assert_array_equal(tolerances, [0.0, 0.001, 0.002])


def test_profile_exposes_no_legacy_sum_statistics() -> None:
    result = sample_entropy_profile([0.0, 0.2, 0.1, 0.4, 0.3], order=2)
    assert not hasattr(result, "TotalSampEn")
    assert not hasattr(result, "AvgSampEn")
    assert result.point_count == result.se_profile.size


def test_profile_result_and_arrays_are_immutable() -> None:
    result = sample_entropy_profile([0.0, 0.2, 0.1, 0.4, 0.3], order=2)
    with pytest.raises(FrozenInstanceError):
        result.order = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        result.se_profile[0] = 10.0


@pytest.mark.parametrize("order", [0, -1])
def test_invalid_order_range_raises(order: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        sample_entropy_profile([0.0, 1.0, 2.0, 3.0], order=order)


@pytest.mark.parametrize("order", [True, 2.5, "2"])
def test_noninteger_order_raises(order: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        sample_entropy_profile([0.0, 1.0, 2.0, 3.0], order=order)  # type: ignore[arg-type]


def test_signal_too_short_for_order_raises() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        sample_entropy_profile([0.0, 1.0, 2.0, 3.0], order=3)


def test_multidimensional_signal_is_rejected_not_flattened() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        sample_entropy_profile(np.array([[0.0, 1.0, 2.0, 3.0]]), order=2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_signal_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        sample_entropy_profile([0.0, bad, 1.0, 2.0], order=2)
