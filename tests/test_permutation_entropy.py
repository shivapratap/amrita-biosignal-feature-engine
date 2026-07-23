from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import permutation_entropy

FloatArray = NDArray[np.float64]


def brute_force_permutation_entropy(
    signal: FloatArray, *, order: int, delay: int, normalize: bool = False
) -> float:
    patterns: dict[tuple[int, ...], int] = {}
    pattern_count = signal.size - (order - 1) * delay
    for start in range(pattern_count):
        template = signal[start : start + order * delay : delay]
        pattern = tuple(np.argsort(template, kind="stable"))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    if len(patterns) < 2:
        return float("nan")
    probabilities = np.array(list(patterns.values()), dtype=float) / pattern_count
    result = -float(np.sum(probabilities * np.log(probabilities)))
    return result / math.lgamma(order + 1) if normalize else result


@pytest.fixture(scope="module")
def synthetic_signals() -> dict[str, FloatArray]:
    periodic_time = np.linspace(0.0, 6.0 * np.pi, 128, endpoint=False)
    chirp_time = np.linspace(0.0, 1.0, 128, endpoint=False)
    return {
        "periodic": np.sin(periodic_time),
        "white_noise": np.random.default_rng(321).normal(size=128),
        "chirp": np.sin(2.0 * np.pi * (2.0 * chirp_time + 10.0 * chirp_time**2)),
    }


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (4, 2)])
def test_matches_independent_oracle(
    name: str,
    order: int,
    delay: int,
    synthetic_signals: dict[str, FloatArray],
) -> None:
    signal = synthetic_signals[name]
    expected = brute_force_permutation_entropy(signal, order=order, delay=delay)
    assert permutation_entropy(signal, order=order, delay=delay) == pytest.approx(
        expected, abs=2e-15
    )


def test_hand_computed_ordinal_pattern_probabilities() -> None:
    # Patterns are (0, 1), (0, 1), and (1, 0), giving probabilities 2/3 and 1/3.
    signal = [1.0, 2.0, 3.0, 1.0]
    expected = -(2.0 / 3.0) * np.log(2.0 / 3.0) - (1.0 / 3.0) * np.log(1.0 / 3.0)
    assert permutation_entropy(signal, order=2) == pytest.approx(expected, abs=1e-15)


def test_normalization_uses_log_factorial_order() -> None:
    signal = np.random.default_rng(42).normal(size=200)
    raw = permutation_entropy(signal, order=4)
    assert permutation_entropy(signal, order=4, normalize=True) == pytest.approx(
        raw / np.log(math.factorial(4)), abs=2e-15
    )


@pytest.mark.parametrize("scale", [2.0, 5.0, 10.0])
def test_positive_scaling_preserves_patterns(scale: float) -> None:
    signal = np.random.default_rng(2).normal(size=128)
    assert permutation_entropy(scale * signal) == permutation_entropy(signal)


@pytest.mark.parametrize("scale", [-2.0, -5.0, -10.0])
def test_negative_scaling_preserves_pattern_probabilities(scale: float) -> None:
    signal = np.random.default_rng(2).normal(size=128)
    assert permutation_entropy(scale * signal) == pytest.approx(
        permutation_entropy(signal), abs=2e-15
    )


def test_z_score_preserves_patterns() -> None:
    signal = np.random.default_rng(7).normal(size=128)
    z_scored = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    assert permutation_entropy(z_scored) == permutation_entropy(signal)


def test_stable_tie_rule_matches_hand_patterns() -> None:
    signal = np.array([1.0, 1.0, 0.0, 0.0])
    expected = brute_force_permutation_entropy(signal, order=2, delay=1)
    assert permutation_entropy(signal, order=2) == expected


@pytest.mark.parametrize("signal", [np.ones(20), np.arange(20, dtype=float)])
def test_single_pattern_input_returns_nan(signal: FloatArray) -> None:
    assert np.isnan(permutation_entropy(signal))


@pytest.mark.parametrize("order", [0, 1, -1])
def test_invalid_order_range_raises(order: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        permutation_entropy(np.arange(10, dtype=float), order=order)


@pytest.mark.parametrize("order", [True, 2.5, "2"])
def test_noninteger_order_raises(order: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        permutation_entropy(np.arange(10, dtype=float), order=order)  # type: ignore[arg-type]


@pytest.mark.parametrize("delay", [0, -1])
def test_invalid_delay_range_raises(delay: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        permutation_entropy(np.arange(10, dtype=float), delay=delay)


@pytest.mark.parametrize("delay", [True, 1.5, "1"])
def test_noninteger_delay_raises(delay: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        permutation_entropy(np.arange(10, dtype=float), delay=delay)  # type: ignore[arg-type]


def test_signal_too_short_for_two_patterns_raises() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        permutation_entropy(np.arange(7, dtype=float), order=4, delay=2)


def test_nonfinite_signal_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        permutation_entropy([0.0, 1.0, np.nan, 2.0])
