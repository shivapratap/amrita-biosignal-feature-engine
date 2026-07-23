from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import approximate_entropy

FloatArray = NDArray[np.float64]


def brute_force_approximate_entropy(
    signal: FloatArray, *, order: int, tolerance: float
) -> float:
    signal = np.asarray(signal, dtype=np.float64)

    def phi(dimension: int) -> float:
        count = signal.size - dimension + 1
        templates = [signal[index : index + dimension] for index in range(count)]
        probabilities = []
        for left in templates:
            matches = sum(np.max(np.abs(left - right)) <= tolerance for right in templates)
            probabilities.append(matches / count)
        return float(np.mean(np.log(probabilities)))

    return phi(order) - phi(order + 1)


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
def test_matches_independent_brute_force_oracle(
    name: str, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    tolerance = 0.2 * float(np.std(signal, ddof=0))
    expected = brute_force_approximate_entropy(signal, order=2, tolerance=tolerance)
    assert approximate_entropy(signal, order=2) == pytest.approx(expected, abs=2e-15)


def test_hand_computed_reference_value() -> None:
    signal = np.array([0.0, 0.0, 1.0, 0.0])
    expected_phi_1 = (3.0 * np.log(3.0 / 4.0) + np.log(1.0 / 4.0)) / 4.0
    expected_phi_2 = np.log(1.0 / 3.0)
    expected = expected_phi_1 - expected_phi_2
    assert approximate_entropy(signal, order=1, tolerance=0.5) == pytest.approx(
        expected, abs=1e-15
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("scale", [2.0, 5.0, 10.0, -2.0, -5.0, -10.0])
def test_default_tolerance_is_scale_invariant(
    name: str, scale: float, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    assert approximate_entropy(scale * signal) == pytest.approx(
        approximate_entropy(signal), abs=2e-15
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_default_tolerance_is_z_score_invariant(
    name: str, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    z_scored = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    assert approximate_entropy(z_scored) == pytest.approx(
        approximate_entropy(signal), abs=2e-15
    )


def test_constant_signal_returns_nan() -> None:
    assert np.isnan(approximate_entropy(np.ones(20)))


@pytest.mark.parametrize("order", [0, -1])
def test_invalid_order_range_raises(order: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        approximate_entropy(np.arange(10, dtype=float), order=order)


@pytest.mark.parametrize("order", [True, 2.5, "2"])
def test_noninteger_order_raises(order: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        approximate_entropy(np.arange(10, dtype=float), order=order)  # type: ignore[arg-type]


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_tolerance_range_raises(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        approximate_entropy(np.arange(10, dtype=float), tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [True, "0.2", object()])
def test_nonreal_tolerance_raises(tolerance: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        approximate_entropy(
            np.arange(10, dtype=float), tolerance=tolerance  # type: ignore[arg-type]
        )


def test_signal_too_short_for_order_raises() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        approximate_entropy([0.0, 1.0, 2.0, 3.0], order=3)


def test_nonfinite_signal_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        approximate_entropy([0.0, 1.0, np.nan, 2.0])
