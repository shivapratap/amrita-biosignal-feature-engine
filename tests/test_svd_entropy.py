from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import svd_entropy

FloatArray = NDArray[np.float64]


def eigenvalue_oracle(
    signal: FloatArray, *, order: int, delay: int, normalize: bool = False
) -> float:
    signal = signal - np.mean(signal)
    row_count = signal.size - (order - 1) * delay
    embedded = np.empty((row_count, order), dtype=np.float64)
    for row in range(row_count):
        for column in range(order):
            embedded[row, column] = signal[row + column * delay]
    eigenvalues = np.linalg.eigvalsh(embedded.T @ embedded)
    gram_tolerance = (
        np.max(np.abs(eigenvalues)) * max(embedded.shape) * np.finfo(float).eps
    )
    singular_values = np.sqrt(np.where(eigenvalues > gram_tolerance, eigenvalues, 0.0))[::-1]
    tolerance = singular_values[0] * max(embedded.shape) * np.finfo(float).eps
    if np.count_nonzero(singular_values > tolerance) <= 1:
        return float("nan")
    probabilities = singular_values / np.sum(singular_values)
    probabilities = probabilities[probabilities > 0.0]
    result = -float(np.sum(probabilities * np.log(probabilities)))
    return result / math.log(order) if normalize else result


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
def test_matches_independent_eigendecomposition_oracle(
    name: str,
    order: int,
    delay: int,
    synthetic_signals: dict[str, FloatArray],
) -> None:
    signal = synthetic_signals[name]
    expected = eigenvalue_oracle(signal, order=order, delay=delay)
    assert svd_entropy(signal, order=order, delay=delay) == pytest.approx(
        expected, abs=2e-14
    )


def test_hand_computed_equal_singular_values() -> None:
    # This zero-mean order-2 embedding is diag(1, -1), with probabilities 1/2, 1/2.
    signal = [1.0, 0.0, -1.0]
    assert svd_entropy(signal, order=2) == pytest.approx(np.log(2.0), abs=1e-15)
    assert svd_entropy(signal, order=2, normalize=True) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("scale", [2.0, 5.0, 10.0, -2.0, -5.0, -10.0])
def test_nonzero_scaling_is_invariant(scale: float) -> None:
    signal = np.random.default_rng(7).normal(size=128)
    assert svd_entropy(scale * signal, order=4, delay=2) == pytest.approx(
        svd_entropy(signal, order=4, delay=2), abs=2e-15
    )


def test_offset_and_z_score_are_invariant() -> None:
    signal = np.random.default_rng(8).normal(size=128)
    z_scored = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    assert svd_entropy(signal + 10.0) == pytest.approx(svd_entropy(signal), abs=2e-15)
    assert svd_entropy(z_scored) == pytest.approx(svd_entropy(signal), abs=2e-15)


@pytest.mark.parametrize("signal", [np.zeros(20), np.ones(20)])
def test_rank_deficient_input_returns_nan(signal: FloatArray) -> None:
    assert np.isnan(svd_entropy(signal))


@pytest.mark.parametrize("order", [0, 1, -1])
def test_invalid_order_range_raises(order: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        svd_entropy(np.arange(10, dtype=float), order=order)


@pytest.mark.parametrize("order", [True, 2.5, "2"])
def test_noninteger_order_raises(order: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        svd_entropy(np.arange(10, dtype=float), order=order)  # type: ignore[arg-type]


@pytest.mark.parametrize("delay", [0, -1])
def test_invalid_delay_range_raises(delay: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        svd_entropy(np.arange(10, dtype=float), delay=delay)


@pytest.mark.parametrize("delay", [True, 1.5, "1"])
def test_noninteger_delay_raises(delay: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        svd_entropy(np.arange(10, dtype=float), delay=delay)  # type: ignore[arg-type]


def test_signal_too_short_for_two_embedding_rows_raises() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        svd_entropy(np.arange(7, dtype=float), order=4, delay=2)


def test_nonfinite_signal_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        svd_entropy([0.0, 1.0, np.nan, 2.0])
