from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import distribution_entropy

FloatArray = NDArray[np.float64]


def manual_distribution_entropy(
    signal: FloatArray,
    *,
    order: int,
    delay: int,
    n_bins: int,
    normalize: bool,
) -> float:
    template_count = signal.size - (order - 1) * delay
    templates = [
        signal[start : start + order * delay : delay] for start in range(template_count)
    ]
    distances = []
    for left_index, left in enumerate(templates):
        for right in templates[left_index + 1 :]:
            distances.append(float(np.max(np.abs(left - right))))
    minimum = min(distances)
    maximum = max(distances)
    if minimum == maximum:
        return float("nan")
    width = (maximum - minimum) / n_bins
    counts = np.zeros(n_bins, dtype=np.int64)
    for distance in distances:
        index = min(int((distance - minimum) / width), n_bins - 1)
        counts[index] += 1
    probabilities = counts[counts > 0] / len(distances)
    result = -float(np.sum(probabilities * np.log(probabilities)))
    return result / math.log(n_bins) if normalize else result


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
@pytest.mark.parametrize("order,delay,n_bins", [(1, 1, 8), (2, 1, 32), (3, 2, 64)])
def test_matches_independent_manual_histogram_oracle(
    name: str,
    order: int,
    delay: int,
    n_bins: int,
    synthetic_signals: dict[str, FloatArray],
) -> None:
    signal = synthetic_signals[name]
    expected = manual_distribution_entropy(
        signal, order=order, delay=delay, n_bins=n_bins, normalize=True
    )
    assert distribution_entropy(
        signal, order=order, delay=delay, n_bins=n_bins
    ) == pytest.approx(expected, abs=2e-15)


def test_hand_computed_two_bin_distribution() -> None:
    signal = np.array([0.0, 1.0, 3.0, 2.0])
    expected_raw = -(2.0 / 3.0) * np.log(2.0 / 3.0) - (1.0 / 3.0) * np.log(1.0 / 3.0)
    assert distribution_entropy(signal, order=2, n_bins=2, normalize=False) == pytest.approx(
        expected_raw, abs=1e-15
    )
    assert distribution_entropy(signal, order=2, n_bins=2) == pytest.approx(
        expected_raw / np.log(2.0), abs=1e-15
    )


def test_maximum_distance_is_included_in_final_bin() -> None:
    signal = np.array([0.0, 1.0, 3.0, 2.0])
    assert distribution_entropy(signal, order=2, n_bins=2) == pytest.approx(
        0.9182958340544894, abs=1e-15
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("scale", [2.0, 5.0, 10.0, -2.0, -5.0, -10.0])
def test_nonzero_scaling_is_invariant(
    name: str, scale: float, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    assert distribution_entropy(scale * signal, n_bins=64) == pytest.approx(
        distribution_entropy(signal, n_bins=64), abs=2e-15
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_offset_and_z_score_are_invariant(
    name: str, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    z_scored = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    expected = distribution_entropy(signal, n_bins=64)
    assert distribution_entropy(signal + 10.0, n_bins=64) == pytest.approx(
        expected, abs=2e-15
    )
    assert distribution_entropy(z_scored, n_bins=64) == pytest.approx(
        expected, abs=2e-15
    )


@pytest.mark.parametrize("signal", [np.ones(20), np.arange(3, dtype=float)])
def test_degenerate_distance_distribution_returns_nan(signal: FloatArray) -> None:
    assert np.isnan(distribution_entropy(signal))


@pytest.mark.parametrize("order", [0, -1])
def test_invalid_order_range_raises(order: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        distribution_entropy(np.arange(10, dtype=float), order=order)


@pytest.mark.parametrize("order", [True, 2.5, "2"])
def test_noninteger_order_raises(order: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        distribution_entropy(np.arange(10, dtype=float), order=order)  # type: ignore[arg-type]


@pytest.mark.parametrize("delay", [0, -1])
def test_invalid_delay_range_raises(delay: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        distribution_entropy(np.arange(10, dtype=float), delay=delay)


@pytest.mark.parametrize("delay", [True, 1.5, "1"])
def test_noninteger_delay_raises(delay: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        distribution_entropy(np.arange(10, dtype=float), delay=delay)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_bins", [0, 1, -1])
def test_invalid_bin_range_raises(n_bins: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        distribution_entropy(np.arange(10, dtype=float), n_bins=n_bins)


@pytest.mark.parametrize("n_bins", [True, 2.5, "2"])
def test_noninteger_bins_raise(n_bins: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        distribution_entropy(np.arange(10, dtype=float), n_bins=n_bins)  # type: ignore[arg-type]


def test_signal_too_short_for_two_embeddings_raises() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        distribution_entropy(np.arange(7, dtype=float), order=4, delay=2)


def test_nonfinite_signal_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        distribution_entropy([0.0, 1.0, np.nan, 2.0])
