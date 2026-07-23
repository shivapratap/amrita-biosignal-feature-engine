from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.entropy import fuzzy_entropy

FloatArray = NDArray[np.float64]


def paper_definition_oracle(
    signal: FloatArray,
    *,
    order: int,
    delay: int,
    tolerance_factor: float,
    exponent: float,
) -> float:
    tolerance = tolerance_factor * float(np.std(signal, ddof=0))
    template_count = signal.size - order * delay

    def phi(dimension: int) -> float:
        templates: list[FloatArray] = []
        for start in range(template_count):
            template = signal[start : start + dimension * delay : delay].copy()
            template -= np.mean(template)
            templates.append(template)
        similarities = []
        for left_index, left in enumerate(templates):
            for right in templates[left_index + 1 :]:
                distance = float(np.max(np.abs(left - right)))
                similarities.append(
                    math.exp(-math.log(2.0) * (distance / tolerance) ** exponent)
                )
        return float(np.mean(similarities))

    phi_order = phi(order)
    phi_next = phi(order + 1)
    if phi_order == 0.0 or phi_next == 0.0:
        return float("nan")
    return math.log(phi_order) - math.log(phi_next)


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
@pytest.mark.parametrize("order,delay", [(1, 1), (2, 1), (3, 2)])
def test_matches_independent_paper_definition_oracle(
    name: str,
    order: int,
    delay: int,
    synthetic_signals: dict[str, FloatArray],
) -> None:
    signal = synthetic_signals[name]
    expected = paper_definition_oracle(
        signal,
        order=order,
        delay=delay,
        tolerance_factor=0.2,
        exponent=2.0,
    )
    assert fuzzy_entropy(signal, order=order, delay=delay) == pytest.approx(
        expected, abs=2e-14
    )


def test_hand_computed_two_pair_similarity_ratio() -> None:
    signal = np.array([0.0, 1.0, 0.0, 1.0])
    tolerance_factor = 2.0
    expected = math.log(2.0) * ((4.0 / 3.0) ** 2 - 1.0)
    assert fuzzy_entropy(
        signal, order=2, tolerance_factor=tolerance_factor
    ) == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("scale", [2.0, 5.0, 10.0, -2.0, -5.0, -10.0])
def test_nonzero_scaling_is_invariant(
    name: str, scale: float, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    assert fuzzy_entropy(scale * signal) == pytest.approx(
        fuzzy_entropy(signal), abs=2e-14
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_offset_and_z_score_are_invariant(
    name: str, synthetic_signals: dict[str, FloatArray]
) -> None:
    signal = synthetic_signals[name]
    z_scored = (signal - np.mean(signal)) / np.std(signal, ddof=0)
    expected = fuzzy_entropy(signal)
    assert fuzzy_entropy(signal + 10.0) == pytest.approx(expected, abs=2e-14)
    assert fuzzy_entropy(z_scored) == pytest.approx(expected, abs=2e-14)


def test_perfectly_regular_nonconstant_signal_returns_real_zero() -> None:
    assert fuzzy_entropy(np.arange(20, dtype=float)) == pytest.approx(0.0, abs=1e-15)


def test_constant_signal_returns_nan() -> None:
    assert np.isnan(fuzzy_entropy(np.ones(20)))


def test_complete_similarity_underflow_returns_nan() -> None:
    signal = np.array([0.0, 1.0, 0.0, 2.0])
    assert np.isnan(fuzzy_entropy(signal, order=2, tolerance_factor=1e-200))


@pytest.mark.parametrize("order", [0, -1])
def test_invalid_order_range_raises(order: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        fuzzy_entropy(np.arange(10, dtype=float), order=order)


@pytest.mark.parametrize("order", [True, 2.5, "2"])
def test_noninteger_order_raises(order: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        fuzzy_entropy(np.arange(10, dtype=float), order=order)  # type: ignore[arg-type]


@pytest.mark.parametrize("delay", [0, -1])
def test_invalid_delay_range_raises(delay: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        fuzzy_entropy(np.arange(10, dtype=float), delay=delay)


@pytest.mark.parametrize("delay", [True, 1.5, "1"])
def test_noninteger_delay_raises(delay: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        fuzzy_entropy(np.arange(10, dtype=float), delay=delay)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["tolerance_factor", "exponent"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_positive_real_range_raises(name: str, value: float) -> None:
    kwargs = {name: value}
    with pytest.raises(ValueError, match="finite and positive"):
        fuzzy_entropy(np.arange(10, dtype=float), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["tolerance_factor", "exponent"])
@pytest.mark.parametrize("value", [True, "2", object()])
def test_nonreal_parameter_raises(name: str, value: object) -> None:
    kwargs = {name: value}
    with pytest.raises(TypeError, match="real number"):
        fuzzy_entropy(np.arange(10, dtype=float), **kwargs)  # type: ignore[arg-type]


def test_signal_too_short_for_two_common_templates_raises() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        fuzzy_entropy(np.arange(7, dtype=float), order=3, delay=2)


def test_nonfinite_signal_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        fuzzy_entropy([0.0, 1.0, np.nan, 2.0])
