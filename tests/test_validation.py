from __future__ import annotations

import numpy as np
import pytest

from amrita_biosignal_feature_engine.validation import (
    validate_nonnegative_threshold,
    validate_signal,
)


def test_validate_signal_returns_float64_without_flattening() -> None:
    result = validate_signal([1, 2, 3])
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_validate_signal_returns_owned_read_only_array() -> None:
    source = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = validate_signal(source)
    assert not np.shares_memory(result, source)
    assert not result.flags.writeable
    source[0] = 99.0
    assert result[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        result[0] = 5.0


@pytest.mark.parametrize(
    "signal",
    [
        np.array([[1.0, 2.0]]),
        np.array([[1.0], [2.0]]),
        np.zeros((2, 2)),
        3.0,
    ],
)
def test_validate_signal_rejects_non_1d_input(signal: object) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_signal(signal)  # type: ignore[arg-type]


def test_validate_signal_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        validate_signal([])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_validate_signal_rejects_nonfinite_values(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_signal([1.0, bad, 2.0])


def test_validate_signal_rejects_nonnumeric_values() -> None:
    with pytest.raises(TypeError, match="numeric"):
        validate_signal(["one", "two"])


@pytest.mark.parametrize(
    "signal",
    [
        ["1.0", "2.0", "3.0"],
        [True, False, True],
        np.array([1.0, 2.0], dtype=object),
        np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        [1, "2", 3.0],
    ],
)
def test_validate_signal_rejects_silently_coercible_nonreal_input(signal: object) -> None:
    with pytest.raises(TypeError, match="real numeric"):
        validate_signal(signal)  # type: ignore[arg-type]


def test_validate_signal_enforces_minimum_length() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        validate_signal([1.0, 2.0], minimum_length=3)


@pytest.mark.parametrize("minimum_length", [True, 2.5, "2"])
def test_validate_signal_rejects_noninteger_minimum_length(minimum_length: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        validate_signal([1.0], minimum_length=minimum_length)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [0, 0.5, np.float64(2.0)])
def test_validate_threshold_accepts_finite_nonnegative_values(threshold: float) -> None:
    assert validate_nonnegative_threshold(threshold) == float(threshold)


@pytest.mark.parametrize("threshold", [-1.0, np.nan, np.inf, -np.inf])
def test_validate_threshold_rejects_invalid_values(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_nonnegative_threshold(threshold)
