from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from amrita_biosignal_feature_engine import time_domain as td

Feature = Callable[[ArrayLike], float]


@pytest.fixture
def signal() -> NDArray[np.float64]:
    return np.array([-2.0, -1.0, 0.0, 3.0, 4.0])


def test_classical_statistics_match_explicit_references(signal: NDArray[np.float64]) -> None:
    assert td.minimum(signal) == -2.0
    assert td.maximum(signal) == 4.0
    assert td.sum_value(signal) == 4.0
    assert td.mean(signal) == pytest.approx(np.mean(signal))
    assert td.median(signal) == pytest.approx(np.median(signal))
    assert td.standard_deviation(signal) == pytest.approx(np.std(signal, ddof=0))
    assert td.variance(signal) == pytest.approx(np.var(signal, ddof=0))
    assert td.skewness(signal) == pytest.approx(stats.skew(signal, bias=False))
    assert td.kurtosis(signal) == pytest.approx(
        stats.kurtosis(signal, fisher=True, bias=False)
    )


def test_amplitude_and_shape_features_match_hand_calculation() -> None:
    signal = np.array([-2.0, 1.0, -3.0, 2.0])
    assert td.mean_absolute_value(signal) == 2.0
    assert td.root_mean_square(signal) == pytest.approx(np.sqrt(4.5))
    assert td.peak_to_peak(signal) == 5.0
    assert td.integrated_absolute_value(signal) == 8.0
    assert td.waveform_length(signal) == 12.0


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        ([-1.0, 1.0], 1.0),
        ([-1.0, 0.0, 1.0], 1.0),
        ([-1.0, 0.0, 0.0, -2.0], 0.0),
        ([1.0, -1.0, 1.0, -1.0], 3.0),
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0], 0.0),
    ],
)
def test_zero_crossing_policy(signal: list[float], expected: float) -> None:
    assert td.zero_crossing_count(signal) == expected


def test_slope_sign_change_count() -> None:
    signal = [0.0, 2.0, 1.0, 4.0, 4.0, 3.0]
    assert td.slope_sign_change_count(signal) == 2.0
    assert td.slope_sign_change_count(signal, threshold=4.0) == 1.0


@pytest.mark.parametrize("signal", [[1.0], [1.0, 2.0], [1.0, 1.0, 1.0]])
def test_slope_sign_change_short_or_flat_signal_is_zero(signal: list[float]) -> None:
    assert td.slope_sign_change_count(signal) == 0.0


def test_constant_signal_policy() -> None:
    signal = np.full(8, 3.5)
    assert td.minimum(signal) == 3.5
    assert td.maximum(signal) == 3.5
    assert td.sum_value(signal) == 28.0
    assert td.mean(signal) == 3.5
    assert td.median(signal) == 3.5
    assert td.standard_deviation(signal) == 0.0
    assert td.variance(signal) == 0.0
    assert td.mean_absolute_value(signal) == 3.5
    assert td.root_mean_square(signal) == 3.5
    assert td.peak_to_peak(signal) == 0.0
    assert td.integrated_absolute_value(signal) == 28.0
    assert td.waveform_length(signal) == 0.0
    assert td.zero_crossing_count(signal) == 0.0
    assert td.slope_sign_change_count(signal) == 0.0
    assert np.isnan(td.skewness(signal))
    assert np.isnan(td.kurtosis(signal))


@pytest.mark.parametrize(
    "function",
    [td.skewness, td.kurtosis],
)
def test_undefined_short_higher_moments_return_nan(function: Feature) -> None:
    assert np.isnan(function([1.0, 2.0]))


def test_scaling_laws_are_explicit_and_full_precision() -> None:
    signal = np.array([-1.125456789, 0.25, 2.75, -0.875, 4.125])

    amplitude_linear: tuple[Feature, ...] = (
        td.minimum,
        td.maximum,
        td.sum_value,
        td.mean,
        td.median,
        td.standard_deviation,
        td.mean_absolute_value,
        td.root_mean_square,
        td.peak_to_peak,
        td.integrated_absolute_value,
        td.waveform_length,
    )
    scale_invariant: tuple[Feature, ...] = (
        td.skewness,
        td.kurtosis,
        td.zero_crossing_count,
        td.slope_sign_change_count,
    )

    for scale in (2.0, 5.0, 10.0):
        scaled = scale * signal
        for function in amplitude_linear:
            assert function(scaled) == pytest.approx(scale * function(signal), rel=1e-13)
        assert td.variance(scaled) == pytest.approx(scale**2 * td.variance(signal), rel=1e-13)
        for function in scale_invariant:
            assert function(scaled) == pytest.approx(function(signal), abs=1e-13)

    assert td.mean(signal) != round(td.mean(signal), 2)


def test_z_score_preserves_shape_features() -> None:
    signal = np.array([-1.1, 0.2, 2.7, -0.8, 4.2, 1.3])
    z_scored = (signal - np.mean(signal)) / np.std(signal)
    assert td.skewness(z_scored) == pytest.approx(td.skewness(signal), abs=1e-13)
    assert td.kurtosis(z_scored) == pytest.approx(td.kurtosis(signal), abs=1e-13)


@pytest.mark.parametrize("scale", [-2.0, -5.0, -10.0])
def test_negative_scaling_laws(scale: float) -> None:
    signal = np.array([-1.125, 0.25, 2.75, -0.875, 4.125])
    scaled = scale * signal
    magnitude = abs(scale)
    assert td.minimum(scaled) == pytest.approx(scale * td.maximum(signal))
    assert td.maximum(scaled) == pytest.approx(scale * td.minimum(signal))
    for function in (td.sum_value, td.mean, td.median):
        assert function(scaled) == pytest.approx(scale * function(signal))
    for function in (
        td.standard_deviation,
        td.mean_absolute_value,
        td.root_mean_square,
        td.peak_to_peak,
        td.integrated_absolute_value,
        td.waveform_length,
    ):
        assert function(scaled) == pytest.approx(magnitude * function(signal))
    assert td.variance(scaled) == pytest.approx(scale**2 * td.variance(signal))
    assert td.skewness(scaled) == pytest.approx(-td.skewness(signal))
    assert td.kurtosis(scaled) == pytest.approx(td.kurtosis(signal))
    assert td.zero_crossing_count(scaled) == td.zero_crossing_count(signal)
    assert td.slope_sign_change_count(scaled) == td.slope_sign_change_count(signal)


PUBLIC_FEATURES: tuple[Feature, ...] = (
    td.minimum,
    td.maximum,
    td.sum_value,
    td.mean,
    td.median,
    td.standard_deviation,
    td.variance,
    td.kurtosis,
    td.skewness,
    td.mean_absolute_value,
    td.root_mean_square,
    td.peak_to_peak,
    td.integrated_absolute_value,
    td.waveform_length,
    td.zero_crossing_count,
    td.slope_sign_change_count,
)


@pytest.mark.parametrize("function", PUBLIC_FEATURES)
def test_all_features_reject_empty_input(function: Feature) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        function([])


@pytest.mark.parametrize("function", PUBLIC_FEATURES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_features_reject_nonfinite_input(function: Feature, bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        function([1.0, bad, 2.0, 3.0])


@pytest.mark.parametrize("function", PUBLIC_FEATURES)
def test_all_features_reject_2d_input(function: Feature) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        function(np.array([[1.0, 2.0, 3.0, 4.0]]))


def test_no_duplicate_average_or_total_aliases() -> None:
    assert not hasattr(td, "average")
    assert not hasattr(td, "total")
    assert not hasattr(td, "summation")
