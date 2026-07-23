"""Fractal-dimension and complexity features for one-dimensional signals."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import validate_signal


def _validate_integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _lz76_phrase_count(sequence: NDArray[np.uint8]) -> int:
    """Return the LZ76 exhaustive-history phrase count for a binary array."""
    complexity = 1
    prefix_length = 1
    substring_length = 1
    maximum_substring_length = 1
    pointer = 0

    while prefix_length + substring_length <= sequence.size:
        if (
            sequence[pointer + substring_length - 1]
            == sequence[prefix_length + substring_length - 1]
        ):
            substring_length += 1
        else:
            maximum_substring_length = max(
                substring_length, maximum_substring_length
            )
            pointer += 1
            if pointer == prefix_length:
                complexity += 1
                prefix_length += maximum_substring_length
                pointer = 0
                maximum_substring_length = 1
            substring_length = 1

    if substring_length != 1:
        complexity += 1
    return complexity


def lempel_ziv_complexity(signal: ArrayLike, *, normalize: bool = True) -> float:
    """Return LZ76 complexity after a deterministic median binary split.

    Samples greater than or equal to the signal median map to one; all others
    map to zero. The raw exhaustive-history phrase count is returned when
    ``normalize`` is false. Otherwise the count is multiplied by
    ``log2(n) / n``. Constant signals return ``NaN``.
    """
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    data = validate_signal(signal, minimum_length=2)
    if float(np.ptp(data)) == 0.0:
        return float("nan")
    binary = np.asarray(data >= np.median(data), dtype=np.uint8)
    phrase_count = _lz76_phrase_count(binary)
    if not normalize:
        return float(phrase_count)
    return float(phrase_count * np.log2(data.size) / data.size)


def hjorth_mobility(signal: ArrayLike) -> float:
    """Return Hjorth mobility in samples⁻¹.

    Mobility is ``sqrt(var(diff(x)) / var(x))`` using population variances.
    A constant signal has undefined mobility and returns ``NaN``.
    """
    data = validate_signal(signal, minimum_length=2)
    signal_variance = float(np.var(data, ddof=0))
    if signal_variance == 0.0:
        return float("nan")
    derivative_variance = float(np.var(np.diff(data), ddof=0))
    return float(np.sqrt(derivative_variance / signal_variance))


def hjorth_complexity(signal: ArrayLike) -> float:
    """Return dimensionless Hjorth complexity.

    Complexity is the mobility of the first difference divided by the
    mobility of the signal. A constant signal or constant first difference
    has undefined complexity and returns ``NaN``.
    """
    data = validate_signal(signal, minimum_length=3)
    first_difference = np.diff(data)
    signal_variance = float(np.var(data, ddof=0))
    derivative_variance = float(np.var(first_difference, ddof=0))
    if signal_variance == 0.0 or derivative_variance == 0.0:
        return float("nan")
    second_derivative_variance = float(np.var(np.diff(first_difference), ddof=0))
    mobility = np.sqrt(derivative_variance / signal_variance)
    derivative_mobility = np.sqrt(second_derivative_variance / derivative_variance)
    return float(derivative_mobility / mobility)


def fisher_information(
    signal: ArrayLike, *, order: int = 2, delay: int = 1
) -> float:
    """Return dimensionless SVD-spectrum Fisher information.

    The signal is mean-centered and delay embedded. If ``p`` is the descending
    normalized singular-value spectrum, the result is
    ``sum((p[i + 1] - p[i])**2 / p[i])`` over positive denominators.
    Numerically rank-zero and rank-one embeddings return ``NaN``.
    """
    order = _validate_integer(order, name="order", minimum=2)
    delay = _validate_integer(delay, name="delay", minimum=1)
    minimum_length = (order - 1) * delay + 2
    data = validate_signal(signal, minimum_length=minimum_length)
    centered = data - np.mean(data)
    row_count = centered.size - (order - 1) * delay
    embedded = np.column_stack(
        [
            centered[offset * delay : offset * delay + row_count]
            for offset in range(order)
        ]
    )
    singular_values = np.linalg.svd(embedded, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return float("nan")
    tolerance = (
        np.finfo(np.float64).eps
        * max(embedded.shape)
        * float(singular_values[0])
    )
    if int(np.count_nonzero(singular_values > tolerance)) <= 1:
        return float("nan")
    probabilities = singular_values / np.sum(singular_values)
    denominators = probabilities[:-1]
    differences = np.diff(probabilities)
    positive = denominators > 0.0
    return float(np.sum((differences[positive] ** 2) / denominators[positive]))


def petrosian_fractal_dimension(signal: ArrayLike) -> float:
    """Return Petrosian fractal dimension.

    Zero first differences are removed before derivative sign transitions are
    counted, so a flat run does not create artificial transitions.
    """
    data = validate_signal(signal, minimum_length=3)
    differences = np.diff(data)
    nonzero_differences = differences[differences != 0.0]
    if nonzero_differences.size < 2:
        sign_changes = 0
    else:
        sign_changes = int(
            np.count_nonzero(
                np.signbit(nonzero_differences[1:])
                != np.signbit(nonzero_differences[:-1])
            )
        )
    sample_count = int(data.size)
    log_length = np.log10(sample_count)
    denominator = log_length + np.log10(
        sample_count / (sample_count + 0.4 * sign_changes)
    )
    return float(log_length / denominator)


def katz_fractal_dimension(signal: ArrayLike) -> float:
    """Return Katz fractal dimension.

    Constant signals and geometries with an indeterminate logarithmic ratio
    return ``NaN``.
    """
    data = validate_signal(signal, minimum_length=2)
    distances = np.abs(np.diff(data))
    path_length = float(np.sum(distances))
    mean_step = float(np.mean(distances))
    maximum_displacement = float(np.max(np.abs(data - data[0])))
    if path_length == 0.0 or mean_step == 0.0 or maximum_displacement == 0.0:
        return float("nan")
    numerator = float(np.log10(path_length / mean_step))
    denominator = float(np.log10(maximum_displacement / mean_step))
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


def higuchi_fractal_dimension(signal: ArrayLike, *, k_max: int = 10) -> float:
    """Return Higuchi fractal dimension using scales one through ``k_max``.

    Each scale averages the normalized lengths of all offset subsampled
    curves. The result is the ordinary least-squares slope of ``log(L(k))``
    against ``log(1 / k)``. Constant or indeterminate inputs return ``NaN``.
    """
    k_max = _validate_integer(k_max, name="k_max", minimum=2)
    data = validate_signal(signal, minimum_length=2 * k_max + 1)
    sample_count = int(data.size)
    scales: list[float] = []
    mean_lengths: list[float] = []
    for scale in range(1, k_max + 1):
        offset_lengths: list[float] = []
        for offset in range(scale):
            interval_count = (sample_count - offset - 1) // scale
            if interval_count < 1:
                continue
            curve = data[offset : offset + (interval_count + 1) * scale : scale]
            distance = float(np.sum(np.abs(np.diff(curve))))
            normalized_length = (
                distance
                * (sample_count - 1)
                / (scale * scale * interval_count)
            )
            offset_lengths.append(normalized_length)
        if len(offset_lengths) != scale:
            continue
        mean_length = float(np.mean(offset_lengths))
        if np.isfinite(mean_length) and mean_length > 0.0:
            scales.append(float(scale))
            mean_lengths.append(mean_length)
    if len(mean_lengths) < 2:
        return float("nan")
    predictor = np.log(1.0 / np.asarray(scales))
    response = np.log(np.asarray(mean_lengths))
    centered_predictor = predictor - np.mean(predictor)
    denominator = float(np.sum(centered_predictor**2))
    if denominator == 0.0:
        return float("nan")
    slope = float(
        np.sum(centered_predictor * (response - np.mean(response))) / denominator
    )
    return slope if np.isfinite(slope) else float("nan")


def _validate_real(
    value: float, *, name: str, minimum_exclusive: float, maximum: float | None = None
) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if (
        not np.isfinite(result)
        or result <= minimum_exclusive
        or (maximum is not None and result > maximum)
    ):
        interval = (
            f"({minimum_exclusive}, {maximum}]"
            if maximum is not None
            else f"> {minimum_exclusive}"
        )
        raise ValueError(f"{name} must be finite and in {interval}")
    return result


def _resolve_dfa_scales(
    signal_length: int,
    *,
    scales: tuple[int, ...] | None,
    minimum_scale: int,
    maximum_scale_fraction: float,
    scale_ratio: float,
    detrend_order: int,
) -> tuple[int, ...]:
    if scales is not None:
        if not isinstance(scales, tuple):
            raise TypeError("scales must be a tuple of integers or None")
        if len(scales) < 2:
            raise ValueError("scales must contain at least two values")
        if any(
            isinstance(scale, bool | np.bool_)
            or not isinstance(scale, int | np.integer)
            for scale in scales
        ):
            raise TypeError("scales must contain integers")
        resolved = tuple(int(scale) for scale in scales)
        if any(
            right <= left
            for left, right in zip(resolved, resolved[1:], strict=False)
        ):
            raise ValueError("scales must be strictly increasing and unique")
    else:
        maximum_scale = int(np.floor(maximum_scale_fraction * signal_length))
        generated: list[int] = []
        exponent = 0
        while True:
            raw_scale = minimum_scale * scale_ratio**exponent
            candidate = int(np.floor(raw_scale))
            if candidate > maximum_scale:
                break
            if not generated or candidate > generated[-1]:
                generated.append(candidate)
            target = candidate + 1
            estimated = int(
                np.ceil(
                    np.log(target / minimum_scale) / np.log(scale_ratio)
                )
            )
            exponent = max(exponent + 1, estimated)
        resolved = tuple(generated)
        if len(resolved) < 2:
            raise ValueError("automatic scale generation produced fewer than two scales")
    if any(scale <= detrend_order + 1 for scale in resolved):
        raise ValueError("every scale must exceed detrend_order + 1")
    if any(signal_length // scale < 2 for scale in resolved):
        raise ValueError("every scale must provide at least two complete segments")
    return resolved


def detrended_fluctuation_analysis(
    signal: ArrayLike,
    *,
    scales: tuple[int, ...] | None = None,
    minimum_scale: int = 4,
    maximum_scale_fraction: float = 0.1,
    scale_ratio: float = 1.2,
    detrend_order: int = 1,
) -> float:
    """Return the detrended-fluctuation scaling exponent.

    The demeaned signal is cumulatively integrated, divided into
    non-overlapping segments at each scale, polynomially detrended, and
    reduced to one RMS fluctuation per scale. The result is the OLS slope of
    log fluctuation against log scale.
    """
    detrend_order = _validate_integer(
        detrend_order, name="detrend_order", minimum=0
    )
    minimum_scale = _validate_integer(
        minimum_scale, name="minimum_scale", minimum=1
    )
    if minimum_scale <= detrend_order + 1:
        raise ValueError("minimum_scale must exceed detrend_order + 1")
    maximum_scale_fraction = _validate_real(
        maximum_scale_fraction,
        name="maximum_scale_fraction",
        minimum_exclusive=0.0,
        maximum=0.5,
    )
    scale_ratio = _validate_real(
        scale_ratio, name="scale_ratio", minimum_exclusive=1.0
    )

    if scales is None:
        data = validate_signal(signal, minimum_length=50)
    else:
        if not isinstance(scales, tuple):
            raise TypeError("scales must be a tuple of integers or None")
        if not scales:
            raise ValueError("scales must contain at least two values")
        valid_integer_scales = all(
            not isinstance(scale, bool | np.bool_)
            and isinstance(scale, int | np.integer)
            for scale in scales
        )
        if not valid_integer_scales:
            raise TypeError("scales must contain integers")
        data = validate_signal(signal, minimum_length=2 * max(int(s) for s in scales))

    resolved_scales = _resolve_dfa_scales(
        int(data.size),
        scales=scales,
        minimum_scale=minimum_scale,
        maximum_scale_fraction=maximum_scale_fraction,
        scale_ratio=scale_ratio,
        detrend_order=detrend_order,
    )
    integrated = np.cumsum(data - np.mean(data))
    valid_scales: list[float] = []
    fluctuations: list[float] = []
    for scale in resolved_scales:
        segment_count = integrated.size // scale
        segments = integrated[: segment_count * scale].reshape(segment_count, scale)
        positions = np.arange(scale, dtype=np.float64)
        residual_sum_squares = 0.0
        for segment in segments:
            coefficients = np.polyfit(positions, segment, detrend_order)
            residuals = segment - np.polyval(coefficients, positions)
            residual_sum_squares += float(np.sum(residuals**2))
        fluctuation = float(
            np.sqrt(residual_sum_squares / (segment_count * scale))
        )
        if np.isfinite(fluctuation) and fluctuation > 0.0:
            valid_scales.append(float(scale))
            fluctuations.append(fluctuation)
    if len(fluctuations) < 2:
        return float("nan")
    predictor = np.log(np.asarray(valid_scales))
    response = np.log(np.asarray(fluctuations))
    centered_predictor = predictor - np.mean(predictor)
    denominator = float(np.sum(centered_predictor**2))
    if denominator == 0.0:
        return float("nan")
    slope = float(
        np.sum(centered_predictor * (response - np.mean(response))) / denominator
    )
    return slope if np.isfinite(slope) else float("nan")


def largest_lyapunov_exponent(
    signal: ArrayLike,
    *,
    sampling_frequency: float,
    embedding_dimension: int,
    delay_samples: int,
    minimum_separation_samples: int,
    fit_start: int,
    fit_end: int,
) -> float:
    """Estimate the largest Lyapunov exponent with Rosenstein's method.

    All reconstruction, temporal-exclusion, and half-open fit-interval
    parameters are explicit. The fitted slope uses time in seconds and is
    returned in inverse seconds.
    """
    sampling_frequency = _validate_real(
        sampling_frequency,
        name="sampling_frequency",
        minimum_exclusive=0.0,
    )
    embedding_dimension = _validate_integer(
        embedding_dimension, name="embedding_dimension", minimum=2
    )
    delay_samples = _validate_integer(
        delay_samples, name="delay_samples", minimum=1
    )
    minimum_separation_samples = _validate_integer(
        minimum_separation_samples,
        name="minimum_separation_samples",
        minimum=0,
    )
    fit_start = _validate_integer(fit_start, name="fit_start", minimum=0)
    fit_end = _validate_integer(fit_end, name="fit_end", minimum=0)
    if fit_end < fit_start + 3:
        raise ValueError("fit_end must be at least fit_start + 3")
    minimum_length = (
        (embedding_dimension - 1) * delay_samples
        + fit_end
        + 2 * minimum_separation_samples
        + 2
    )
    data = validate_signal(signal, minimum_length=minimum_length)
    centered = data - np.mean(data)
    vector_count = centered.size - (embedding_dimension - 1) * delay_samples
    embedded = np.column_stack(
        [
            centered[
                coordinate * delay_samples : coordinate * delay_samples
                + vector_count
            ]
            for coordinate in range(embedding_dimension)
        ]
    )

    neighbor_pairs: list[tuple[int, int]] = []
    indices = np.arange(vector_count)
    for reference_index in range(vector_count):
        admissible = (
            np.abs(indices - reference_index) > minimum_separation_samples
        )
        if not np.any(admissible):
            continue
        distances = np.linalg.norm(
            embedded - embedded[reference_index], axis=1
        )
        distances[~admissible] = np.inf
        neighbor_index = int(np.argmin(distances))
        if np.isfinite(distances[neighbor_index]) and distances[neighbor_index] > 0.0:
            neighbor_pairs.append((reference_index, neighbor_index))

    valid_steps: list[float] = []
    mean_log_distances: list[float] = []
    for step in range(fit_start, fit_end):
        log_distances: list[float] = []
        for reference_index, neighbor_index in neighbor_pairs:
            if (
                reference_index + step >= vector_count
                or neighbor_index + step >= vector_count
            ):
                continue
            separation = float(
                np.linalg.norm(
                    embedded[reference_index + step]
                    - embedded[neighbor_index + step]
                )
            )
            if np.isfinite(separation) and separation > 0.0:
                log_distances.append(float(np.log(separation)))
        if len(log_distances) >= 2:
            valid_steps.append(step / sampling_frequency)
            mean_log_distances.append(float(np.mean(log_distances)))
    if len(valid_steps) < 3:
        return float("nan")
    predictor = np.asarray(valid_steps)
    response = np.asarray(mean_log_distances)
    centered_predictor = predictor - np.mean(predictor)
    denominator = float(np.sum(centered_predictor**2))
    if denominator == 0.0:
        return float("nan")
    slope = float(
        np.sum(centered_predictor * (response - np.mean(response))) / denominator
    )
    return slope if np.isfinite(slope) else float("nan")


__all__ = [
    "detrended_fluctuation_analysis",
    "fisher_information",
    "higuchi_fractal_dimension",
    "hjorth_complexity",
    "hjorth_mobility",
    "katz_fractal_dimension",
    "lempel_ziv_complexity",
    "largest_lyapunov_exponent",
    "petrosian_fractal_dimension",
]
