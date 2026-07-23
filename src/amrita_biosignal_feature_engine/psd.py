"""Explicit power-spectral-density configuration and computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.signal import detrend as scipy_detrend
from scipy.signal import welch
from scipy.signal.windows import dpss

from .validation import FloatArray, validate_signal

DetrendMode: TypeAlias = Literal["constant", "linear", "none"]
ScalingMode: TypeAlias = Literal["density", "bin_power"]
MultitaperWeighting: TypeAlias = Literal["uniform", "eigen"]


@dataclass(frozen=True, slots=True)
class WelchPSDConfig:
    """Complete, immutable Welch PSD configuration.

    ``window_length`` and ``overlap`` are expressed in seconds. Each is
    multiplied by the sampling frequency and converted to samples with
    Python's round-half-to-even rule. ``scaling='density'`` produces power/Hz;
    ``'bin_power'`` multiplies density by FFT-bin spacing to produce power per
    bin. ``bin_power`` is deliberately not named ``spectrum`` because it does
    not use SciPy's window-dependent ``scaling='spectrum'`` normalization.
    """

    window_length: float = 2.0
    overlap: float = 1.0
    detrend: DetrendMode = "constant"
    scaling: ScalingMode = "density"
    window: str = "hann"


@dataclass(frozen=True, slots=True)
class MultitaperPSDConfig:
    """Complete, immutable DPSS multitaper PSD configuration.

    ``window_length`` and ``overlap`` are expressed in seconds. Each is
    multiplied by the sampling frequency and converted to samples with
    Python's round-half-to-even rule. ``bandwidth`` is the full effective
    bandwidth in Hz, giving a DPSS time-half-bandwidth product of
    ``window_length * bandwidth / 2``. ``scaling='density'`` produces
    power/Hz; ``'bin_power'`` produces power per FFT bin.
    """

    window_length: float = 2.0
    overlap: float = 1.0
    detrend: DetrendMode = "constant"
    scaling: ScalingMode = "density"
    bandwidth: float = 4.0
    low_bias: bool = True
    multitaper_weighting: MultitaperWeighting = "eigen"


PSDConfig: TypeAlias = WelchPSDConfig | MultitaperPSDConfig


@dataclass(frozen=True, slots=True)
class PSDResult:
    """A validated, immutable one-sided PSD with complete provenance."""

    frequencies: FloatArray
    values: FloatArray
    bin_spacing: float
    effective_bandwidth: float | None
    sampling_frequency: float
    signal_length: int
    samples_per_window: int
    samples_overlap: int
    segment_count: int
    config: PSDConfig

    def __post_init__(self) -> None:
        frequencies = np.array(self.frequencies, dtype=np.float64, copy=True)
        values = np.array(self.values, dtype=np.float64, copy=True)
        if frequencies.ndim != 1 or values.ndim != 1 or frequencies.shape != values.shape:
            raise ValueError("PSD frequencies and values must be aligned one-dimensional arrays")
        if frequencies.size < 2:
            raise ValueError("PSD must contain at least two frequency bins")
        if not np.all(np.isfinite(frequencies)) or not np.all(np.diff(frequencies) > 0):
            raise ValueError("PSD frequencies must be finite and strictly increasing")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("PSD values must be finite and nonnegative")

        sampling_frequency = _validate_real(
            self.sampling_frequency, "sampling_frequency", positive=True
        )
        bin_spacing = _validate_real(self.bin_spacing, "bin_spacing", positive=True)
        if not np.allclose(np.diff(frequencies), bin_spacing, rtol=1e-12, atol=1e-15):
            raise ValueError("bin_spacing must match the uniformly spaced frequency axis")
        if frequencies[0] < 0 or frequencies[-1] > sampling_frequency / 2.0 + bin_spacing * 1e-12:
            raise ValueError("PSD frequencies must lie between zero and Nyquist")
        if self.effective_bandwidth is not None:
            _validate_real(self.effective_bandwidth, "effective_bandwidth", positive=True)
        if not isinstance(self.config, WelchPSDConfig | MultitaperPSDConfig):
            raise TypeError("config must be a WelchPSDConfig or MultitaperPSDConfig")

        for value, name, minimum in (
            (self.signal_length, "signal_length", 1),
            (self.samples_per_window, "samples_per_window", 2),
            (self.samples_overlap, "samples_overlap", 0),
            (self.segment_count, "segment_count", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.samples_per_window > self.signal_length:
            raise ValueError("samples_per_window cannot exceed signal_length")
        if self.samples_overlap >= self.samples_per_window:
            raise ValueError("samples_overlap must be smaller than samples_per_window")
        expected_segments = 1 + (self.signal_length - self.samples_per_window) // (
            self.samples_per_window - self.samples_overlap
        )
        if self.segment_count != expected_segments:
            raise ValueError(
                f"segment_count must be {expected_segments} for the recorded window metadata"
            )
        if isinstance(self.config, MultitaperPSDConfig):
            if self.effective_bandwidth != float(self.config.bandwidth):
                raise ValueError("multitaper effective_bandwidth must equal config.bandwidth")
        elif self.effective_bandwidth is not None:
            raise ValueError("Welch effective_bandwidth must be None")

        frequencies.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sampling_frequency", sampling_frequency)
        object.__setattr__(self, "bin_spacing", bin_spacing)


def _validate_real(value: float, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _validated_parameters(
    signal_length: int, sampling_frequency: float, config: PSDConfig
) -> tuple[float, int, int]:
    fs = _validate_real(sampling_frequency, "sampling_frequency", positive=True)
    if config.detrend not in ("constant", "linear", "none"):
        raise ValueError("detrend must be 'constant', 'linear', or 'none'")
    if config.scaling not in ("density", "bin_power"):
        raise ValueError("scaling must be 'density' or 'bin_power'")
    window_length = _validate_real(config.window_length, "window_length", positive=True)
    overlap = _validate_real(config.overlap, "overlap")
    nperseg = int(round(window_length * fs))
    noverlap = int(round(overlap * fs))
    if nperseg < 2:
        raise ValueError("window_length must represent at least two samples")
    if nperseg > signal_length:
        raise ValueError(
            f"window_length requires {nperseg} samples but signal has {signal_length}"
        )
    if noverlap >= nperseg:
        raise ValueError("overlap must be smaller than window_length")
    return fs, nperseg, noverlap


def _segment_view(data: FloatArray, nperseg: int, noverlap: int) -> NDArray[np.float64]:
    step = nperseg - noverlap
    count = 1 + (data.size - nperseg) // step
    shape = (count, nperseg)
    strides = (data.strides[0] * step, data.strides[0])
    segments = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides).copy()
    return np.asarray(segments, dtype=np.float64)


def _detrend_segments(segments: NDArray[np.float64], mode: DetrendMode) -> NDArray[np.float64]:
    if mode == "none":
        return segments
    return np.asarray(scipy_detrend(segments, axis=-1, type=mode), dtype=np.float64)


def _multitaper_psd(
    data: FloatArray,
    fs: float,
    nperseg: int,
    noverlap: int,
    config: MultitaperPSDConfig,
) -> tuple[FloatArray, FloatArray, int]:
    bandwidth = _validate_real(config.bandwidth, "bandwidth", positive=True)
    if config.multitaper_weighting not in ("uniform", "eigen"):
        raise ValueError("multitaper_weighting must be 'uniform' or 'eigen'")
    time_half_bandwidth = config.window_length * bandwidth / 2.0
    if time_half_bandwidth < 1.0:
        raise ValueError("bandwidth and window_length must give time-half-bandwidth >= 1")
    # Match the established MNE candidate pool: request floor(2*NW) tapers,
    # then apply the low-bias concentration filter when requested.
    taper_count = max(1, int(np.floor(2.0 * time_half_bandwidth)))
    tapers, ratios = dpss(
        nperseg,
        time_half_bandwidth,
        Kmax=taper_count,
        sym=False,
        norm=2,
        return_ratios=True,
    )
    tapers = np.atleast_2d(np.asarray(tapers, dtype=np.float64))
    ratios = np.atleast_1d(np.asarray(ratios, dtype=np.float64))
    if config.low_bias:
        keep = ratios >= 0.9
        if np.any(keep):
            tapers = tapers[keep]
            ratios = ratios[keep]

    segments = _detrend_segments(_segment_view(data, nperseg, noverlap), config.detrend)
    tapered = segments[:, None, :] * tapers[None, :, :]
    transforms = np.fft.rfft(tapered, axis=-1)
    powers = np.abs(transforms) ** 2 / fs
    if nperseg % 2 == 0:
        powers[..., 1:-1] *= 2.0
    else:
        powers[..., 1:] *= 2.0
    weights = np.ones_like(ratios) if config.multitaper_weighting == "uniform" else ratios
    values = np.average(powers, axis=1, weights=weights).mean(axis=0)
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        int(segments.shape[0]),
    )


def compute_psd(
    signal: object,
    sampling_frequency: float,
    config: PSDConfig,
) -> PSDResult:
    """Compute a one-sided PSD without hidden library defaults."""
    if not isinstance(config, WelchPSDConfig | MultitaperPSDConfig):
        raise TypeError("config must be a WelchPSDConfig or MultitaperPSDConfig")
    data = validate_signal(signal)  # type: ignore[arg-type]
    fs, nperseg, noverlap = _validated_parameters(data.size, sampling_frequency, config)
    segment_count = 1 + (data.size - nperseg) // (nperseg - noverlap)

    if isinstance(config, WelchPSDConfig):
        scipy_detrend_mode: str | bool = False if config.detrend == "none" else config.detrend
        frequencies, values = welch(
            data,
            fs=fs,
            window=config.window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            detrend=scipy_detrend_mode,
            return_onesided=True,
            scaling="density",
            average="mean",
        )
        frequencies = np.asarray(frequencies, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
    else:
        frequencies, values, segment_count = _multitaper_psd(
            data, fs, nperseg, noverlap, config
        )

    bin_spacing = fs / nperseg
    if config.scaling == "bin_power":
        values = values * bin_spacing
    return PSDResult(
        frequencies=frequencies,
        values=values,
        bin_spacing=bin_spacing,
        effective_bandwidth=(
            float(config.bandwidth) if isinstance(config, MultitaperPSDConfig) else None
        ),
        sampling_frequency=fs,
        signal_length=int(data.size),
        samples_per_window=nperseg,
        samples_overlap=noverlap,
        segment_count=segment_count,
        config=config,
    )


__all__ = ["MultitaperPSDConfig", "PSDResult", "WelchPSDConfig", "compute_psd"]
