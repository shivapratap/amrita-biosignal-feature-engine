"""Immutable request and result models for ABFE extraction."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from . import time_domain
from .complexity import (
    _resolve_dfa_scales,
    detrended_fluctuation_analysis,
    fisher_information,
    higuchi_fractal_dimension,
    hjorth_complexity,
    hjorth_mobility,
    katz_fractal_dimension,
    lempel_ziv_complexity,
    petrosian_fractal_dimension,
)
from .diagnostics import (
    DiagnosticCode,
    DiagnosticSeverity,
    FeatureDiagnostic,
    FrequencyResolutionWarning,
)
from .entropy import (
    approximate_entropy,
    distribution_entropy,
    fuzzy_entropy,
    permutation_entropy,
    svd_entropy,
)
from .feature_registry import (
    DEFAULT_FEATURE_NAMES,
    FEATURE_REGISTRY,
    FeatureInput,
    get_feature_spec,
)
from .frequency_domain import (
    band_power,
    band_power_ratio,
    mean_frequency,
    median_frequency,
    peak_frequency,
    spectral_edge_frequency,
    spectral_entropy,
)
from .psd import (
    MultitaperPSDConfig,
    PSDConfig,
    PSDResult,
    WelchPSDConfig,
    compute_psd,
)
from .validation import FloatArray, validate_signal

ScalarSignalFunction: TypeAlias = Callable[[ArrayLike], float]
ScalarPSDFunction: TypeAlias = Callable[[PSDResult], float]


def _validate_output_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("output_name must be a string")
    if not value or value.strip() != value:
        raise ValueError("output_name must be nonempty and have no surrounding whitespace")
    return value


def _validate_band(value: tuple[float, float], *, name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a (low_hz, high_hz) tuple")
    if any(isinstance(limit, bool | np.bool_) or not isinstance(limit, Real) for limit in value):
        raise TypeError(f"{name} limits must be real numbers")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite(low) or not np.isfinite(high) or low < 0.0 or high <= low:
        raise ValueError(f"{name} must satisfy 0 <= low_hz < high_hz")
    return low, high


@dataclass(frozen=True, slots=True)
class BandPowerRequest:
    """Explicit absolute or relative band-power output request."""

    output_name: str
    band: tuple[float, float]
    relative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_name", _validate_output_name(self.output_name))
        object.__setattr__(self, "band", _validate_band(self.band, name="band"))
        if not isinstance(self.relative, bool):
            raise TypeError("relative must be a boolean")


@dataclass(frozen=True, slots=True)
class BandPowerRatioRequest:
    """Explicit numerator-band/denominator-band ratio output request."""

    output_name: str
    numerator_band: tuple[float, float]
    denominator_band: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_name", _validate_output_name(self.output_name))
        object.__setattr__(
            self,
            "numerator_band",
            _validate_band(self.numerator_band, name="numerator_band"),
        )
        object.__setattr__(
            self,
            "denominator_band",
            _validate_band(self.denominator_band, name="denominator_band"),
        )


@dataclass(frozen=True, slots=True)
class ExtractorConfig:
    """Immutable sampling and PSD configuration for feature extraction."""

    sampling_frequency: float
    psd: PSDConfig

    def __post_init__(self) -> None:
        if isinstance(self.sampling_frequency, bool | np.bool_) or not isinstance(
            self.sampling_frequency, Real
        ):
            raise TypeError("sampling_frequency must be a real number")
        sampling_frequency = float(self.sampling_frequency)
        if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be finite and positive")
        if not isinstance(self.psd, WelchPSDConfig | MultitaperPSDConfig):
            raise TypeError("psd must be a WelchPSDConfig or MultitaperPSDConfig")
        object.__setattr__(self, "sampling_frequency", sampling_frequency)


@dataclass(frozen=True, slots=True)
class ExtractionProvenance:
    """Immutable record of the configuration and shared PSD actually used."""

    package_version: str
    signal_length: int
    sampling_frequency: float
    requested_features: tuple[str, ...]
    psd_config: PSDConfig | None = None
    psd_bin_spacing: float | None = None
    psd_effective_bandwidth: float | None = None
    psd_segment_count: int | None = None
    feature_parameters: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.package_version:
            raise ValueError("package_version must be nonempty")
        if isinstance(self.signal_length, bool) or not isinstance(self.signal_length, int):
            raise TypeError("signal_length must be an integer")
        if self.signal_length < 0:
            raise ValueError("signal_length must be nonnegative")
        if isinstance(self.sampling_frequency, bool | np.bool_) or not isinstance(
            self.sampling_frequency, Real
        ):
            raise TypeError("sampling_frequency must be a real number")
        if not np.isfinite(self.sampling_frequency) or self.sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be finite and positive")
        requested = tuple(self.requested_features)
        if any(not isinstance(name, str) or not name for name in requested):
            raise ValueError("requested_features must contain nonempty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("requested_features must be unique")
        object.__setattr__(self, "requested_features", requested)
        copied_parameters: dict[str, Mapping[str, object]] = {}
        for output_name, parameters in self.feature_parameters.items():
            if output_name not in requested:
                raise ValueError(
                    "feature_parameters keys must identify requested features"
                )
            if not isinstance(parameters, Mapping):
                raise TypeError("feature parameter records must be mappings")
            copied: dict[str, object] = {}
            for name, value in parameters.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("feature parameter names must be nonempty strings")
                if isinstance(value, tuple):
                    if any(
                        not (
                            isinstance(item, str | bool | int | float)
                            or item is None
                        )
                        for item in value
                    ):
                        raise TypeError(
                            "feature parameter tuples must contain scalar values"
                        )
                    copied[name] = tuple(value)
                elif isinstance(value, str | bool | int | float) or value is None:
                    copied[name] = value
                else:
                    raise TypeError(
                        "feature parameter values must be scalar values or tuples"
                    )
            copied_parameters[output_name] = MappingProxyType(copied)
        object.__setattr__(
            self, "feature_parameters", MappingProxyType(copied_parameters)
        )

        metadata = (
            self.psd_bin_spacing,
            self.psd_effective_bandwidth,
            self.psd_segment_count,
        )
        if self.psd_config is None:
            if any(value is not None for value in metadata):
                raise ValueError("PSD metadata requires psd_config")
            return
        if not isinstance(self.psd_config, WelchPSDConfig | MultitaperPSDConfig):
            raise TypeError("psd_config must be a WelchPSDConfig or MultitaperPSDConfig")
        if self.psd_bin_spacing is None and self.psd_segment_count is None:
            if self.psd_effective_bandwidth is not None:
                raise ValueError("effective bandwidth requires successful PSD metadata")
            return
        if self.psd_bin_spacing is None or self.psd_segment_count is None:
            raise ValueError("successful PSD provenance requires bin spacing and segment count")
        if not np.isfinite(self.psd_bin_spacing) or self.psd_bin_spacing <= 0.0:
            raise ValueError("psd_bin_spacing must be finite and positive")
        if isinstance(self.psd_segment_count, bool) or not isinstance(
            self.psd_segment_count, int
        ):
            raise TypeError("psd_segment_count must be an integer")
        if self.psd_segment_count < 1:
            raise ValueError("psd_segment_count must be an integer >= 1")
        if self.psd_effective_bandwidth is not None and (
            not np.isfinite(self.psd_effective_bandwidth)
            or self.psd_effective_bandwidth <= 0.0
        ):
            raise ValueError("psd_effective_bandwidth must be finite and positive")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Immutable values, diagnostics, and provenance for one signal window."""

    values: Mapping[str, float]
    diagnostics: tuple[FeatureDiagnostic, ...]
    provenance: ExtractionProvenance

    def __post_init__(self) -> None:
        copied_values = {name: float(value) for name, value in self.values.items()}
        if any(not isinstance(name, str) or not name for name in copied_values):
            raise ValueError("value names must be nonempty strings")
        if tuple(copied_values) != self.provenance.requested_features:
            raise ValueError("value names and order must match requested_features")
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, FeatureDiagnostic) for item in diagnostics):
            raise TypeError("diagnostics must contain FeatureDiagnostic values")
        if any(item.feature_name not in copied_values for item in diagnostics):
            raise ValueError("diagnostic feature_name must identify a result value")
        object.__setattr__(self, "values", MappingProxyType(copied_values))
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def failed_features(self) -> tuple[str, ...]:
        """Return unique outputs with undefined or error diagnostics, in result order."""
        failed = {
            item.feature_name
            for item in self.diagnostics
            if item.severity in (DiagnosticSeverity.UNDEFINED, DiagnosticSeverity.ERROR)
        }
        return tuple(name for name in self.values if name in failed)


@dataclass(frozen=True, slots=True)
class BatchExtractionResult:
    """Immutable, row-aligned collection of single-window results."""

    rows: tuple[ExtractionResult, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if any(not isinstance(row, ExtractionResult) for row in rows):
            raise TypeError("rows must contain ExtractionResult values")
        if rows:
            expected = tuple(rows[0].values)
            if any(tuple(row.values) != expected for row in rows[1:]):
                raise ValueError("all batch rows must have the same feature schema and order")
        object.__setattr__(self, "rows", rows)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the shared ordered feature schema, or an empty tuple."""
        return tuple(self.rows[0].values) if self.rows else ()


FeatureRequest: TypeAlias = str | BandPowerRequest | BandPowerRatioRequest


@dataclass(frozen=True, slots=True)
class _ResolvedFeature:
    output_name: str
    registered_name: str | None
    request: BandPowerRequest | BandPowerRatioRequest | None
    input_kind: FeatureInput


_SIGNAL_DISPATCH: Mapping[str, ScalarSignalFunction] = MappingProxyType(
    {
        "minimum": time_domain.minimum,
        "maximum": time_domain.maximum,
        "sum_value": time_domain.sum_value,
        "mean": time_domain.mean,
        "median": time_domain.median,
        "standard_deviation": time_domain.standard_deviation,
        "variance": time_domain.variance,
        "kurtosis": time_domain.kurtosis,
        "skewness": time_domain.skewness,
        "mean_absolute_value": time_domain.mean_absolute_value,
        "root_mean_square": time_domain.root_mean_square,
        "peak_to_peak": time_domain.peak_to_peak,
        "integrated_absolute_value": time_domain.integrated_absolute_value,
        "waveform_length": time_domain.waveform_length,
        "zero_crossing_count": time_domain.zero_crossing_count,
        "slope_sign_change_count": time_domain.slope_sign_change_count,
        "approximate_entropy": approximate_entropy,
        "permutation_entropy": permutation_entropy,
        "fuzzy_entropy": fuzzy_entropy,
        "distribution_entropy": distribution_entropy,
        "svd_entropy": svd_entropy,
        "lempel_ziv_complexity": lempel_ziv_complexity,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
        "fisher_information": fisher_information,
        "petrosian_fractal_dimension": petrosian_fractal_dimension,
        "katz_fractal_dimension": katz_fractal_dimension,
        "higuchi_fractal_dimension": higuchi_fractal_dimension,
        "detrended_fluctuation_analysis": detrended_fluctuation_analysis,
    }
)


def _spectral_edge_frequency_95(psd: PSDResult) -> float:
    return spectral_edge_frequency(psd, fraction=0.95)


_PSD_DISPATCH: Mapping[str, ScalarPSDFunction] = MappingProxyType(
    {
        "peak_frequency": peak_frequency,
        "mean_frequency": mean_frequency,
        "median_frequency": median_frequency,
        "spectral_edge_frequency_95": _spectral_edge_frequency_95,
        "spectral_entropy": spectral_entropy,
    }
)

if set(_SIGNAL_DISPATCH) | set(_PSD_DISPATCH) != set(FEATURE_REGISTRY):
    raise RuntimeError("extractor dispatch and feature registry are inconsistent")


@cache
def _package_version() -> str:
    """Return the installed version, resolving distribution metadata once."""
    try:
        return version("amrita-biosignal-feature-engine")
    except PackageNotFoundError:  # pragma: no cover - source tree without installation
        return "0+unknown"


def _resolve_features(features: Iterable[FeatureRequest]) -> tuple[_ResolvedFeature, ...]:
    if isinstance(features, str):
        raise TypeError("features must be an iterable of feature names or request objects")
    resolved: list[_ResolvedFeature] = []
    for item in features:
        if isinstance(item, str):
            spec = get_feature_spec(item)
            if spec.request_required:
                raise ValueError(f"feature {item!r} requires an explicit request object")
            resolved.append(_ResolvedFeature(item, item, None, spec.input_kind))
        elif isinstance(item, BandPowerRequest | BandPowerRatioRequest):
            resolved.append(_ResolvedFeature(item.output_name, None, item, FeatureInput.PSD))
        else:
            raise TypeError(
                "features must contain strings, BandPowerRequest, or BandPowerRatioRequest"
            )
    output_names = tuple(item.output_name for item in resolved)
    if len(set(output_names)) != len(output_names):
        raise ValueError("requested output names must be unique")
    return tuple(resolved)


def _feature_parameters(
    resolved: tuple[_ResolvedFeature, ...],
    *,
    signal_length: int | None,
) -> Mapping[str, Mapping[str, object]]:
    parameters: dict[str, Mapping[str, object]] = {}
    for item in resolved:
        if item.registered_name == "lempel_ziv_complexity":
            parameters[item.output_name] = {"normalize": True}
        elif item.registered_name == "fisher_information":
            parameters[item.output_name] = {"order": 2, "delay": 1}
        elif item.registered_name == "higuchi_fractal_dimension":
            parameters[item.output_name] = {"k_max": 10}
        elif item.registered_name == "detrended_fluctuation_analysis":
            dfa_parameters: dict[str, object] = {
                "minimum_scale": 4,
                "maximum_scale_fraction": 0.1,
                "scale_ratio": 1.2,
                "detrend_order": 1,
            }
            if signal_length is not None and signal_length >= 50:
                dfa_parameters["scales"] = _resolve_dfa_scales(
                    signal_length,
                    scales=None,
                    minimum_scale=4,
                    maximum_scale_fraction=0.1,
                    scale_ratio=1.2,
                    detrend_order=1,
                )
            parameters[item.output_name] = dfa_parameters
        elif isinstance(item.request, BandPowerRequest):
            parameters[item.output_name] = {
                "band": item.request.band,
                "relative": item.request.relative,
                "output_units": (
                    "dimensionless"
                    if item.request.relative
                    else "signal_units_squared"
                ),
            }
        elif isinstance(item.request, BandPowerRatioRequest):
            parameters[item.output_name] = {
                "numerator_band": item.request.numerator_band,
                "denominator_band": item.request.denominator_band,
                "output_units": "dimensionless",
            }
    return parameters


def _resolution_diagnostics(
    feature_name: str, captured: list[warnings.WarningMessage]
) -> tuple[FeatureDiagnostic, ...]:
    diagnostics: list[FeatureDiagnostic] = []
    for warning in captured:
        if issubclass(warning.category, FrequencyResolutionWarning):
            diagnostics.append(
                FeatureDiagnostic(
                    feature_name,
                    DiagnosticSeverity.WARNING,
                    DiagnosticCode.FREQUENCY_RESOLUTION,
                    str(warning.message),
                )
            )
            warnings.warn(
                str(warning.message),
                warning.category,
                stacklevel=4,
            )
    return tuple(diagnostics)


def _value_diagnostic(feature_name: str, value: float) -> FeatureDiagnostic | None:
    if np.isnan(value):
        return FeatureDiagnostic(
            feature_name,
            DiagnosticSeverity.UNDEFINED,
            DiagnosticCode.UNDEFINED_RESULT,
            "feature is mathematically undefined for this signal",
        )
    if not np.isfinite(value):
        return FeatureDiagnostic(
            feature_name,
            DiagnosticSeverity.ERROR,
            DiagnosticCode.COMPUTATION_ERROR,
            "feature returned a nonfinite value other than NaN",
        )
    return None


class FeatureExtractor:
    """Extract registered scalar features with shared PSD computation."""

    def __init__(self, config: ExtractorConfig) -> None:
        if not isinstance(config, ExtractorConfig):
            raise TypeError("config must be an ExtractorConfig")
        self._config = config

    @property
    def config(self) -> ExtractorConfig:
        """Return the immutable extractor configuration."""
        return self._config

    def _validate_bands(self, resolved: tuple[_ResolvedFeature, ...]) -> None:
        nyquist = self.config.sampling_frequency / 2.0
        for item in resolved:
            request = item.request
            bands: tuple[tuple[float, float], ...]
            if isinstance(request, BandPowerRequest):
                bands = (request.band,)
            elif isinstance(request, BandPowerRatioRequest):
                bands = (request.numerator_band, request.denominator_band)
            else:
                continue
            if any(high > nyquist for _, high in bands):
                raise ValueError(
                    f"feature {item.output_name!r} has a band above Nyquist frequency {nyquist}"
                )

    def _compute_psd(self, data: FloatArray) -> PSDResult:
        return compute_psd(data, self.config.sampling_frequency, self.config.psd)

    def _compute_psd_feature(self, item: _ResolvedFeature, psd: PSDResult) -> float:
        request = item.request
        if isinstance(request, BandPowerRequest):
            return band_power(psd, band=request.band, relative=request.relative)
        if isinstance(request, BandPowerRatioRequest):
            return band_power_ratio(
                psd,
                numerator_band=request.numerator_band,
                denominator_band=request.denominator_band,
            )
        if item.registered_name is None:
            raise RuntimeError("resolved PSD feature has no dispatch name")
        return _PSD_DISPATCH[item.registered_name](psd)

    def extract(
        self,
        signal: ArrayLike,
        *,
        features: Iterable[FeatureRequest] = DEFAULT_FEATURE_NAMES,
    ) -> ExtractionResult:
        """Extract one valid pre-windowed signal.

        Structural signal and request errors raise. Per-feature numerical
        failures become ``NaN`` values with structured diagnostics.
        """
        resolved = _resolve_features(features)
        self._validate_bands(resolved)
        data = validate_signal(signal)
        needs_psd = any(item.input_kind is FeatureInput.PSD for item in resolved)
        psd: PSDResult | None = None
        psd_error: Exception | None = None
        if needs_psd:
            try:
                psd = self._compute_psd(data)
            except Exception as exc:
                psd_error = exc

        values: dict[str, float] = {}
        diagnostics: list[FeatureDiagnostic] = []
        for item in resolved:
            if item.input_kind is FeatureInput.PSD and psd is None:
                values[item.output_name] = float("nan")
                assert psd_error is not None
                diagnostics.append(
                    FeatureDiagnostic(
                        item.output_name,
                        DiagnosticSeverity.ERROR,
                        DiagnosticCode.PSD_COMPUTATION_ERROR,
                        str(psd_error) or "PSD computation failed",
                        type(psd_error).__name__,
                    )
                )
                continue
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always", FrequencyResolutionWarning)
                    if item.input_kind is FeatureInput.SIGNAL:
                        assert item.registered_name is not None
                        value = float(_SIGNAL_DISPATCH[item.registered_name](data))
                    else:
                        assert psd is not None
                        value = float(self._compute_psd_feature(item, psd))
                diagnostics.extend(_resolution_diagnostics(item.output_name, captured))
            except Exception as exc:
                value = float("nan")
                diagnostics.append(
                    FeatureDiagnostic(
                        item.output_name,
                        DiagnosticSeverity.ERROR,
                        DiagnosticCode.COMPUTATION_ERROR,
                        str(exc) or "feature computation failed",
                        type(exc).__name__,
                    )
                )
            value_diagnostic = _value_diagnostic(item.output_name, value)
            if value_diagnostic is not None and not any(
                diagnostic.feature_name == item.output_name
                and diagnostic.severity is DiagnosticSeverity.ERROR
                for diagnostic in diagnostics
            ):
                diagnostics.append(value_diagnostic)
            if not np.isfinite(value) and not np.isnan(value):
                value = float("nan")
            values[item.output_name] = value

        provenance = ExtractionProvenance(
            package_version=_package_version(),
            signal_length=int(data.size),
            sampling_frequency=self.config.sampling_frequency,
            requested_features=tuple(values),
            psd_config=self.config.psd if needs_psd else None,
            psd_bin_spacing=psd.bin_spacing if psd is not None else None,
            psd_effective_bandwidth=psd.effective_bandwidth if psd is not None else None,
            psd_segment_count=psd.segment_count if psd is not None else None,
            feature_parameters=_feature_parameters(
                resolved, signal_length=int(data.size)
            ),
        )
        return ExtractionResult(values, tuple(diagnostics), provenance)

    def _invalid_batch_row(
        self,
        signal: object,
        resolved: tuple[_ResolvedFeature, ...],
        error: Exception,
    ) -> ExtractionResult:
        try:
            raw = np.asarray(signal)
            signal_length = int(raw.size) if raw.ndim == 1 else 0
        except Exception:
            signal_length = 0
        names = tuple(item.output_name for item in resolved)
        values = {name: float("nan") for name in names}
        diagnostics = tuple(
            FeatureDiagnostic(
                name,
                DiagnosticSeverity.ERROR,
                DiagnosticCode.INVALID_WINDOW,
                str(error) or "invalid signal window",
                type(error).__name__,
            )
            for name in names
        )
        provenance = ExtractionProvenance(
            _package_version(),
            signal_length,
            self.config.sampling_frequency,
            names,
            feature_parameters=_feature_parameters(
                resolved, signal_length=signal_length
            ),
        )
        return ExtractionResult(values, diagnostics, provenance)

    def extract_batch(
        self,
        signals: Iterable[ArrayLike],
        *,
        features: Iterable[FeatureRequest] = DEFAULT_FEATURE_NAMES,
    ) -> BatchExtractionResult:
        """Extract row-aligned windows, retaining invalid rows as diagnosed NaNs."""
        materialized_features = tuple(features)
        resolved = _resolve_features(materialized_features)
        self._validate_bands(resolved)
        rows: list[ExtractionResult] = []
        for signal in signals:
            try:
                rows.append(self.extract(signal, features=materialized_features))
            except Exception as exc:
                rows.append(self._invalid_batch_row(signal, resolved, exc))
        return BatchExtractionResult(tuple(rows))


__all__ = [
    "BandPowerRatioRequest",
    "BandPowerRequest",
    "BatchExtractionResult",
    "ExtractionProvenance",
    "ExtractionResult",
    "ExtractorConfig",
    "FeatureExtractor",
]
