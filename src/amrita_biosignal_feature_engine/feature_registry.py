"""Immutable metadata registry for scalar ABFE features."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class FeatureDomain(str, Enum):
    """Scientific domain containing a registered feature."""

    TIME = "time"
    FREQUENCY = "frequency"
    ENTROPY = "entropy"
    COMPLEXITY = "complexity"


class FeatureInput(str, Enum):
    """Shared computation consumed by a registered feature."""

    SIGNAL = "signal"
    PSD = "psd"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable metadata for one scalar feature."""

    name: str
    domain: FeatureDomain
    input_kind: FeatureInput
    description: str
    request_required: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("feature name must be nonempty and have no surrounding whitespace")
        if not self.description:
            raise ValueError("feature description must be nonempty")
        if not isinstance(self.request_required, bool):
            raise TypeError("request_required must be a boolean")


def _spec(
    name: str,
    domain: FeatureDomain,
    input_kind: FeatureInput,
    description: str,
    *,
    request_required: bool = False,
) -> FeatureSpec:
    return FeatureSpec(name, domain, input_kind, description, request_required)


_FEATURE_SPECS = (
    _spec("minimum", FeatureDomain.TIME, FeatureInput.SIGNAL, "Minimum sample value."),
    _spec("maximum", FeatureDomain.TIME, FeatureInput.SIGNAL, "Maximum sample value."),
    _spec("sum_value", FeatureDomain.TIME, FeatureInput.SIGNAL, "Sum of sample values."),
    _spec("mean", FeatureDomain.TIME, FeatureInput.SIGNAL, "Arithmetic sample mean."),
    _spec("median", FeatureDomain.TIME, FeatureInput.SIGNAL, "Sample median."),
    _spec(
        "standard_deviation",
        FeatureDomain.TIME,
        FeatureInput.SIGNAL,
        "Population standard deviation.",
    ),
    _spec("variance", FeatureDomain.TIME, FeatureInput.SIGNAL, "Population variance."),
    _spec("kurtosis", FeatureDomain.TIME, FeatureInput.SIGNAL, "Excess kurtosis."),
    _spec("skewness", FeatureDomain.TIME, FeatureInput.SIGNAL, "Bias-corrected skewness."),
    _spec(
        "mean_absolute_value",
        FeatureDomain.TIME,
        FeatureInput.SIGNAL,
        "Mean absolute sample amplitude.",
    ),
    _spec(
        "root_mean_square", FeatureDomain.TIME, FeatureInput.SIGNAL, "Root mean square amplitude."
    ),
    _spec("peak_to_peak", FeatureDomain.TIME, FeatureInput.SIGNAL, "Peak-to-peak amplitude."),
    _spec(
        "integrated_absolute_value",
        FeatureDomain.TIME,
        FeatureInput.SIGNAL,
        "Sum of absolute sample amplitudes.",
    ),
    _spec(
        "waveform_length",
        FeatureDomain.TIME,
        FeatureInput.SIGNAL,
        "Cumulative absolute sample-to-sample change.",
    ),
    _spec(
        "zero_crossing_count",
        FeatureDomain.TIME,
        FeatureInput.SIGNAL,
        "Count of sign transitions with zero runs removed.",
    ),
    _spec(
        "slope_sign_change_count",
        FeatureDomain.TIME,
        FeatureInput.SIGNAL,
        "Count of adjacent slope reversals.",
    ),
    _spec(
        "approximate_entropy",
        FeatureDomain.ENTROPY,
        FeatureInput.SIGNAL,
        "Self-match-inclusive Pincus approximate entropy.",
    ),
    _spec(
        "permutation_entropy",
        FeatureDomain.ENTROPY,
        FeatureInput.SIGNAL,
        "Bandt-Pompe ordinal-pattern entropy.",
    ),
    _spec(
        "fuzzy_entropy",
        FeatureDomain.ENTROPY,
        FeatureInput.SIGNAL,
        "Shape-based fuzzy entropy.",
    ),
    _spec(
        "distribution_entropy",
        FeatureDomain.ENTROPY,
        FeatureInput.SIGNAL,
        "Normalized entropy of embedding-distance bins.",
    ),
    _spec(
        "svd_entropy",
        FeatureDomain.ENTROPY,
        FeatureInput.SIGNAL,
        "Entropy of delayed-embedding singular values.",
    ),
    _spec(
        "lempel_ziv_complexity",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "Median-binarized LZ76 exhaustive-history complexity.",
    ),
    _spec(
        "hjorth_mobility",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "Hjorth mobility in samples^-1.",
    ),
    _spec(
        "hjorth_complexity",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "Dimensionless Hjorth complexity.",
    ),
    _spec(
        "fisher_information",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "SVD-spectrum Fisher information.",
    ),
    _spec(
        "petrosian_fractal_dimension",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "Petrosian fractal dimension.",
    ),
    _spec(
        "katz_fractal_dimension",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "Katz fractal dimension.",
    ),
    _spec(
        "higuchi_fractal_dimension",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "Higuchi multiscale curve-length fractal dimension.",
    ),
    _spec(
        "detrended_fluctuation_analysis",
        FeatureDomain.COMPLEXITY,
        FeatureInput.SIGNAL,
        "Detrended-fluctuation scaling exponent.",
    ),
    _spec(
        "peak_frequency", FeatureDomain.FREQUENCY, FeatureInput.PSD, "Frequency of maximum PSD."
    ),
    _spec(
        "mean_frequency",
        FeatureDomain.FREQUENCY,
        FeatureInput.PSD,
        "Power-weighted mean frequency.",
    ),
    _spec(
        "median_frequency",
        FeatureDomain.FREQUENCY,
        FeatureInput.PSD,
        "Frequency below which half of integrated power lies.",
    ),
    _spec(
        "spectral_edge_frequency_95",
        FeatureDomain.FREQUENCY,
        FeatureInput.PSD,
        "Frequency below which 95 percent of integrated power lies.",
    ),
    _spec(
        "spectral_entropy",
        FeatureDomain.FREQUENCY,
        FeatureInput.PSD,
        "Normalized Shannon entropy of PSD-bin powers.",
    ),
)

if len({spec.name for spec in _FEATURE_SPECS}) != len(_FEATURE_SPECS):
    raise RuntimeError("feature registry contains duplicate names")

FEATURE_REGISTRY: Mapping[str, FeatureSpec] = MappingProxyType(
    {spec.name: spec for spec in _FEATURE_SPECS}
)
DEFAULT_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name, spec in FEATURE_REGISTRY.items() if not spec.request_required
)


def get_feature_spec(name: str) -> FeatureSpec:
    """Return metadata for ``name``, raising a descriptive ``KeyError``."""
    try:
        return FEATURE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown feature name: {name!r}") from exc


def select_features(
    *, domains: Iterable[FeatureDomain] | None = None
) -> tuple[FeatureSpec, ...]:
    """Return a new ordered tuple selected by scientific domain."""
    if domains is None:
        return tuple(FEATURE_REGISTRY.values())
    selected_domains = frozenset(domains)
    if any(not isinstance(domain, FeatureDomain) for domain in selected_domains):
        raise TypeError("domains must contain FeatureDomain values")
    return tuple(spec for spec in FEATURE_REGISTRY.values() if spec.domain in selected_domains)


__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "FeatureDomain",
    "FeatureInput",
    "FeatureSpec",
    "get_feature_spec",
    "select_features",
]
