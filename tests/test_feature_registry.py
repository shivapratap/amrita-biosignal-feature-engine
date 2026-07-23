from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from amrita_biosignal_feature_engine.feature_registry import (
    DEFAULT_FEATURE_NAMES,
    FEATURE_REGISTRY,
    FeatureDomain,
    FeatureInput,
    get_feature_spec,
    select_features,
)

EXPECTED_FEATURE_NAMES = (
    "minimum",
    "maximum",
    "sum_value",
    "mean",
    "median",
    "standard_deviation",
    "variance",
    "kurtosis",
    "skewness",
    "mean_absolute_value",
    "root_mean_square",
    "peak_to_peak",
    "integrated_absolute_value",
    "waveform_length",
    "zero_crossing_count",
    "slope_sign_change_count",
    "approximate_entropy",
    "permutation_entropy",
    "fuzzy_entropy",
    "distribution_entropy",
    "svd_entropy",
    "lempel_ziv_complexity",
    "hjorth_mobility",
    "hjorth_complexity",
    "fisher_information",
    "petrosian_fractal_dimension",
    "katz_fractal_dimension",
    "higuchi_fractal_dimension",
    "detrended_fluctuation_analysis",
    "largest_lyapunov_exponent",
    "peak_frequency",
    "mean_frequency",
    "median_frequency",
    "spectral_edge_frequency_95",
    "spectral_entropy",
)


def test_registry_has_exact_stable_scalar_catalog() -> None:
    assert tuple(FEATURE_REGISTRY) == EXPECTED_FEATURE_NAMES
    assert DEFAULT_FEATURE_NAMES == tuple(
        name for name in EXPECTED_FEATURE_NAMES if name != "largest_lyapunov_exponent"
    )
    assert len(FEATURE_REGISTRY) == len(set(FEATURE_REGISTRY))
    assert "sample_entropy_profile" not in FEATURE_REGISTRY


def test_registry_and_specs_are_immutable() -> None:
    with pytest.raises(TypeError):
        FEATURE_REGISTRY["new"] = FEATURE_REGISTRY["mean"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        FEATURE_REGISTRY["mean"].name = "changed"  # type: ignore[misc]


def test_get_feature_spec_and_unknown_name() -> None:
    spec = get_feature_spec("spectral_entropy")
    assert spec.domain is FeatureDomain.FREQUENCY
    assert spec.input_kind is FeatureInput.PSD
    with pytest.raises(KeyError, match="unknown feature name"):
        get_feature_spec("not_a_feature")


def test_select_features_returns_new_ordered_tuple_without_mutation() -> None:
    before = tuple(FEATURE_REGISTRY)
    entropy = select_features(domains=[FeatureDomain.ENTROPY])
    assert tuple(spec.name for spec in entropy) == (
        "approximate_entropy",
        "permutation_entropy",
        "fuzzy_entropy",
        "distribution_entropy",
        "svd_entropy",
    )
    assert tuple(FEATURE_REGISTRY) == before
    assert select_features() is not select_features()
    complexity = select_features(domains=[FeatureDomain.COMPLEXITY])
    assert tuple(spec.name for spec in complexity) == (
        "lempel_ziv_complexity",
        "hjorth_mobility",
        "hjorth_complexity",
        "fisher_information",
        "petrosian_fractal_dimension",
        "katz_fractal_dimension",
        "higuchi_fractal_dimension",
        "detrended_fluctuation_analysis",
        "largest_lyapunov_exponent",
    )
    assert all(
        spec.request_required == (spec.name == "largest_lyapunov_exponent")
        for spec in complexity
    )


def test_select_features_accepts_multiple_domains_in_registry_order() -> None:
    selected = select_features(domains=[FeatureDomain.FREQUENCY, FeatureDomain.TIME])
    assert selected[0].name == "minimum"
    assert selected[-1].name == "spectral_entropy"
    assert all(spec.domain is not FeatureDomain.ENTROPY for spec in selected)


def test_select_features_rejects_non_domain_values() -> None:
    with pytest.raises(TypeError, match="FeatureDomain"):
        select_features(domains=["entropy"])  # type: ignore[list-item]
