from __future__ import annotations

import amrita_biosignal_feature_engine as abfe


def test_top_level_exports_only_core_infrastructure() -> None:
    assert set(abfe.__all__) == {
        "BandPowerRatioRequest",
        "BandPowerRequest",
        "BatchExtractionResult",
        "ExtractionProvenance",
        "ExtractionResult",
        "ExtractorConfig",
        "FeatureExtractor",
        "MultitaperPSDConfig",
        "PSDResult",
        "WelchPSDConfig",
        "__version__",
        "compute_psd",
    }


def test_individual_feature_functions_are_not_flattened_at_top_level() -> None:
    for name in (
        "root_mean_square",
        "band_power",
        "distribution_entropy",
        "fuzzy_entropy",
        "sample_entropy_profile",
        "permutation_entropy",
        "svd_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
        "petrosian_fractal_dimension",
        "katz_fractal_dimension",
        "lempel_ziv_complexity",
        "fisher_information",
    ):
        assert name not in abfe.__all__
        assert not hasattr(abfe, name)
