from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from amrita_biosignal_feature_engine import (
    BandPowerRatioRequest,
    BandPowerRequest,
    BatchExtractionResult,
    ExtractionProvenance,
    ExtractionResult,
    ExtractorConfig,
    WelchPSDConfig,
)
from amrita_biosignal_feature_engine.diagnostics import (
    DiagnosticCode,
    DiagnosticSeverity,
    FeatureDiagnostic,
)


def provenance(*names: str) -> ExtractionProvenance:
    return ExtractionProvenance(
        package_version="0.1.0.dev0",
        signal_length=100,
        sampling_frequency=100.0,
        requested_features=names,
    )


def test_band_power_request_is_frozen_and_validated() -> None:
    request = BandPowerRequest("power_8_12", (8, 12), relative=True)
    assert request.band == (8.0, 12.0)
    with pytest.raises(FrozenInstanceError):
        request.band = (1.0, 2.0)  # type: ignore[misc]


@pytest.mark.parametrize("band", [(-1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (0.0, np.inf)])
def test_band_power_request_rejects_invalid_band(band: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        BandPowerRequest("power", band)


@pytest.mark.parametrize("band", [[1.0, 2.0], (True, 2.0), ("1", 2.0)])
def test_band_power_request_rejects_invalid_band_type(band: object) -> None:
    with pytest.raises(TypeError):
        BandPowerRequest("power", band)  # type: ignore[arg-type]


def test_band_ratio_request_preserves_explicit_direction() -> None:
    request = BandPowerRatioRequest("low_over_high", (8, 12), (20, 30))
    assert request.numerator_band == (8.0, 12.0)
    assert request.denominator_band == (20.0, 30.0)


@pytest.mark.parametrize("name", ["", " name", "name "])
def test_requests_reject_invalid_output_name(name: str) -> None:
    with pytest.raises(ValueError, match="output_name"):
        BandPowerRequest(name, (1.0, 2.0))


def test_extractor_config_requires_explicit_valid_sampling_frequency() -> None:
    config = ExtractorConfig(100, WelchPSDConfig())
    assert config.sampling_frequency == 100.0
    with pytest.raises(TypeError, match="real number"):
        ExtractorConfig(True, WelchPSDConfig())
    with pytest.raises(ValueError, match="positive"):
        ExtractorConfig(0.0, WelchPSDConfig())
    with pytest.raises(TypeError, match="WelchPSDConfig"):
        ExtractorConfig(100.0, object())  # type: ignore[arg-type]


def test_provenance_without_psd_rejects_psd_metadata() -> None:
    with pytest.raises(ValueError, match="requires psd_config"):
        ExtractionProvenance(
            "0.1.0", 100, 100.0, ("mean",), psd_bin_spacing=0.5
        )


def test_provenance_with_psd_records_complete_shared_metadata() -> None:
    config = WelchPSDConfig()
    result = ExtractionProvenance(
        "0.1.0",
        1000,
        100.0,
        ("spectral_entropy",),
        psd_config=config,
        psd_bin_spacing=0.5,
        psd_segment_count=9,
    )
    assert result.psd_config is config
    assert result.psd_bin_spacing == 0.5
    assert result.psd_segment_count == 9


def test_provenance_defensively_freezes_feature_parameters() -> None:
    source: dict[str, dict[str, object]] = {
        "lempel_ziv_complexity": {"normalize": True}
    }
    result = ExtractionProvenance(
        "0.2.0.dev0",
        100,
        100.0,
        ("lempel_ziv_complexity",),
        feature_parameters=source,
    )
    source["lempel_ziv_complexity"]["normalize"] = False
    assert result.feature_parameters["lempel_ziv_complexity"]["normalize"] is True
    with pytest.raises(TypeError):
        result.feature_parameters["lempel_ziv_complexity"]["normalize"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        result.feature_parameters["new"] = {}  # type: ignore[index]


def test_provenance_rejects_unrequested_or_mutable_parameters() -> None:
    with pytest.raises(ValueError, match="requested features"):
        ExtractionProvenance(
            "0.2.0.dev0",
            100,
            100.0,
            ("mean",),
            feature_parameters={"other": {"value": 1}},
        )
    with pytest.raises(TypeError, match="scalar values or tuples"):
        ExtractionProvenance(
            "0.2.0.dev0",
            100,
            100.0,
            ("mean",),
            feature_parameters={"mean": {"value": [1, 2]}},
        )


def test_extraction_result_copies_and_freezes_values() -> None:
    source = {"mean": 2.0}
    result = ExtractionResult(source, (), provenance("mean"))
    source["mean"] = 10.0
    assert result.values["mean"] == 2.0
    with pytest.raises(TypeError):
        result.values["mean"] = 3.0  # type: ignore[index]


def test_result_distinguishes_warning_from_failed_features() -> None:
    diagnostics = (
        FeatureDiagnostic(
            "band",
            DiagnosticSeverity.WARNING,
            DiagnosticCode.FREQUENCY_RESOLUTION,
            "band is under-resolved",
        ),
        FeatureDiagnostic(
            "entropy",
            DiagnosticSeverity.UNDEFINED,
            DiagnosticCode.UNDEFINED_RESULT,
            "result is mathematically undefined",
        ),
        FeatureDiagnostic(
            "frequency",
            DiagnosticSeverity.ERROR,
            DiagnosticCode.PSD_COMPUTATION_ERROR,
            "PSD failed",
            "ValueError",
        ),
    )
    result = ExtractionResult(
        {"band": 1.0, "entropy": np.nan, "frequency": np.nan},
        diagnostics,
        provenance("band", "entropy", "frequency"),
    )
    assert result.failed_features == ("entropy", "frequency")


def test_result_schema_must_match_provenance_order() -> None:
    with pytest.raises(ValueError, match="names and order"):
        ExtractionResult({"mean": 1.0, "maximum": 2.0}, (), provenance("maximum", "mean"))


def test_diagnostic_must_reference_a_result_value() -> None:
    diagnostic = FeatureDiagnostic(
        "missing",
        DiagnosticSeverity.ERROR,
        DiagnosticCode.COMPUTATION_ERROR,
        "failed",
    )
    with pytest.raises(ValueError, match="identify a result"):
        ExtractionResult({"mean": 1.0}, (diagnostic,), provenance("mean"))


def test_batch_result_preserves_rows_and_shared_schema() -> None:
    first = ExtractionResult({"mean": 1.0}, (), provenance("mean"))
    second = ExtractionResult({"mean": 2.0}, (), provenance("mean"))
    batch = BatchExtractionResult((first, second))
    assert batch.feature_names == ("mean",)
    assert batch.rows == (first, second)
    assert BatchExtractionResult(()).feature_names == ()


def test_batch_rejects_inconsistent_schema_order() -> None:
    first = ExtractionResult(
        {"mean": 1.0, "maximum": 2.0}, (), provenance("mean", "maximum")
    )
    second = ExtractionResult(
        {"maximum": 2.0, "mean": 1.0}, (), provenance("maximum", "mean")
    )
    with pytest.raises(ValueError, match="same feature schema"):
        BatchExtractionResult((first, second))
