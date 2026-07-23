from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

import amrita_biosignal_feature_engine.extractor as extractor_module
from amrita_biosignal_feature_engine import (
    BandPowerRatioRequest,
    BandPowerRequest,
    ExtractorConfig,
    FeatureExtractor,
    MultitaperPSDConfig,
    WelchPSDConfig,
)
from amrita_biosignal_feature_engine.diagnostics import (
    DiagnosticCode,
    DiagnosticSeverity,
    FrequencyResolutionWarning,
)


def signal_and_extractor() -> tuple[NDArray[np.float64], FeatureExtractor]:
    sampling_frequency = 100.0
    time = np.arange(400, dtype=np.float64) / sampling_frequency
    signal = np.sin(2.0 * np.pi * 10.0 * time)
    config = ExtractorConfig(
        sampling_frequency,
        WelchPSDConfig(window_length=2.0, overlap=1.0),
    )
    return signal, FeatureExtractor(config)


def test_extract_mixed_features_preserves_order_and_provenance() -> None:
    signal, extractor = signal_and_extractor()
    result = extractor.extract(
        signal,
        features=("root_mean_square", "approximate_entropy", "peak_frequency"),
    )
    assert tuple(result.values) == (
        "root_mean_square",
        "approximate_entropy",
        "peak_frequency",
    )
    assert result.values["root_mean_square"] == pytest.approx(np.sqrt(0.5))
    assert np.isfinite(result.values["approximate_entropy"])
    assert result.values["peak_frequency"] == 10.0
    assert result.failed_features == ()
    assert result.provenance.signal_length == 400
    assert result.provenance.psd_config is extractor.config.psd
    assert result.provenance.psd_bin_spacing == 0.5
    assert result.provenance.psd_segment_count == 3


def test_psd_is_computed_once_and_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    signal, extractor = signal_and_extractor()
    original = extractor._compute_psd
    calls = 0

    def counted(data: NDArray[np.float64]) -> object:
        nonlocal calls
        calls += 1
        return original(data)

    monkeypatch.setattr(extractor, "_compute_psd", counted)
    result = extractor.extract(
        signal,
        features=("peak_frequency", "mean_frequency", "spectral_entropy"),
    )
    assert calls == 1
    assert all(np.isfinite(value) for value in result.values.values())


def test_signal_only_request_does_not_compute_psd(monkeypatch: pytest.MonkeyPatch) -> None:
    signal, extractor = signal_and_extractor()

    def forbidden(_data: NDArray[np.float64]) -> object:
        raise AssertionError("PSD should not be computed")

    monkeypatch.setattr(extractor, "_compute_psd", forbidden)
    result = extractor.extract(signal, features=("mean", "root_mean_square"))
    assert result.provenance.psd_config is None
    assert result.failed_features == ()


def test_package_version_metadata_is_resolved_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal, extractor = signal_and_extractor()
    calls = 0

    def counted(_distribution_name: str) -> str:
        nonlocal calls
        calls += 1
        return "9.8.7"

    extractor_module._package_version.cache_clear()
    monkeypatch.setattr(extractor_module, "version", counted)
    try:
        first = extractor.extract(signal, features=("mean",))
        second = extractor.extract(signal, features=("root_mean_square",))
        batch = extractor.extract_batch((signal, signal), features=("mean",))
        assert calls == 1
        assert first.provenance.package_version == "9.8.7"
        assert second.provenance.package_version == "9.8.7"
        assert all(row.provenance.package_version == "9.8.7" for row in batch.rows)
    finally:
        extractor_module._package_version.cache_clear()


def test_explicit_band_power_and_ratio_requests() -> None:
    signal, extractor = signal_and_extractor()
    time = np.arange(signal.size, dtype=np.float64) / extractor.config.sampling_frequency
    signal = 2.0 * np.sin(2.0 * np.pi * 10.0 * time) + np.sin(
        2.0 * np.pi * 25.0 * time
    )
    result = extractor.extract(
        signal,
        features=(
            BandPowerRequest("power_8_12", (8.0, 12.0)),
            BandPowerRequest("relative_8_12", (8.0, 12.0), relative=True),
            BandPowerRatioRequest("low_over_high", (8.0, 12.0), (20.0, 30.0)),
        ),
    )
    assert result.values["power_8_12"] == pytest.approx(2.0, rel=1e-12)
    assert result.values["relative_8_12"] == pytest.approx(0.8, abs=1e-12)
    assert result.values["low_over_high"] == pytest.approx(4.0, rel=1e-12)
    assert result.failed_features == ()


def test_underresolved_band_retains_value_warning_and_diagnostic() -> None:
    signal, extractor = signal_and_extractor()
    request = BandPowerRequest("narrow_power", (9.8, 10.2))
    with pytest.warns(FrequencyResolutionWarning):
        result = extractor.extract(signal, features=(request,))
    assert result.values["narrow_power"] > 0.0
    assert result.failed_features == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].severity is DiagnosticSeverity.WARNING
    assert result.diagnostics[0].code is DiagnosticCode.FREQUENCY_RESOLUTION


def test_undefined_nan_is_not_confused_with_real_zero() -> None:
    _, extractor = signal_and_extractor()
    result = extractor.extract(
        np.ones(400), features=("zero_crossing_count", "approximate_entropy")
    )
    assert result.values["zero_crossing_count"] == 0.0
    assert "zero_crossing_count" not in result.failed_features
    assert np.isnan(result.values["approximate_entropy"])
    assert result.failed_features == ("approximate_entropy",)
    assert result.diagnostics[0].code is DiagnosticCode.UNDEFINED_RESULT


def test_unexpected_feature_exception_becomes_structured_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal, extractor = signal_and_extractor()

    def broken(_signal: object) -> float:
        raise RuntimeError("deliberate failure")

    patched: Mapping[str, extractor_module.ScalarSignalFunction] = {
        **extractor_module._SIGNAL_DISPATCH,
        "mean": broken,
    }
    monkeypatch.setattr(extractor_module, "_SIGNAL_DISPATCH", patched)
    result = extractor.extract(signal, features=("mean",))
    assert np.isnan(result.values["mean"])
    assert result.failed_features == ("mean",)
    assert result.diagnostics[0].code is DiagnosticCode.COMPUTATION_ERROR
    assert result.diagnostics[0].exception_type == "RuntimeError"


def test_shared_psd_failure_marks_only_psd_features() -> None:
    signal = np.arange(20, dtype=np.float64)
    extractor = FeatureExtractor(
        ExtractorConfig(100.0, WelchPSDConfig(window_length=2.0, overlap=1.0))
    )
    result = extractor.extract(signal, features=("mean", "peak_frequency", "spectral_entropy"))
    assert result.values["mean"] == pytest.approx(9.5)
    assert np.isnan(result.values["peak_frequency"])
    assert np.isnan(result.values["spectral_entropy"])
    assert result.failed_features == ("peak_frequency", "spectral_entropy")
    assert all(
        diagnostic.code is DiagnosticCode.PSD_COMPUTATION_ERROR
        for diagnostic in result.diagnostics
    )
    assert result.provenance.psd_config is extractor.config.psd
    assert result.provenance.psd_bin_spacing is None


def test_multitaper_provenance_records_effective_bandwidth() -> None:
    sampling_frequency = 100.0
    time = np.arange(400, dtype=np.float64) / sampling_frequency
    signal = np.sin(2.0 * np.pi * 10.0 * time)
    extractor = FeatureExtractor(
        ExtractorConfig(
            sampling_frequency,
            MultitaperPSDConfig(window_length=2.0, overlap=1.0, bandwidth=4.0),
        )
    )
    result = extractor.extract(signal, features=("spectral_entropy",))
    assert result.provenance.psd_effective_bandwidth == 4.0


def test_request_validation_happens_before_signal_validation() -> None:
    _, extractor = signal_and_extractor()
    with pytest.raises(KeyError, match="unknown feature"):
        extractor.extract([np.nan], features=("unknown",))
    with pytest.raises(ValueError, match="unique"):
        extractor.extract(
            [np.nan], features=("mean", BandPowerRequest("mean", (1.0, 2.0)))
        )
    with pytest.raises(TypeError, match="iterable"):
        extractor.extract([1.0, 2.0], features="mean")


def test_band_above_nyquist_is_rejected_before_psd() -> None:
    signal, extractor = signal_and_extractor()
    with pytest.raises(ValueError, match="Nyquist"):
        extractor.extract(signal, features=(BandPowerRequest("bad", (40.0, 60.0)),))


@pytest.mark.parametrize(
    "bad_signal",
    [
        [0.0, np.nan],
        np.array([[1.0, 2.0]]),
        [True, False],
        [1.0 + 1.0j, 2.0 + 0.0j],
    ],
)
def test_single_extract_preserves_strict_structural_validation(bad_signal: object) -> None:
    _, extractor = signal_and_extractor()
    with pytest.raises((TypeError, ValueError)):
        extractor.extract(bad_signal, features=("mean",))  # type: ignore[arg-type]


def test_source_input_is_not_modified() -> None:
    signal, extractor = signal_and_extractor()
    original = signal.copy()
    extractor.extract(signal, features=("mean", "spectral_entropy"))
    np.testing.assert_array_equal(signal, original)
    assert signal.flags.writeable


def test_batch_preserves_invalid_row_alignment_and_schema() -> None:
    signal, extractor = signal_and_extractor()
    batch = extractor.extract_batch(
        (signal, [0.0, np.nan], 2.0 * signal),
        features=("mean", "peak_frequency"),
    )
    assert len(batch.rows) == 3
    assert batch.feature_names == ("mean", "peak_frequency")
    assert batch.rows[0].failed_features == ()
    assert batch.rows[1].failed_features == ("mean", "peak_frequency")
    assert all(
        diagnostic.code is DiagnosticCode.INVALID_WINDOW
        for diagnostic in batch.rows[1].diagnostics
    )
    assert batch.rows[2].failed_features == ()


def test_empty_batch_and_empty_feature_request_are_supported() -> None:
    signal, extractor = signal_and_extractor()
    assert extractor.extract_batch((), features=("mean",)).rows == ()
    result = extractor.extract(signal, features=())
    assert dict(result.values) == {}
    assert result.diagnostics == ()
    assert result.provenance.psd_config is None
