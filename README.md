# Amrita BioSignal Feature Engine

Amrita BioSignal Feature Engine (ABFE) is a typed, domain-neutral Python
package for reproducible feature extraction from pre-windowed, one-dimensional
biomedical signals.

ABFE computes time-domain, frequency-domain, and entropy features. It does not
segment, filter, round, or infer physiological frequency bands. Those choices
remain explicit responsibilities of the caller's signal-processing pipeline.

> The ABFE `v0.1.0` release candidate is frozen. Once the GitHub Release is
> published, install its versioned artifact rather than a floating branch in
> research pipelines.

## Installation

ABFE is currently installed from GitHub Releases:

1. Open the [ABFE Releases page](https://github.com/shivapratap/amrita-biosignal-feature-engine/releases).
2. Select the release you want to install.
3. Under **Assets**, download the file whose name ends in `.whl` (for v0.1.0,
   this is `amrita_biosignal_feature_engine-0.1.0-py3-none-any.whl`).
4. Open a terminal in the folder containing the downloaded file and run:

```bash
python -m pip install amrita_biosignal_feature_engine-0.1.0-py3-none-any.whl
```

If you downloaded the file to another folder, provide its full path instead of
just the filename. You can verify the installation with:

```bash
python -c "import amrita_biosignal_feature_engine as abfe; print(abfe.__version__)"
```

ABFE supports Python 3.10 through 3.13 and requires NumPy 1.24 or later and
SciPy 1.10 or later. PyPI publication is intentionally deferred until the
package has received additional user testing.

## Quick start

```python
import numpy as np

from amrita_biosignal_feature_engine import (
    BandPowerRatioRequest,
    BandPowerRequest,
    ExtractorConfig,
    FeatureExtractor,
    WelchPSDConfig,
)

sampling_frequency = 200.0
time = np.arange(400, dtype=np.float64) / sampling_frequency
window = np.sin(2.0 * np.pi * 10.0 * time) + 0.5 * np.sin(
    2.0 * np.pi * 25.0 * time
)

extractor = FeatureExtractor(
    ExtractorConfig(
        sampling_frequency,
        WelchPSDConfig(window_length=2.0, overlap=1.0),
    )
)
result = extractor.extract(
    window,
    features=(
        "root_mean_square",
        "approximate_entropy",
        "spectral_entropy",
        BandPowerRequest("power_8_12_hz", (8.0, 12.0)),
        BandPowerRatioRequest(
            "power_8_12_over_20_30",
            (8.0, 12.0),
            (20.0, 30.0),
        ),
    ),
)

print(result.values)
print(result.failed_features)
print(result.diagnostics)
print(result.provenance)
```

All requested frequency features share one explicitly configured PSD. The
result records the PSD configuration, achieved bin spacing, segment count,
sampling frequency, signal length, and package version.

## Implemented features

The immutable scalar registry contains 26 ordered features:

- Time domain: minimum, maximum, sum, mean, median, population standard
  deviation and variance, excess kurtosis, skewness, mean absolute value, RMS,
  peak-to-peak amplitude, integrated absolute value, waveform length,
  zero-crossing count, and slope-sign-change count.
- Entropy: approximate, permutation, fuzzy, distribution, and SVD entropy.
- Frequency domain: peak, mean, and median frequency; SEF95; and spectral
  entropy.

Caller-named band power, relative band power, and directed band-power ratios
are available through explicit request objects. The structured sample-entropy
profile remains a direct function rather than being collapsed into an invented
scalar registry value.

## Public API at a glance

The main package contains extraction and PSD orchestration. Individual
scientific calculations live in domain-specific submodules so their inputs are
explicit.

| Import path | Public API |
| --- | --- |
| `amrita_biosignal_feature_engine` | `FeatureExtractor`, `ExtractorConfig`, `BandPowerRequest`, `BandPowerRatioRequest`, `ExtractionResult`, `BatchExtractionResult`, `ExtractionProvenance`, `WelchPSDConfig`, `MultitaperPSDConfig`, `PSDResult`, `compute_psd`, `__version__` |
| `amrita_biosignal_feature_engine.time_domain` | `minimum`, `maximum`, `sum_value`, `mean`, `median`, `standard_deviation`, `variance`, `kurtosis`, `skewness`, `mean_absolute_value`, `root_mean_square`, `peak_to_peak`, `integrated_absolute_value`, `waveform_length`, `zero_crossing_count`, `slope_sign_change_count` |
| `amrita_biosignal_feature_engine.frequency_domain` | `peak_frequency`, `mean_frequency`, `median_frequency`, `spectral_edge_frequency`, `spectral_entropy`, `band_power`, `band_power_ratio` |
| `amrita_biosignal_feature_engine.entropy` | `approximate_entropy`, `permutation_entropy`, `fuzzy_entropy`, `distribution_entropy`, `svd_entropy`, `sample_entropy_profile`, `SampleEntropyProfile` |
| `amrita_biosignal_feature_engine.feature_registry` | `FEATURE_REGISTRY`, `DEFAULT_FEATURE_NAMES`, `FeatureSpec`, `FeatureDomain`, `FeatureInput`, `get_feature_spec`, `select_features` |
| `amrita_biosignal_feature_engine.diagnostics` | `ABFEWarning`, `FrequencyResolutionWarning`, `DiagnosticSeverity`, `DiagnosticCode`, `FeatureDiagnostic` |
| `amrita_biosignal_feature_engine.validation` | `validate_signal`, `validate_nonnegative_threshold` |

Welch and multitaper PSD construction is explicit through `WelchPSDConfig` and
`MultitaperPSDConfig`. Frequency functions consume a `PSDResult`, allowing one
PSD to be reused without hidden spectral defaults.

See the [complete API reference](docs/api-reference.md),
[PSD definitions](docs/psd.md), [validation contract](docs/validation.md),
[entropy validation](docs/entropy-validation.md), and
[API design](docs/api-design.md) for parameters, return values, exceptions,
diagnostics, and numerical definitions.

## Test your installation

After installing ABFE, download or clone this repository and run the included
self-checking example:

```bash
python examples/api_smoke_test.py
```

The script exercises the public extractor, direct time-domain, PSD,
frequency-domain, entropy, registry, and batch APIs. A successful run ends
with `ABFE API smoke test passed.`

## Failure and degeneracy policy

Structural input and request errors raise descriptive exceptions before
computation. Mathematically undefined scalar results use `NaN`, never a false
`0.0`. Extraction results pair undefined or failed values with structured
diagnostics, while genuine numerical zero remains distinguishable. Batch
extraction preserves row alignment and diagnoses invalid windows.

Validated signal arrays are finite, real, numeric, one-dimensional, owned,
read-only `float64` arrays. ABFE returns full floating-point precision and does
not perform reporting-time rounding.

## Scientific validation

The offline suite covers definitions, invariants, invalid inputs, degeneracy,
immutability, dispatch, diagnostics, and provenance. A separate CI job compares
entropy and multitaper behavior against pinned external authorities, including
AntroPy, MNE, `sampen-profile`, and the relevant DIHC reference implementations.
External authorities are validation dependencies only and are not installed at
runtime.

## Performance

ABFE includes a deterministic timing, peak-memory, and profiling harness. The
v0.1.0 development baseline and its limitations are documented in
[docs/performance-baseline.md](docs/performance-baseline.md). Pairwise-distance
entropy functions and the structured sample-entropy profile have quadratic
time and/or memory characteristics and deserve particular care for long
windows. Performance redesign is planned for v0.2.0; v0.1.0 prioritizes the
currently validated numerical definitions.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not reference"
python -m ruff check .
python -m mypy --no-incremental
```

Numerically consequential changes must include definitions, before/after
evidence, and focused regression tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). ABFE is released
under the [MIT License](LICENSE).
