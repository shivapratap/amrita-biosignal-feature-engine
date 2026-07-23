# Public API reference

ABFE accepts pre-windowed one-dimensional signals. It does not filter or
segment input data. Detailed parameter and return contracts are also available
in the docstrings of the objects below.

## Top-level orchestration and PSD API

- `FeatureExtractor`: validates and dispatches scalar feature requests.
- `ExtractorConfig`: stores sampling frequency and one explicit PSD
  configuration.
- `BandPowerRequest`: names an absolute or relative caller-defined band-power
  output.
- `BandPowerRatioRequest`: names a directed numerator/denominator band ratio.
- `ExtractionResult`: immutable ordered values, diagnostics, and provenance for
  one window.
- `BatchExtractionResult`: immutable row-aligned results for multiple windows.
- `ExtractionProvenance`: package, signal, and successful/failed PSD metadata.
- `WelchPSDConfig` and `MultitaperPSDConfig`: method-specific PSD parameters.
- `PSDResult`: validated immutable frequency axis, PSD values, and computation
  metadata.
- `compute_psd`: computes a PSD using only the supplied configuration.
- `__version__`: installed distribution version obtained through
  `importlib.metadata`.

These twelve names form the deliberately small top-level API. Scientific
functions are submodule-only by design.

## Time-domain functions

`amrita_biosignal_feature_engine.time_domain` exports canonical functions for
minimum, maximum, sum, mean, median, population standard deviation and
variance, excess kurtosis, bias-corrected skewness, mean absolute value, root
mean square, peak-to-peak amplitude, integrated absolute value, waveform
length, zero-crossing count, and slope-sign-change count.

## Frequency-domain functions

`amrita_biosignal_feature_engine.frequency_domain` exports peak frequency,
mean frequency, median frequency, spectral edge frequency, spectral entropy,
band power, and band-power ratio. These functions accept `PSDResult`, not raw
signals. Under-resolved caller-defined bands emit
`FrequencyResolutionWarning`; the computed value is retained.

## Entropy functions

`amrita_biosignal_feature_engine.entropy` exports:

- `approximate_entropy`
- `permutation_entropy`
- `fuzzy_entropy`
- `distribution_entropy`
- `svd_entropy`
- `sample_entropy_profile`
- `SampleEntropyProfile`

Spectral entropy remains in `frequency_domain` because it consumes an explicit
shared PSD. Entropy defaults and validation tolerances are described in
`entropy-validation.md` and individual function docstrings.

## Fractal and complexity functions

`amrita_biosignal_feature_engine.complexity` exports:

- `hjorth_mobility`
- `hjorth_complexity`
- `petrosian_fractal_dimension`
- `katz_fractal_dimension`

These functions consume raw validated signals. Their formulas, units,
minimum-length rules, degeneracy behavior, and intentional Petrosian plateau
policy are recorded in `complexity-validation.md`.

## Registry and diagnostics

`amrita_biosignal_feature_engine.feature_registry` exposes immutable feature
metadata and ordered selection helpers. The structured sample-entropy profile
is excluded because it is not scalar. Caller-defined band outputs are requests,
not global registry mutations.

`amrita_biosignal_feature_engine.diagnostics` defines diagnostic codes,
severities, immutable diagnostic records, and frequency-resolution warnings.
Extraction distinguishes genuine zero, mathematically undefined output, a
feature computation error, PSD failure, and invalid batch windows.

## Exceptions and undefined values

Malformed feature requests and structurally invalid signals raise `TypeError`,
`ValueError`, or `KeyError` as appropriate. Mathematically undefined feature
values return `NaN`. `FeatureExtractor` adds structured diagnostics for
undefined values and per-feature failures; it does not replace either with
zero. `KeyboardInterrupt` and `SystemExit` are never swallowed.
