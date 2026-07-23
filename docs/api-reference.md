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
- `LargestLyapunovRequest`: names a Rosenstein largest-Lyapunov output and
  requires every reconstruction and fit parameter.
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

These thirteen names form the deliberately small top-level API. Scientific
functions are submodule-only by design.

## Time-domain functions

`amrita_biosignal_feature_engine.time_domain` exports canonical functions for
`minimum`, `maximum`, `sum_value`, `mean`, `median`, `standard_deviation`,
`variance`, `kurtosis`, `skewness`, `mean_absolute_value`,
`root_mean_square`, `peak_to_peak`, `integrated_absolute_value`,
`waveform_length`, `zero_crossing_count`, and `slope_sign_change_count`.

## Frequency-domain functions

`amrita_biosignal_feature_engine.frequency_domain` exports
`peak_frequency`, `mean_frequency`, `median_frequency`,
`spectral_edge_frequency`, `spectral_entropy`, `band_power`, and
`band_power_ratio`. These functions accept `PSDResult`, not raw signals.
Under-resolved caller-defined bands emit
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

- `lempel_ziv_complexity`
- `hjorth_mobility`
- `hjorth_complexity`
- `fisher_information`
- `petrosian_fractal_dimension`
- `katz_fractal_dimension`
- `higuchi_fractal_dimension`
- `detrended_fluctuation_analysis`
- `largest_lyapunov_exponent`

These functions consume raw validated signals. Their formulas, units,
minimum-length rules, degeneracy behavior, and intentional Petrosian plateau
policy are recorded in `complexity-validation.md`.

### Feature families

| Family | Features |
| --- | --- |
| Time domain | `minimum`, `maximum`, `sum_value`, `mean`, `median`, `standard_deviation`, `variance`, `kurtosis`, `skewness`, `mean_absolute_value`, `root_mean_square`, `peak_to_peak`, `integrated_absolute_value`, `waveform_length`, `zero_crossing_count`, `slope_sign_change_count` |
| Frequency domain | `peak_frequency`, `mean_frequency`, `median_frequency`, `spectral_edge_frequency`, `spectral_entropy` |
| Entropy | `approximate_entropy`, `permutation_entropy`, `fuzzy_entropy`, `distribution_entropy`, `svd_entropy`, `sample_entropy_profile`, `SampleEntropyProfile` |
| Complexity and fractal features | `lempel_ziv_complexity`, `hjorth_mobility`, `hjorth_complexity`, `fisher_information`, `petrosian_fractal_dimension`, `katz_fractal_dimension`, `higuchi_fractal_dimension`, `detrended_fluctuation_analysis`, `largest_lyapunov_exponent` |

### Complexity signatures and contracts

| Function | Keyword parameters | Unit | Minimum signal length | Undefined result |
| --- | --- | --- | --- | --- |
| `lempel_ziv_complexity(signal, *, normalize=True)` | `normalize`: strict boolean | dimensionless | 2 | constant signal |
| `hjorth_mobility(signal)` | none | samples⁻¹ | 2 | zero signal variance |
| `hjorth_complexity(signal)` | none | dimensionless | 3 | zero signal or first-difference variance |
| `fisher_information(signal, *, order=2, delay=1)` | integer `order >= 2`; integer `delay >= 1` | dimensionless | `(order - 1) * delay + 2` | numerical embedding rank at most one |
| `petrosian_fractal_dimension(signal)` | none | dimensionless | 3 | no additional estimator-specific `NaN` case |
| `katz_fractal_dimension(signal)` | none | dimensionless | 2 | zero path geometry or indeterminate logarithmic ratio |
| `higuchi_fractal_dimension(signal, *, k_max=10)` | integer `k_max >= 2` | dimensionless | `2 * k_max + 1` | constant curve or indeterminate regression |
| `detrended_fluctuation_analysis(signal, *, scales=None, minimum_scale=4, maximum_scale_fraction=0.1, scale_ratio=1.2, detrend_order=1)` | explicit scale tuple or deterministic scale generator | dimensionless | 50 by default; `2 * max(scales)` for explicit scales | fewer than two positive fluctuations or indeterminate regression |
| `largest_lyapunov_exponent(signal, *, sampling_frequency, embedding_dimension, delay_samples, minimum_separation_samples, fit_start, fit_end)` | all six keyword-only parameters are required | s⁻¹ | derived from reconstruction and fit parameters | no admissible nonzero neighbours, insufficient divergence points, or indeterminate regression |

All integer parameters reject booleans. Invalid structure and parameters raise
`TypeError` or `ValueError`; estimator degeneracy returns `NaN`. Shared signal
validation requires a finite, real, numeric, one-dimensional input and returns
an owned, read-only `float64` array internally.

`largest_lyapunov_exponent` requires:

- finite positive `sampling_frequency`;
- `embedding_dimension >= 2`;
- `delay_samples >= 1`;
- `minimum_separation_samples >= 0`;
- `fit_start >= 0`;
- `fit_end >= fit_start + 3`.

The fit interval is half-open, `[fit_start, fit_end)`. Exact duplicate
embeddings are excluded before nearest-neighbour selection. The result uses
time in seconds, and a positive value alone is not proof of deterministic
chaos.

## Registry and diagnostics

`amrita_biosignal_feature_engine.feature_registry` exposes immutable feature
metadata and ordered selection helpers. The structured sample-entropy profile
is excluded because it is not scalar. Caller-defined band outputs are requests,
not global registry mutations.

`ExtractionProvenance.feature_parameters` is an immutable per-output mapping.
It records canonical defaults for registered parameterized features and the
resolved settings of explicit band requests.

`LargestLyapunovRequest` is a top-level orchestration request. It requires the
embedding dimension, delay, temporal exclusion, and half-open fit interval;
the extractor supplies its configured sampling frequency.

```python
LargestLyapunovRequest(
    output_name="largest_lyapunov_s_inverse",
    embedding_dimension=3,
    delay_samples=2,
    minimum_separation_samples=10,
    fit_start=0,
    fit_end=6,
)
```

It is frozen and request-only. `largest_lyapunov_exponent` is registered for
discovery but excluded from `DEFAULT_FEATURE_NAMES`; the other eight
complexity features are default scalar outputs. Successful extraction records
all resolved parameters in immutable per-output provenance.

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
