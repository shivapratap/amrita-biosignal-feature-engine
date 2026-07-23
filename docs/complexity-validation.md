# Fractal and complexity numerical design

This document freezes the v0.2.0 definitions for eight fractal/complexity
features mapped from DIHC Feature Manager and one new
largest-Lyapunov-exponent feature. Lempel–Ziv complexity, Hjorth mobility,
Hjorth complexity, SVD-spectrum Fisher information, Petrosian fractal
dimension, Katz fractal dimension, and Higuchi fractal dimension are
implemented. The remaining two definitions are design specifications and are
not yet implementation claims.

The nine planned public function names are:

1. `lempel_ziv_complexity`
2. `hjorth_mobility`
3. `hjorth_complexity`
4. `fisher_information`
5. `petrosian_fractal_dimension`
6. `katz_fractal_dimension`
7. `higuchi_fractal_dimension`
8. `detrended_fluctuation_analysis`
9. `largest_lyapunov_exponent`

They live, or upon implementation will live, in
`amrita_biosignal_feature_engine.complexity`. Individual scientific functions
will remain submodule-only; they will not be promoted into the package's
top-level namespace.

## Package-wide contract

All functions accept finite, real, numeric, one-dimensional pre-windowed
signals through `validate_signal`. Boolean, string, object, complex, empty,
multidimensional, NaN, and infinite inputs are rejected. Integer parameters
reject booleans.

Invalid structure or configuration raises `TypeError` or `ValueError`.
Mathematically undefined estimates return `NaN`. Implementations must never
catch an arbitrary exception and return `0.0`; a numerical zero is retained
only when the requested quantity is genuinely defined as zero.

All scalar results are Python `float` values at full precision. No
reporting-time rounding, hidden filtering, resampling, segmentation, or
physiological preset is permitted.

The immutable registry includes `FeatureDomain.COMPLEXITY`; all nine
computations consume `FeatureInput.SIGNAL`. Parameterized extractor use must
record every resolved parameter in immutable provenance. The largest Lyapunov
exponent will use a frozen `LargestLyapunovRequest` because its reconstruction
and fit parameters have no scientifically universal defaults.

NumPy and SciPy remain the only runtime dependencies. External packages are
allowed only as pinned validation authorities.

## 1. Lempel-Ziv complexity

### Definition

The real-valued signal is converted to a binary sequence using a deterministic
median split:

```text
symbol[i] = 1 if signal[i] >= median(signal), otherwise 0
```

The sequence is parsed left to right with the Lempel-Ziv 1976 exhaustive
history production rule. Let `c(n)` be the resulting phrase count for a binary
sequence of length `n`.

The public function exposes:

```text
lempel_ziv_complexity(signal, *, normalize=True)
```

When `normalize=False`, the function returns `c(n)`. When `normalize=True`,
the result is:

```text
c(n) * log2(n) / n
```

The normalized finite-sample value is dimensionless but is not forcibly
clipped to `[0, 1]`.

### Validation and degeneracy

- Minimum signal length: 2 samples.
- `normalize` must be a boolean.
- A constant signal returns `NaN`; its one-symbol encoding cannot represent
  the intended variability/complexity estimate.
- Median ties are always assigned to symbol 1.

### Expected invariances

The result is invariant to translation and positive scaling. For signals with
no samples equal to the median, negative scaling complements the binary labels
and preserves the phrase count. Median ties remain assigned to symbol 1 after
negative scaling, so exact negative-scale invariance is not claimed for tied
signals and must be tested as a documented edge case.

### Authorities

- Lempel and Ziv, *On the Complexity of Finite Sequences*, IEEE Transactions
  on Information Theory 22(1), 1976, DOI
  [10.1109/TIT.1976.1055501](https://doi.org/10.1109/TIT.1976.1055501).
- Pinned AntroPy `lziv_complexity`, supplied with ABFE's independently
  generated binary sequence.
- Pinned DIHC Feature Manager is a compatibility comparison for the median
  binarization decision, not the sole numerical authority.

## 2. Hjorth mobility

### Definition

For population variance `var` and first difference `dx = diff(signal)`:

```text
mobility = sqrt(var(dx, ddof=0) / var(signal, ddof=0))
```

The value is expressed in samples⁻¹ and is dimensionless with respect to
signal amplitude. ABFE will not silently multiply it by sampling frequency.

### Validation and degeneracy

- Minimum signal length: 2 samples.
- Zero signal variance returns `NaN`.
- A nonconstant linear ramp has zero derivative variance and therefore returns
  the genuine value `0.0`.

### Expected invariances

Mobility is invariant to translation and multiplication by any nonzero
constant, including sign reversal.

### Authorities

- Hjorth, *EEG analysis based on time domain properties*,
  Electroencephalography and Clinical Neurophysiology 29(3), 1970, DOI
  [10.1016/0013-4694(70)90143-4](https://doi.org/10.1016/0013-4694(70)90143-4).
- Pinned AntroPy `hjorth_params`.
- An independent direct population-variance oracle.

## 3. Hjorth complexity

### Definition

Let:

```text
mobility_x  = sqrt(var(diff(x)) / var(x))
mobility_dx = sqrt(var(diff(diff(x))) / var(diff(x)))
complexity  = mobility_dx / mobility_x
```

All variances use `ddof=0`. The result is dimensionless.

### Validation and degeneracy

- Minimum signal length: 3 samples.
- Zero signal variance returns `NaN`.
- Zero first-difference variance or zero signal mobility returns `NaN`;
  consequently, the complexity of a perfectly linear ramp is undefined.

### Expected invariances

Complexity is invariant to translation and multiplication by any nonzero
constant, including sign reversal.

### Authorities

- Hjorth (1970), as cited above.
- Pinned AntroPy `hjorth_params`.
- An independent direct population-variance oracle.

## 4. SVD-spectrum Fisher information

### Definition

The public name is `fisher_information`, but the documentation must call the
quantity **SVD-spectrum Fisher information** to distinguish it from other
Fisher-information definitions.

The function signature is:

```text
fisher_information(signal, *, order=2, delay=1)
```

The signal is mean-centered. A delayed embedding is formed with rows:

```text
[x[i], x[i + delay], ..., x[i + (order - 1) * delay]]
```

If `s` contains singular values in NumPy's descending order and
`p = s / sum(s)`, the result is:

```text
sum((p[i + 1] - p[i])**2 / p[i] for i = 0 .. order - 2)
```

Only positive denominator probabilities participate. The result is
dimensionless.

### Parameters and validation

- `order`: integer, default 2, minimum 2.
- `delay`: integer samples, default 1, minimum 1.
- Minimum signal length: `(order - 1) * delay + 2`.
- Rank-zero or rank-one embeddings return `NaN`.
- A zero singular-value sum returns `NaN`.

### Expected invariances

Mean-centering makes the estimate invariant to translation. Normalizing the
singular values makes it invariant to multiplication by any nonzero constant,
including sign reversal.

### Authorities

- The explicit formula in the DIHC methodology and corrected DIHC Feature
  Manager implementation, pinned to a commit during validation.
- An independent eigenvalue-based oracle using the embedding Gram matrix.
- Hand-calculated two- and three-probability spectra.

ABFE will not use the legacy `pyeeg.fisher_info` behavior, which computes an
SVD-entropy quantity rather than this Fisher-information formula.

## 5. Petrosian fractal dimension

### Definition

Let `n` be the signal length and `N_delta` the number of sign changes between
successive first differences. Zero differences do not create a sign change.

```text
PFD = log10(n) /
      (log10(n) + log10(n / (n + 0.4 * N_delta)))
```

The result is dimensionless.

### Validation and degeneracy

- Minimum signal length: 3 samples.
- A constant signal has `N_delta = 0` and returns the genuine graph dimension
  `1.0`.
- Repeated values are handled deterministically through the zero-difference
  rule.

### Expected invariances

The result is invariant to translation and multiplication by any nonzero
constant, including sign reversal.

### Authorities

- Petrosian, *Kolmogorov complexity of finite sequences and recognition of
  different preictal EEG patterns*, IEEE Symposium on Computer-Based Medical
  Systems, 1995, DOI
  [10.1109/CBMS.1995.465426](https://doi.org/10.1109/CBMS.1995.465426).
- Pinned AntroPy `petrosian_fd`.
- An independent sign-change/formula oracle.

## 6. Katz fractal dimension

### Definition

For:

```text
L = sum(abs(diff(signal)))
a = L / (n - 1)
d = max(abs(signal - signal[0]))
```

the result is:

```text
KFD = log10(L / a) /
      (log10(d / L) + log10(L / a))
```

Algebraically, `L / a = n - 1`; the unreduced form is retained in the
documentation to state Katz's geometric quantities. The result is
dimensionless.

### Validation and degeneracy

- Minimum signal length: 2 samples.
- `L == 0`, `a == 0`, or `d == 0` returns `NaN`.
- A zero denominator returns `NaN`.

### Expected invariances

The result is invariant to translation and multiplication by any nonzero
constant, including sign reversal.

### Authorities

- Katz, *Fractals and the analysis of waveforms*, Computers in Biology and
  Medicine 18(3), 1988, DOI
  [10.1016/0010-4825(88)90041-8](https://doi.org/10.1016/0010-4825(88)90041-8).
- Pinned AntroPy `katz_fd`.
- An independent path-length/displacement oracle.

## 7. Higuchi fractal dimension

### Definition

The function signature is:

```text
higuchi_fractal_dimension(signal, *, k_max=10)
```

For each scale `k = 1 .. k_max` and offset `m = 1 .. k`, ABFE constructs
Higuchi's subsampled curve, computes its normalized length `L_m(k)`, and
averages valid offsets to obtain `L(k)`. The fractal dimension is the ordinary
least-squares slope of:

```text
log(L(k)) versus log(1 / k)
```

Natural logarithms are used; the fitted slope is independent of logarithm
base. The result is dimensionless.

### Parameters and validation

- `k_max`: integer, default 10, minimum 2.
- Minimum signal length: `2 * k_max + 1`.
- Every fitted scale must contain at least one valid two-point curve for every
  included offset.
- At least two positive finite `L(k)` values are required.
- Constant input or an indeterminate regression returns `NaN`.

### Expected invariances

The fitted slope is invariant to translation and multiplication by any
nonzero constant, including sign reversal.

### Authorities

- Higuchi, *Approach to an irregular time series on the basis of the fractal
  theory*, Physica D 31, 1988, DOI
  [10.1016/0167-2789(88)90081-4](https://doi.org/10.1016/0167-2789(88)90081-4).
- Pinned AntroPy `higuchi_fd` with the same `k_max`.
- An independent literal-loop implementation of Higuchi's equations.

AntroPy's private linear regression adds a fixed `1e-9` to the OLS
denominator. ABFE uses the unmodified OLS equation above. Pinned AntroPy and
DIHC comparisons therefore use a tight absolute tolerance of `5e-9`; the
independent equation-level oracle is the full-precision authority.

## 8. Detrended fluctuation analysis

### Definition

The public function will expose:

```text
detrended_fluctuation_analysis(
    signal,
    *,
    scales=None,
    minimum_scale=4,
    maximum_scale_fraction=0.1,
    scale_ratio=1.2,
    detrend_order=1,
)
```

ABFE first forms the integrated demeaned signal:

```text
y = cumsum(signal - mean(signal))
```

For each scale, `y` is divided into non-overlapping segments beginning at the
first sample. A polynomial of `detrend_order` is fitted and removed in each
segment. `F(scale)` is the root mean square of all retained residual samples.
The DFA exponent is the ordinary least-squares slope of:

```text
log(F(scale)) versus log(scale)
```

The result is dimensionless.

### Parameters, defaults, and validation

- Explicit `scales`, when supplied, must be a strictly increasing tuple of at
  least two unique integers.
- When `scales=None`, candidate scales are
  `floor(minimum_scale * scale_ratio**i)` for successive nonnegative integers
  `i`. Duplicate integers are removed while preserving order, and values above
  `floor(maximum_scale_fraction * signal_length)` are excluded.
- `minimum_scale`: integer, default 4, and strictly greater than
  `detrend_order + 1`.
- `maximum_scale_fraction`: finite and in `(0, 0.5]`, default 0.1.
- `scale_ratio`: finite and strictly greater than 1, default 1.2.
- `detrend_order`: integer, default 1, minimum 0.
- Each scale must exceed `detrend_order + 1` and provide at least two complete
  segments.
- Default minimum signal length: 50 samples; explicit scales impose their own
  derived minimum.
- At least two positive finite fluctuation values are required.
- Constant input or an indeterminate regression returns `NaN`.

### Expected invariances

The exponent is invariant to translation and multiplication by any nonzero
constant, including sign reversal. Multiplication changes the log-fluctuation
intercept, not its slope.

### Authorities

- Peng et al., *Mosaic organization of DNA nucleotides*, Physical Review E
  49(2), 1994, DOI
  [10.1103/PhysRevE.49.1685](https://doi.org/10.1103/PhysRevE.49.1685).
- Pinned AntroPy `detrended_fluctuation` whenever its resolved default scale
  tuple matches ABFE's. For 50–57 samples, ABFE's explicit generator includes
  scales `(4, 5)`, while AntroPy 0.2.2 stops at `(4,)`; this short-input
  difference is deliberate because a one-point log-log regression is
  indeterminate.
- An independent explicit segmentation, polynomial-fit, and regression oracle.

## 9. Largest Lyapunov exponent

### Scope and method

ABFE will estimate only the **largest** Lyapunov exponent with the Rosenstein,
Collins, and De Luca method. It will not claim to calculate a full Lyapunov
spectrum.

The direct function will require all scientifically consequential parameters:

```text
largest_lyapunov_exponent(
    signal,
    *,
    sampling_frequency,
    embedding_dimension,
    delay_samples,
    minimum_separation_samples,
    fit_start,
    fit_end,
)
```

There are deliberately no defaults for these six keyword-only parameters.
Extractor use will be through a frozen `LargestLyapunovRequest`; the
extractor's sampling frequency may satisfy `sampling_frequency`, while all
reconstruction and fit parameters remain explicit.

### Definition

1. Mean-center the signal and construct a delay embedding with
   `embedding_dimension` coordinates separated by `delay_samples`.
2. For each reference vector, find its nearest finite Euclidean-distance
   neighbor whose index differs by more than
   `minimum_separation_samples`.
3. For each future step `k` in `range(fit_start, fit_end)`, retain pairs for
   which both trajectories remain in the embedding and have positive finite
   separation.
4. Compute the mean natural logarithm of separation at each valid `k`.
5. Fit ordinary least squares against time `k / sampling_frequency`.
6. Return the fitted slope in inverse seconds (`s^-1`).

The fit interval is half-open: `fit_start` is included and `fit_end` is
excluded.

### Parameters and validation

- `sampling_frequency`: finite positive real number in hertz.
- `embedding_dimension`: integer, minimum 2.
- `delay_samples`: integer, minimum 1.
- `minimum_separation_samples`: integer, minimum 0.
- `fit_start`: integer, minimum 0.
- `fit_end`: integer and at least `fit_start + 3`.
- At least three valid divergence-time points and two valid neighbor pairs per
  fitted point are required.
- The derived conservative minimum signal length is:

```text
(embedding_dimension - 1) * delay_samples
+ fit_end
+ 2 * minimum_separation_samples
+ 2
```

- No admissible neighbor set, zero separations only, insufficient divergence
  points, or an indeterminate fit returns `NaN`.
- ABFE will not automatically choose a delay, embedding dimension, temporal
  exclusion, or visually convenient linear fit region.

### Units and invariances

The result is in `s^-1`. Translation and multiplication by any nonzero
constant do not change the ideal slope: amplitude scaling shifts the
log-distance intercept. Changing the sampling-frequency interpretation changes
the physical time axis and therefore changes the numerical slope in inverse
seconds.

No automatic positive-slope or fit-quality threshold will be imposed. A
positive estimate alone is not sufficient evidence of deterministic chaos.

### Authorities

- Rosenstein, Collins, and De Luca, *A practical method for calculating
  largest Lyapunov exponents from small data sets*, Physica D 65, 1993, DOI
  [10.1016/0167-2789(93)90009-P](https://doi.org/10.1016/0167-2789(93)90009-P).
- A pinned independent Rosenstein implementation used only in reference CI.
- Independent literal nearest-neighbor/divergence/regression oracles.
- Synthetic periodic, quasiperiodic, logistic-map, and Lorenz-system cases
  with fixed seeds and documented sampling conventions.

## Registry and extractor policy

The implemented DIHC-mapped names are registered with the snake-case
identifiers at the beginning of this document. Registry metadata identifies
`FeatureDomain.COMPLEXITY` and `FeatureInput.SIGNAL`; the remaining names will
be added only when their implementations and validation land.

The parameter-free Hjorth, Petrosian, and Katz features may be requested by
registry name. Lempel-Ziv, Fisher information, Higuchi fractal dimension, and
DFA will use documented canonical defaults for name-only extraction and will
also expose direct function parameters.

`FeatureSpec` has an immutable `request_required: bool = False` field.
The `largest_lyapunov_exponent` specification will set it to `True`, remain
discoverable in `FEATURE_REGISTRY`, and be excluded from
`DEFAULT_FEATURE_NAMES`. Passing its bare registry string to extraction will
raise a descriptive `ValueError`.

Lyapunov extraction requires
`LargestLyapunovRequest(output_name=..., embedding_dimension=...,
delay_samples=..., minimum_separation_samples=..., fit_start=...,
fit_end=...)`, analogous to caller-defined band-power requests. Sampling
frequency comes from `ExtractorConfig`.

`ExtractionProvenance` has an immutable per-output `feature_parameters`
mapping. Bare registered parameterized features record
their resolved canonical defaults; request-driven features record every
caller-supplied parameter plus output units. Parameter-free features need no
entry. Nested mappings and contained tuples must be defensively copied and
read-only.

## Numerical test matrix

Every implementation must include:

- invalid type, shape, finiteness, boolean, and parameter tests;
- exact minimum-length and one-sample-short tests;
- constant, linear, periodic, quasiperiodic, seeded white-noise, chirp, and
  seeded chaotic inputs where scientifically applicable;
- translation, positive scaling, negative scaling, and z-score invariance;
- hand-calculated or literal-loop independent oracles;
- pinned external-reference comparisons with version or commit recorded;
- assertions that undefined results are `NaN`, never false zero;
- immutable registry, dispatch, diagnostic, batch-alignment, and provenance
  tests;
- installed-wheel and source-distribution smoke coverage.

Reference comparisons must document any deliberate difference rather than
loosening tolerances until tests pass.

## Performance expectations

Correctness baselines precede optimization.

- Hjorth, Petrosian, Katz, and Lempel-Ziv should be linear or near-linear in
  signal length.
- Higuchi cost grows with signal length and `k_max`.
- DFA cost grows with the number and size of scales.
- Fisher information includes delay embedding and SVD.
- A literal Rosenstein nearest-neighbor search is quadratic in embedding
  count; a later validated implementation may use SciPy spatial indexing while
  preserving temporal-exclusion semantics.

Benchmarks must report parameters, signal length, timing, and peak memory
without encoding machine-specific release thresholds.

## Implementation gate

This design received explicit approval before implementation began. Any later
change to a formula, default, minimum-length rule, degeneracy policy, or unit
is a numerically consequential change and must include before/after evidence
plus focused regression tests under `CONTRIBUTING.md`.
