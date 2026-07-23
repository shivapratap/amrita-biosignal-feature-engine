# Changes

This project records every numerically consequential change with evidence,
before/after values, and a regression test.

## Unreleased

### Fractal and complexity design

- Closed the complexity implementation phase after focused independent
  re-audit of remediation commit `e02f88e`: duplicate-embedding Lyapunov
  selection matched the full-distance oracle within approximately `1.1e-16`,
  all 131,068 binary sequences through length 16 matched the independent LZ76
  parser, and no release-blocking finding remained.
- Added the staged v0.2.0 integration and release-readiness design covering
  repository hygiene, public documentation, evidence closure, CI/artifacts,
  final independent audit, version freeze, integration, and publication.
- Corrected Rosenstein nearest-neighbor selection to exclude exact
  zero-distance duplicate embeddings before choosing the nearest admissible
  neighbor. Before the correction, the audited repeated-pattern input selected
  zero pairs and returned `NaN`; afterward it selects the available nonzero
  neighbors and returns `0.23969930756318522 s^-1`, matching the independent
  full-distance-matrix oracle.
- Replaced the quadratic Python-level LZ76 history scan with an
  earliest-occurrence suffix-automaton parser. The phrase definition and
  numerical results are unchanged; exhaustive binary-sequence and pinned
  AntroPy/DIHC comparisons remain the compatibility gates. Added reproducible
  512–4,096-sample benchmark cases to expose parser scaling. Same-machine
  median timings improved from 3.68–197.66 ms to 0.40–3.61 ms across those
  sizes, a 9.2–54.7× speedup.
- Opened the v0.2.0 development cycle at version `0.2.0.dev0`.
- Implemented the first correctness batch: Hjorth mobility, Hjorth complexity,
  Petrosian fractal dimension, and Katz fractal dimension, with strict shared
  validation, explicit degeneracy behavior, registry/extractor dispatch, and
  installed-artifact smoke coverage.
- Added independent formula oracles, invariance and boundary tests, and pinned
  AntroPy comparisons. Petrosian zero-derivative plateaus deliberately remove
  zero slopes before transition counting; this documented ABFE policy differs
  from AntroPy's direct `signbit` transition count.
- Implemented median-binarized Lempel–Ziv 1976 exhaustive-history complexity
  with raw and normalized outputs, deterministic median ties, constant-signal
  degeneracy, published hand-parsed examples, and pinned AntroPy comparisons.
- Added immutable per-output feature-parameter provenance. Name-only
  Lempel–Ziv extraction records its canonical `normalize=True` default, while
  explicit band requests record their resolved bands, relative-power mode, and
  output units.
- Strengthened Lempel–Ziv validation with a structurally independent
  exhaustive-history parser, all 8,188 binary sequences of lengths 2 through
  12, and exact raw-count comparisons against DIHC Feature Manager commit
  `852a79d0178eb762aeaf81960820043365a059c3`.
- Implemented mean-centered SVD-spectrum Fisher information with configurable
  embedding order and delay, numerical rank-zero/rank-one `NaN` handling,
  immutable default-parameter provenance, a hand-derived two-value spectrum,
  an independent Gram-eigenvalue oracle, and exact ordinary-case comparisons
  against the pinned corrected DIHC implementation.
- Implemented Higuchi fractal dimension with configurable `k_max`, the frozen
  minimum-length and degeneracy policies, immutable default-parameter
  provenance, an independent literal-loop oracle, and pinned AntroPy and DIHC
  comparisons. ABFE uses the exact OLS denominator; the reference packages add
  a fixed `1e-9`, producing a documented maximum comparison delta below
  `5e-9`.
- Implemented detrended fluctuation analysis with explicit or deterministic
  automatic scales, polynomial detrending, exact OLS regression, strict
  scale/segment validation, and immutable provenance containing the resolved
  scale tuple and canonical generator settings.
- Added an independent Vandermonde least-squares segmentation oracle,
  seeded white-noise and random-walk checks, affine invariance, and equivalent
  pinned AntroPy/DIHC comparisons. For 50–57 samples, ABFE deliberately uses
  two scales instead of reproducing the references' false one-point `0.0`.
- Implemented the Rosenstein largest Lyapunov exponent with fully explicit
  reconstruction, temporal-exclusion, half-open fit-region, and sampling-rate
  parameters; no scientific defaults or positive-slope threshold are inferred.
- Added the frozen request-only registry policy, `LargestLyapunovRequest`,
  inverse-seconds output, complete immutable provenance, conservative
  minimum-length validation, and undefined-neighbor/divergence handling.
- Validated against an independent full-distance-matrix oracle, seeded
  periodic, quasiperiodic, logistic-map, and RK4 Lorenz trajectories, and
  pinned `nolds 0.6.2`; both Rosenstein implementations recover the logistic
  map's theoretical `ln(2)` exponent within `0.01`.
- Specified all eight time-domain fractal/complexity features mapped from DIHC
  Feature Manager plus a new Rosenstein largest Lyapunov exponent estimator
  before beginning their staged implementation.
- Froze proposed formulas, parameters, defaults, units, minimum-length rules,
  undefined-value behavior, invariances, provenance requirements, and
  independent reference-validation authorities in
  `docs/complexity-validation.md`.
- Retained ABFE's strict policy: malformed inputs and parameters raise,
  mathematically undefined estimates return `NaN`, and no failed computation
  is replaced with a false `0.0`.

## 0.1.0 - 2026-07-23

### Release readiness

- Closed final independent release-candidate audit finding RC-001 by replacing
  the benchmark harness's bare `np.ndarray` return annotation with
  `NDArray[np.float64]`. This is a typing-only correction with no runtime,
  benchmark-methodology, or numerical behavior change. Gate 5 is closed.
- Added a staged release-readiness design with explicit repository, metadata,
  documentation, benchmark, CI/artifact, independent-audit, version-freeze,
  and publication gates. Publishing, pushing, and tagging remain separately
  approved actions.
- Established GitHub-only distribution for v0.1.0, canonical repository and
  maintainer metadata, typed-package/Python classifiers, comprehensive README
  and API guidance, prepared citation metadata, and a policy excluding local
  independent-audit reports from the public repository.
- Added a dependency-free, deterministic benchmark harness covering direct
  time features, signal-only extraction, Welch and multitaper PSD, linear and
  quadratic entropy groups, the structured sample-entropy profile, full mixed
  extraction, and batch extraction. It
  records isolated-process peak memory, environment metadata, correctness
  checks, JSON timing evidence, and cumulative profiles without imposing
  machine-specific performance thresholds.
- The installed-wheel baseline found no v0.1.0 performance blocker. It
  identified repeated package-version metadata reads as a narrow pre-release
  cleanup candidate and deferred pairwise-distance entropy and
  reference-compatible sample-profile redesign to v0.2.0, where numerical
  equivalence can receive dedicated validation.
- Cached installed package-version metadata once per process instead of
  reparsing distribution metadata for every extraction row. Provenance remains
  identical; a regression test verifies repeated single and batch extraction
  performs exactly one metadata lookup.
- Hardened distribution CI to validate wheel and source-distribution metadata
  and manifests, reject caches/audit debris, confirm typed-package and license
  files, install each artifact into an independent environment, and exercise
  direct, PSD, entropy, single-window, batch, version, and provenance behavior
  outside the checkout. A GitHub-only tag workflow is prepared but deliberately
  disabled pending separate release approval.

### Step 6: immutable registry and extraction models

- Closed focused audit finding STEP6-001 by replacing three bare test
  `np.ndarray` annotations with explicit `NDArray[np.float64]` annotations.
  This is a typing-only correction with no runtime or numerical behavior change.
- Added an immutable 26-feature scalar registry with stable ordering, explicit
  scientific domain and signal/PSD input metadata, and non-mutating selection.
- Kept `sample_entropy_profile` direct-only because it is structured output;
  no profile sum or average was invented to force it into a scalar schema.
- Added frozen explicit band-power and band-ratio requests, extractor
  configuration, structured diagnostics, complete PSD provenance, immutable
  single-window results, and row-aligned batch results.
- Result models distinguish warning-only outputs, mathematically undefined
  values, and computation errors without replacing any failure with `0.0`.
- Added explicit signal/PSD dispatch with one shared PSD per window,
  caller-named band power and ratio execution, structured warning capture,
  per-feature undefined/error diagnostics, complete successful/failed PSD
  provenance, and row-aligned batch handling for invalid windows.
- The focused independent audit found no runtime, scientific, or architectural
  defects. Step 6 is frozen after clean offline, reference, lint, strict-typing,
  build, and installed-wheel smoke gates.

### Entropy phase audit closeout

- Closed focused audit finding ENT-001 by parameterizing permutation entropy's
  private ordinal-pattern array as `NDArray[np.intp]`. This is a typing-only
  correction with no runtime or numerical behavior change.
- The focused independent re-audit found no scientific or numerical defects
  across approximate, permutation, fuzzy, distribution, SVD, spectral, or
  sample-profile entropy behavior. The entropy phase is frozen after clean
  offline, pinned-reference, lint, strict-typing, build, and wheel-smoke gates.

### Spectral entropy reference validation

- Completed the entropy-phase validation of the existing PSD-consuming
  spectral entropy API against pinned AntroPy 0.2.2 with explicitly equivalent
  Welch parameters.
- Added periodic, white-noise, chirp, positive/negative scaling, offset,
  z-score, randomized-window, raw-bit, normalized, density, and bin-power
  comparisons. The installed-wheel smoke now executes spectral entropy.
- Recorded the constant-input exception: ABFE returns `NaN` for zero spectral
  mass, while AntroPy returns `-0.0` from residual floating-point PSD mass.
- Kept spectral entropy in `frequency_domain` by design so callers compute one
  explicit `PSDResult`; no duplicate raw-signal wrapper with hidden spectral
  defaults was introduced.

### Distribution entropy

- Added normalized distribution entropy with explicit embedding order, delay,
  bin count, and normalization; equal-width observed-range bins include the
  maximum distance in the final bin.
- Added an independent manual histogram oracle, a hand-computed two-bin value,
  affine/z-score invariance checks, and pinned comparisons with DIHC commit
  `852a79d0178eb762aeaf81960820043365a059c3`. Natural-log ABFE and base-2
  DIHC normalization differ by at most `4.996e-15` in the 500-bin reference
  matrix; the protected comparison tolerance is `6e-15`.
- Degenerate identical-distance inputs return `NaN` instead of DIHC's `0.0`
  fallback, distinguishing undefined estimation from genuine zero entropy.

### Fuzzy entropy

- Added Chen-style fuzzy entropy with local template-mean removal,
  self-match exclusion, equal template populations at both dimensions,
  Chebyshev distance, and an explicit fuzzy exponent and SD-relative
  half-membership tolerance.
- Added an independent paper-definition oracle, a hand-computed membership
  ratio, affine/z-score invariance tests, and a pinned comparison with DIHC
  commit `852a79d0178eb762aeaf81960820043365a059c3`.
- Deliberately corrected DIHC's raw-coordinate template comparison, unequal
  template populations, constant-input division error, and zero fallback on
  complete similarity underflow. Undefined cases return `NaN`; a real zero
  remains valid for regular nonconstant input. On a 64-sample ramp, ABFE
  returns `0.0`; DIHC returns `-0.015497710010711062` because its two
  dimensions use unequal template populations.

### SVD entropy

- Added delayed-embedding SVD entropy with explicit order and delay, natural
  logarithms, optional normalization by `log(order)`, and explicit
  mean-centering to guarantee offset and z-score invariance.
- Added an independent Gram-matrix eigendecomposition oracle, an equal-singular
  value hand calculation, nonzero scaling checks, and pinned antropy 0.2.2
  comparisons. Rank-zero and rank-one embeddings return `NaN` under ABFE's
  entropy degeneracy policy.

### Permutation entropy

- Added Bandt-Pompe permutation entropy with explicit order, delay, optional
  normalization, natural logarithms, and deterministic stable tie handling.
- Inputs with fewer than two distinct ordinal patterns return `NaN` under
  ABFE's entropy degeneracy policy instead of reporting zero complexity.
- Added an independent ordinal-pattern oracle, hand-computed and tie cases,
  scaling invariance checks, and pinned antropy 0.2.2 comparisons. A protected
  exception records antropy's order-4/delay-2 periodic fast-path result
  (`1.4610149834435189`) versus the stable-sort oracle and ABFE result
  (`1.473881741077073`); antropy's positional epsilon changes the ordering of
  near-equal but distinct samples in that case.

### Approximate entropy

- Added self-match-inclusive approximate entropy with Chebyshev distance,
  natural logarithms, explicit embedding order, and explicit tolerance.
- `tolerance=None` uses `0.2 * std(signal, ddof=0)` and is protected by
  2x/5x/10x, negative-scale, and z-score invariance tests on periodic,
  white-noise, and chirp signals.
- Added an independent nested-loop oracle, a hand-computed four-sample value,
  and 21 pinned antropy 0.2.2 reference comparisons. The maximum absolute
  difference across periodic, white-noise, chirp, scaled, z-scored, and
  randomized inputs was `0.0`. Constant input returns `NaN` under ABFE's
  entropy degeneracy policy rather than antropy's zero fallback.
- Expanded the installed-wheel CI smoke test to execute a time-domain feature,
  Welch PSD, frequency-domain peak frequency, and sample-entropy profile from
  outside the source checkout.

### Audit remediation: validation ownership and public API

- `validate_signal` now rejects boolean, string, complex, mixed, and object
  inputs instead of silently coercing them into plausible-looking samples. It
  always returns an owned, read-only `float64` array, so validation never
  aliases caller-owned memory.
- The top-level API is intentionally limited to versioning and core PSD
  infrastructure. Individual scientific features remain in `time_domain`,
  `frequency_domain`, and `entropy`; this policy is documented and protected by
  tests.
- Added negative-amplitude scaling regressions, odd/even PSD window tests, and
  DC/Nyquist band-boundary checks.
- Increased the independent sample-entropy oracle tolerance from `2e-15` to
  `5e-15`, giving platform margin over the observed `1.9984e-15` maximum while
  retaining an exact external-authority comparison.
- Added GitHub Actions for Python 3.10 dependency-floor checks, current Python,
  linting, strict typing, wheel smoke tests, and pinned external numerical
  references. External network access is confined to the reference-validation
  job, which installs `sampen-profile` commit
  `263b72f7b906208241291c0579e8ec103fb52d79` and MNE 1.12.1.

### Audit remediation: frequency-resolution diagnostics

- **Before:** multitaper band-resolution warnings compared band width only to
  raw FFT-bin spacing. With a 200 Hz sample rate, 2 s window, 0.5 Hz bin
  spacing, 4 Hz multitaper bandwidth, and requested band `(9, 11)` Hz, the
  2 Hz band produced no warning even though it is narrower than the method's
  4 Hz effective bandwidth.
- **After:** Welch uses FFT-bin spacing and multitaper uses
  `max(bin_spacing, bandwidth)` as the warning threshold. The same `(9, 11)`
  Hz request now emits `FrequencyResolutionWarning`.
- Warning attribution now originates from the public feature call rather than
  a library-internal integration helper.
- Strict NumPy array typing was made consistent across source and tests, and
  test integration now uses SciPy's `trapezoid` to retain NumPy 1.24 support.

### Audit remediation: explicit PSD APIs and invariants

- Replaced the single mixed `PSDConfig` with `WelchPSDConfig` and
  `MultitaperPSDConfig`. Method-inappropriate parameters are now rejected by
  construction instead of silently ignored.
- Renamed `scaling="spectrum"` to `scaling="bin_power"`. The old name could be
  confused with SciPy's window-dependent spectrum normalization: for the
  default Hann window, SciPy's spectrum is exactly 1.5 times ABFE's
  density-times-bin-width quantity. `bin_power` states ABFE's units directly.
- Replaced ambiguous `frequency_resolution` provenance with explicit
  `bin_spacing` and `effective_bandwidth`. Welch records no single effective
  bandwidth; multitaper records its configured DPSS bandwidth.
- `PSDResult` now validates its frequency axis, nonnegative power, Nyquist
  limit, spacing, segment metadata, and configuration consistency. It always
  owns read-only `float64` copies of its arrays, including when constructed
  directly.
- Window and overlap seconds are explicitly documented and tested as converting
  to samples with Python's round-half-to-even rule.
- Multitaper now requests `floor(2*NW)` DPSS candidates before applying the
  optional 90% low-bias filter. On seeded white noise (`seed=123`, 4000
  samples, 200 Hz, 4 s windows, 2 Hz bandwidth), the default `low_bias=True`
  PSD changed by at most `9.20e-17` absolute (`8.84e-15` relative), while
  `low_bias=False` changed by up to `0.00131` absolute (`13.29%` relative)
  because it now includes the eighth, lower-concentration taper instead of
  hard-capping the candidate pool at seven. The candidate count is protected
  by a regression test.

### Initial time-domain implementation

- Added strict finite, numeric, one-dimensional input validation. Unlike the
  reference projects, ABFE does not flatten multidimensional input.
- Added canonical time-domain statistics and biomedical amplitude/shape
  features without computation-time rounding.
- Defined population variance and standard deviation with `ddof=0`.
- Defined skewness and Fisher excess kurtosis using SciPy's bias-corrected
  estimators; constant or insufficient input returns `NaN` rather than a false
  numeric zero.
- Defined exact-zero crossing behavior and protected it with hand-calculated
  regression cases.
- Added 2x, 5x, 10x, and z-score scaling-law tests.

### Initial frequency-domain implementation

- Added immutable, fully explicit Welch and DPSS multitaper PSD
  configurations. The achieved resolution and all effective window parameters
  are retained in each PSD result.
- Added peak, mean, median, and spectral-edge frequencies; spectral entropy;
  caller-defined absolute/relative band power; and explicitly named band-power
  ratios.
- Added `FrequencyResolutionWarning` when a requested band is narrower than
  the achieved PSD resolution.
- Band edges are included through linear interpolation before trapezoidal
  integration. No domain band presets or filtering paths were introduced.
- Absolute power follows the expected amplitude-squared scaling law; frequency
  locations, normalized spectral entropy, relative power, and ratios are
  protected by amplitude-invariance tests.

### Authoritative sample-entropy profile

- Reimplemented the Cumulative Histogram Method to match `sampen-profile`
  exactly while enforcing ABFE's strict one-dimensional input contract.
- Direct comparison on periodic, seeded white-noise, and chirp signals found a
  maximum absolute difference of `0.0` for both tolerance and entropy arrays.
- The same exact agreement was observed after 2x, 5x, 10x, and z-score
  transformations.
- Preserved three-decimal half-even distance quantization as an explicitly
  documented compatibility exception. Scaling can change profile point count;
  this is reference behavior, not presented as mathematical scale invariance.
- Excluded the legacy `TotalSampEn` and `AvgSampEn` profile summaries.
