# Entropy validation

## Sample-entropy profile

ABFE's sample-entropy profile is validated against `sampen-profile`, the
behavioral authority for the Cumulative Histogram Method implementation.

The compatibility contract includes:

- `N - order` templates at both dimensions `order` and `order + 1`;
- Chebyshev distance;
- NumPy half-to-even rounding to three decimal places;
- a tolerance axis formed from the sorted union of both distance sets;
- `log((b + 1e-12) / (a + 1e-12))`;
- retention only where both `a` and `b` exceed `1e-12`.

Validation has three independent layers:

1. Direct comparison with `sampen-profile` on periodic, seeded white-noise,
   and chirp signals. Both tolerance and entropy arrays match exactly.
2. A test-only unordered-pair brute-force oracle, structurally independent of
   the package's matrix/CDF implementation. Its comparison tolerance is
   `5e-15`, providing platform margin over the measured `1.9984e-15` maximum
   while accounting for the different floating-point normalization order.
3. Hand-computed constant and zero-match examples, plus a protected
   half-to-even rounding boundary.

The direct comparison also covers 2x, 5x, 10x, and z-scored versions of all
three synthetic signals.

The normal offline suite runs the independent oracle. A separately marked
`reference` suite imports a pinned `sampen-profile` commit and runs in the
network-enabled GitHub Actions reference-validation job, making the direct
authority comparison reproducible without adding it as a runtime dependency.

### Compatibility exception: scale invariance

The fixed three-decimal distance quantization is scale-coupled. Scaling a
signal can therefore change the number of distinct tolerance points and the
profile values. ABFE preserves this behavior because removing it would violate
the explicit requirement to match `sampen-profile`. Tests require every scaled
result to continue matching the authority; they do not falsely assert strict
invariance for this one compatibility algorithm.

ABFE exposes only `r_values` and `se_profile`. Legacy profile sums and averages
are intentionally excluded.

## Approximate entropy

ABFE implements the standard self-match-inclusive Pincus approximate entropy:

- Chebyshev distance;
- `N - m + 1` templates at dimension `m`;
- `N - m` templates at dimension `m + 1`;
- self-matches included;
- each match count normalized by that dimension's template count;
- mean natural logarithm of the normalized counts;
- `ApEn = phi(m) - phi(m + 1)`.

The default tolerance is `0.2 * std(x, ddof=0)`. Offline validation uses a
structurally independent nested-loop oracle and a hand-computed four-sample
case. The pinned external-reference job compares periodic, white-noise, chirp,
scaled, z-scored, and randomized signals against antropy 0.2.2 for orders 2
and 3. Order 1 is validated by the independent oracle and hand calculation
because antropy rejects orders below 2.

Constant input returns `NaN` under ABFE's uniform entropy degeneracy policy,
which intentionally differs from antropy's numeric-zero fallback for that
case. Nonconstant reference cases must agree within `2e-15` absolute tolerance.

## Permutation entropy

ABFE implements Bandt-Pompe permutation entropy as the Shannon entropy of
observed ordinal-pattern probabilities, using natural logarithms. Delayed
embeddings have `N - (order - 1) * delay` patterns. Ties use stable
left-to-right index order, making their treatment deterministic. Normalization
divides by `log(order!)`.

Offline validation includes an independent dictionary-counting oracle, a
hand-computed two-pattern example, explicit tie behavior, orders 2 through 4,
delays 1 and 2, and positive/negative-scale and z-score invariance checks. The
pinned external-reference job converts antropy 0.2.2's unnormalized bit result
to nats and directly compares normalized results across periodic, white-noise,
chirp, scaled, z-scored, and randomized signals.

One pinned exception is recorded rather than copied: for the periodic signal
at order 4 and delay 2, antropy's comparison-based fast path adds positional
epsilon jitter that reorders distinct samples separated by less than that
jitter. It returns `1.4610149834435189` nats, while stable ordinal sorting and
ABFE's independent oracle both return `1.473881741077073`. A reference
regression records both values. Other order-4 inputs remain directly compared,
and order 5 exercises antropy's stable-argsort general path.

Inputs producing fewer than two distinct ordinal patterns return `NaN` under
ABFE's entropy degeneracy policy. This intentionally differs from antropy's
zero for constant and strictly monotonic inputs.

## SVD entropy

ABFE first mean-centers the signal, constructs a delayed embedding, normalizes
its singular values by their sum, and computes their Shannon entropy using
natural logarithms. Optional normalization divides by `log(order)`. This
explicit centering makes the result invariant to additive offsets, nonzero
amplitude scaling (including sign reversal), and z-scoring, as required by the
ABFE entropy contract. Antropy does not center internally, so reference tests
apply Antropy to the same explicitly centered signal.

Offline validation uses a separately constructed embedding and eigenvalues of
its Gram matrix as an independent oracle, plus a hand-computed identity-matrix
case. The pinned external-reference job converts antropy 0.2.2's bit result to
nats and compares periodic, white-noise, chirp, scaled, negative-scaled,
offset, z-scored, and randomized signals. Numerical-rank-zero or rank-one embeddings return `NaN`
under ABFE's entropy degeneracy policy; constant input is a documented
exception to antropy's finite near-zero result.

## Fuzzy entropy

ABFE follows Chen et al. (2007): each delayed template is locally
mean-centered before Chebyshev distance is calculated, self-matches are
excluded, and the first `N - order * delay` templates are used for both
dimensions. Similarity is `exp(-log(2) * (distance / tolerance) ** exponent)`
with `tolerance = tolerance_factor * std(signal, ddof=0)`. The default
`tolerance_factor=0.2` and `exponent=2` make membership one half at the
tolerance boundary and guarantee nonzero affine-scale and z-score invariance.

The independent offline oracle uses nested Python loops and direct paper
equations. A hand-computed alternating four-sample case protects the
membership ratio. The pinned external suite cross-checks DIHC commit
`852a79d0178eb762aeaf81960820043365a059c3`. DIHC omits local template-mean
removal and uses unequal template populations, so ABFE records rather than
copies those numerical differences. On a 64-sample regular ramp, ABFE returns
the paper-consistent `0.0`, while DIHC returns the impossible negative entropy
`-0.015497710010711062`. ABFE also changes DIHC's constant-input division
error to `NaN` and changes similarity underflow from a misleading zero to
`NaN`.

## Distribution entropy

ABFE forms delayed embeddings, calculates all unordered off-diagonal
Chebyshev distances, and estimates their empirical distribution with
equal-width bins spanning the observed distance range. The maximum is
explicitly included in the final bin. Shannon entropy uses natural logarithms
and is normalized by `log(n_bins)` by default, making the normalized result
independent of logarithm base and bounded by `[0, 1]`.

The offline suite uses a manual nested-loop distance and bin-assignment oracle,
plus a hand-computed two-bin distribution. Affine, negative-scale, and z-score
invariance are tested on periodic, white-noise, and chirp signals. The pinned
external suite compares delay-1 behavior against DIHC commit
`852a79d0178eb762aeaf81960820043365a059c3`; ABFE adds explicit delay support.
DIHC evaluates the normalized expression in base 2 while ABFE uses natural
logs. The measured maximum rounding difference at 500 bins is
`4.996e-15`, so the pinned comparison uses `6e-15` absolute tolerance.
When all pairwise distances are identical, ABFE returns `NaN` rather than
DIHC's misleading `0.0` degeneracy fallback.

## Spectral entropy

Spectral entropy remains in `frequency_domain` because it consumes ABFE's
explicit, provenance-carrying `PSDResult`; ABFE does not provide a second
raw-signal wrapper with hidden PSD defaults. It normalizes PSD-bin values into
probabilities and computes Shannon entropy in bits, optionally dividing by
`log2(number_of_bins)`.

The offline hand calculation protects exact bin probabilities. The pinned
AntroPy 0.2.2 suite aligns Welch's Hann window, constant detrending, 50%
overlap, density scaling, mean averaging, and FFT length. It compares periodic,
white-noise, chirp, 2x/5x/10x, negative, offset, z-scored, and randomized
signals in raw and normalized forms. Both ABFE `density` and `bin_power`
representations are tested because their constant bin-width factor must cancel
during probability normalization. For constant detrended input, ABFE detects
zero spectral mass and returns `NaN`; AntroPy returns `-0.0` from residual
floating-point PSD mass. This is a pinned degeneracy-policy exception rather
than a value ABFE copies.
