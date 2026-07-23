# v0.1.0.dev0 performance baseline

This baseline measures the installed ABFE wheel before v0.1.0. It is
observational evidence, not a cross-machine performance guarantee.

## Environment and method

- Date: 2026-07-23 (UTC timestamp is retained in the JSON result)
- Platform: macOS 26.5.2, arm64, 11 logical CPUs
- Python: 3.13.9
- NumPy: 2.3.5
- SciPy: 1.16.3
- ABFE wheel: 0.1.0.dev0
- Fixed signal seed: 20260723
- Repeats: five; reported timing is the median wall time per operation after a
  warm-up
- Memory: isolated-process peak resident set, including the interpreter,
  NumPy, SciPy, and ABFE

Input signals are prepared outside the measured operation. Every case checks
determinism, output structure, and finite checksums. Full raw evidence is in
`benchmarks/results/baseline.json`; selected cumulative profiles are in
`benchmarks/results/profiles.md`.

## Results

| Workload | Window/batch | Median | Peak RSS |
|---|---:|---:|---:|
| Direct time features | 256 | 0.028 ms | 95.0 MiB |
| Direct time features | 1,024 | 0.033 ms | 94.2 MiB |
| Direct time features | 4,096 | 0.052 ms | 94.2 MiB |
| Signal-only extractor | 256 | 0.079 ms | 94.7 MiB |
| Signal-only extractor | 1,024 | 0.085 ms | 95.0 MiB |
| Welch PSD | 1,024 | 0.264 ms | 95.1 MiB |
| Welch PSD | 4,096 | 0.422 ms | 95.0 MiB |
| Multitaper PSD | 1,024 | 0.434 ms | 95.2 MiB |
| Multitaper PSD | 4,096 | 0.518 ms | 97.2 MiB |
| Permutation + SVD entropy | 256 | 0.110 ms | 94.4 MiB |
| Permutation + SVD entropy | 1,024 | 0.426 ms | 94.9 MiB |
| Approximate + fuzzy + distribution entropy | 256 | 0.755 ms | 99.8 MiB |
| Approximate + fuzzy + distribution entropy | 1,024 | 14.805 ms | 209.9 MiB |
| Sample-entropy profile | 256 | 17.039 ms | 102.2 MiB |
| Sample-entropy profile | 1,024 | 155.027 ms | 246.8 MiB |
| All 26 registered scalar features | 256 | 1.652 ms | 101.8 MiB |
| All 26 registered scalar features | 1,024 | 15.898 ms | 216.7 MiB |
| Eight signal features in batch | 16 x 256 | 1.242 ms | 94.6 MiB |

Small differences between neighboring sub-millisecond cases are not treated as
meaningful on a general-purpose operating system. Comparisons should use the
raw repeat bounds and the same environment.

## Profiling findings and disposition

### No v0.1.0 release blocker

All representative workloads completed with correct, deterministic outputs.
Direct features and explicit PSD computations remain sub-millisecond in this
environment. A full 26-feature extraction completed in about 1.8 ms at 256
samples and 15.5 ms at 1,024 samples. The 16-window signal-feature batch was
about 0.24 ms per window.

### Completed safe pre-release cleanup

`FeatureExtractor.extract` calls `importlib.metadata.version` for every result.
The batch profile shows 16 metadata reads for 16 rows, including distribution
metadata parsing and filesystem access. The installed-wheel signal extractor
is correspondingly dominated by fixed overhead at small window sizes.

The installed package version is now cached once per process. A focused test
proves repeated single and batch extraction makes one metadata query while
preserving the exact provenance version. The complete pre-change evidence is
retained in `baseline-before-version-cache.json` and
`profiles-before-version-cache.md`.

| Workload | Before | After | Improvement |
|---|---:|---:|---:|
| Signal-only extractor, 256 samples | 0.266 ms | 0.079 ms | 70.3% |
| Signal-only extractor, 1,024 samples | 0.243 ms | 0.085 ms | 65.0% |
| Batch, 16 x 256 samples | 3.781 ms | 1.242 ms | 67.2% |

The mixed 26-feature workloads are dominated by entropy arithmetic, so their
small run-to-run differences are not attributed to this fixed-overhead change.
No scientific computation, feature ordering, diagnostics, or public API was
changed.

### v0.2.0 optimization candidates

The approximate/fuzzy/distribution group increased about 18 times when window
length increased fourfold from 256 to 1,024, consistent with pairwise-distance
work and storage. Its 1,024-sample peak process RSS was about 200 MiB versus a
roughly 95 MiB process baseline. Profiles locate the cost in Chebyshev `cdist`
and `pdist`, fuzzy similarity evaluation, and the distribution histogram.

The authoritative-compatible sample-entropy profile was the slowest workload:
about 16.7 ms at 256 samples and 152.6 ms at 1,024 samples. Its 1,024-sample
peak process RSS was about 240 MiB. Profiling locates most time in sorting each
distance-matrix row and applying `searchsorted`; construction, uniqueness, and
storage of two quadratic distance matrices also contribute.

Recommended v0.2.0 investigations:

1. Explore chunked or streaming distance evaluation to reduce peak memory for
   approximate, fuzzy, and distribution entropy.
2. Explore an exactly reference-compatible cumulative-histogram strategy that
   reduces repeated sorting/search overhead in the sample-entropy profile.
3. Consider shared intermediates only where definitions, template populations,
   centering, quantization, and tolerances remain exactly equivalent.
4. Evaluate batch vectorization, DPSS caching, Numba, or parallelism only on
   representative user workloads; current PSD timings do not justify adding
   complexity before v0.1.0.

Any entropy optimization requires before/after numerical evidence, the complete
pinned reference suite, invariance and degeneracy tests, and documented
tolerances. Changes that alter floating-point summation order or memory layout
must not be described as behavior-preserving without measurement.

## Release conclusion

The current performance is suitable for a first alpha release on the measured
workloads. v0.1.0 retains the measured package-version caching improvement and
defers scientific algorithm redesign to the v0.2.0 performance phase.
