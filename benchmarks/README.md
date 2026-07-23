# ABFE performance baseline

The benchmark harness measures representative direct-feature, extractor, PSD,
entropy, fractal/complexity, mixed, and batch workloads using deterministic
synthetic signals. It has no dependency beyond ABFE's runtime dependencies.

Run the full baseline from an installed editable checkout:

```bash
python -m pip install -e .
python -m benchmarks.benchmark_baseline
```

For a fast harness and correctness check:

```bash
python -m benchmarks.benchmark_baseline --quick
```

Each workload runs in an isolated spawned process. Timing is the median wall
time per completed operation after a warm-up. Peak resident memory therefore
includes the Python interpreter, NumPy, SciPy, and ABFE, but avoids pollution
from earlier benchmark cases. Results include environment versions, fixed seed,
iteration counts, checksums, and raw timing bounds in JSON. The three slowest
workloads and the representative batch workload also receive cumulative-time
`cProfile` reports.

The baseline is observational. It deliberately has no machine-specific
performance thresholds and is not a scientific correctness substitute. Every
workload checks output structure and finiteness before reporting performance.

Quadratic entropy workloads are separated from approximately linear entropy
workloads so their scaling and memory behavior remain visible. The structured
sample-entropy profile is measured separately because it builds, quantizes,
sorts, and summarizes two pairwise-distance matrices. Performance changes
should compare the same case, environment, and dependency versions.

Lempel-Ziv complexity is measured independently at 512, 1,024, 2,048, and
4,096 samples. These cases expose scaling changes in its exhaustive-history
parser without imposing machine-specific pass/fail timing limits. The
same-machine before/after evidence for the suffix-automaton remediation is
recorded in `results/lz76-remediation.json`; exact phrase-count equality was a
precondition for every measured case.
