# Contributing to ABFE

Numerical changes require all of the following:

1. A clear definition and primary/reference source where applicable.
2. Before-and-after numeric evidence in `CHANGES.md`.
3. A focused regression test that would fail under the previous behavior.
4. Tests for invalid and degenerate input.
5. Reference comparisons and documented tolerances for entropy features.

Feature computations must return full precision. Do not silently convert a
failure into `0.0`, fuse reporting-time rounding into computation, introduce
domain presets into the default import path, or add hidden filtering/windowing.

Before submitting a change, run:

```bash
pytest
ruff check .
mypy
```
