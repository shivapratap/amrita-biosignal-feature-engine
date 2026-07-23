# ABFE v0.1.0 release-readiness design

This document defines the gates for the first public release of Amrita
BioSignal Feature Engine (ABFE). Passing a gate permits work to proceed to the
next gate; it does not authorize publishing, pushing, or tagging.

## Release principles

- Release only reviewed source that is reproducible from a clean checkout.
- Keep numerical behavior frozen unless a release audit finds a demonstrated
  defect. Any numerical correction follows the evidence and regression-test
  discipline in `CONTRIBUTING.md`.
- Build artifacts from the exact commit tagged `v0.1.0`.
- Never publish from a floating branch or retag an existing version.
- Keep external numerical authorities pinned in reference-validation CI; they
  are validation dependencies, not runtime dependencies.
- Require explicit approval before creating a remote repository, pushing,
  publishing an artifact, or creating the `v0.1.0` tag.

## Gate 1: repository and metadata preparation

1. Resolve the public repository owner and canonical GitHub URL.
2. Resolve the author/maintainer metadata and public contact information.
3. Confirm whether v0.1.0 will be distributed through GitHub Releases only or
   through both GitHub Releases and PyPI.
4. Check name availability on every selected distribution service before
   changing the version from `0.1.0.dev0`.
5. Complete `pyproject.toml` metadata: URLs, maintainers, supported Python
   classifiers, typed-package classifier, and release-appropriate development
   status. Keep the MIT license declaration consistent with `LICENSE`.
6. Confirm ignored build/cache artifacts are untracked and that no stale Git
   lock prevents creation of the initial history.
7. Decide whether independent audit reports belong in the public repository.
   If retained, move them under a clearly named documentation directory; do
   not silently ship workspace-only reports in the source distribution.

Exit evidence: metadata inspection, name-availability evidence, clean artifact
inventory, and a written distribution decision.

## Gate 2: user and research documentation

1. Update the README so it describes all implemented time-domain,
   frequency-domain, entropy, registry, extraction, diagnostic, and provenance
   capabilities.
2. Add installation instructions, a minimal extraction example, feature and
   namespace tables, input/degeneracy behavior, PSD reproducibility guidance,
   and links to detailed validation documents.
3. Run every README code example as a test or smoke script.
4. Add a public API reference covering parameters, return models, exceptions,
   warnings, and the top-level versus submodule export policy.
5. Add `CITATION.cff` with the final authorship, repository URL, version, and
   release date. Do not invent a DOI; add one later only if issued.
6. Add a concise release history entry for v0.1.0 without claiming checks that
   have not actually run.

Exit evidence: documentation review plus executable examples passing outside
the source checkout.

## Gate 3: reproducible performance baseline

1. Add a small `benchmarks/` suite for representative window sizes and for
   signal-only, PSD-only, entropy, mixed extraction, and batch extraction.
2. Use deterministic inputs and fixed seeds. Report timings without encoding
   machine-specific thresholds as correctness tests.
3. Add a correctness guard to each benchmark so a fast but invalid result
   cannot be reported as a performance success.
4. Record environment and dependency versions with benchmark output.

Benchmarks are observational for v0.1.0. They must not trigger unprofiled API
or algorithm changes during release preparation.

Exit evidence: reproducible benchmark command, recorded environment, and a
baseline report.

The implemented harness is `benchmarks.benchmark_baseline`. It records timing,
isolated-process peak resident memory, deterministic correctness checks, JSON
environment metadata, and cumulative profiles for the three slowest cases plus
the representative batch case. Results are evidence for classification, not
cross-machine release thresholds.

## Gate 4: CI and artifact hardening

1. Retain offline testing at the declared NumPy/SciPy floor and current Python
   leg, strict mypy for Python 3.10 and 3.13 targets, Ruff, and cache rejection.
2. Retain the dedicated pinned external-reference job and confirm it runs in a
   clean environment rather than through local editable reference checkouts.
3. Build both wheel and source distribution in CI.
4. Validate metadata and artifact contents, including `LICENSE`, README,
   `py.typed`, source modules, and exclusion of caches and local audit/build
   debris.
5. Install and smoke-test the wheel and source distribution independently from
   outside the checkout. Exercise direct functions, PSD computation,
   `FeatureExtractor.extract`, and `extract_batch`.
6. Assert that installed `__version__` matches the project version.
7. Add a tag-oriented release workflow only after the publishing destination
   and trusted-publishing policy are approved. The workflow must build from the
   tag and must not publish on ordinary pushes.

The GitHub-only workflow is active as `.github/workflows/release.yml`. Its
manual dry-run path has read-only repository permission and cannot publish; its
separate publishing job runs only for a pushed `v*` tag. The v0.1.0 dry run
successfully built and validated both distributions and verified their hashes
while the publishing job remained skipped.

Exit evidence: green CI, artifact manifests, metadata validation, and clean
wheel/source-distribution smoke tests.

## Gate 5: independent release-candidate audit

Give the complete candidate, still versioned `0.1.0.dev0`, to an independent
read-only reviewer. The audit must cover:

- scientific definitions and documented claims;
- registry/dispatch completeness and immutable result contracts;
- error, undefined-value, warning, diagnostic, and provenance behavior;
- public exports and typed-package behavior;
- dependency floors and supported Python versions;
- pinned reference validation and CI reproducibility;
- README/API example execution;
- wheel and source-distribution contents and clean-install behavior;
- license, citation, metadata, and release workflow safety;
- absence of hidden filtering/windowing, domain presets, false-zero fallbacks,
  fused rounding, generated caches, and unapproved artifacts.

All blocking findings must be remediated and rechecked before the version is
frozen. Non-blocking findings must be documented with an explicit disposition.

Exit evidence: independent audit report with a release recommendation and no
unresolved blocking findings.

## Gate 6: version freeze and publication

This gate requires separate user approval.

Preparation status on 2026-07-23: the candidate is frozen at `0.1.0`,
release-date metadata and the changelog are finalized, the reviewed Git history
is pushed to the public repository, remote CI is green, and the active release
workflow has passed its manual dry run. No tag or GitHub Release has been
created.

1. Change `0.1.0.dev0` to `0.1.0`, update release-date metadata, and freeze the
   changelog.
2. Run all quality, reference, documentation, benchmark, build, metadata,
   artifact-content, and clean-install gates again on the release commit.
3. Establish reviewed Git history in logical commits; do not combine all
   implementation and release work into an opaque initial change.
4. Push the exact approved release commit and require green remote CI.
5. Create annotated tag `v0.1.0` only on that commit.
6. Build/publish artifacts from the tag through the approved release workflow.
7. Verify the public artifact by installing it from the selected distribution
   service in a fresh environment and rerunning the installed-package smoke
   test.
8. Record artifact hashes and the public release URL.

## Current status and remaining publication steps

- The public repository is
  `shivapratap/amrita-biosignal-feature-engine`; local `main` tracks the remote
  `main` branch.
- Shivapratap Gopakumar is the author and maintainer. Distribution is through
  GitHub Releases only for v0.1.0; PyPI is deferred until additional student
  testing has been completed.
- `pyproject.toml` and citation metadata are frozen at `0.1.0`, dated
  2026-07-23. Publication remains pending.
- Audit reports are workspace evidence and are excluded from the public
  repository by policy.
- The benchmark baseline, citation file, artifact validator, independent
  wheel/source-distribution smoke tests, active release workflow, and manual
  release rehearsal are complete.
- The package-wide independent execution-based Gate 5 audit completed with
  `PASS WITH MINOR FOLLOW-UPS`. Its only source blocker, RC-001, was closed by
  parameterizing the benchmark signal return type as `NDArray[np.float64]`.
  Gate 5 is closed.
- The remaining publication work is to create and push the annotated `v0.1.0`
  tag, monitor the tag-gated publication job, and verify the downloadable
  artifacts and checksums in a fresh installation. These actions require
  separate explicit approval.
