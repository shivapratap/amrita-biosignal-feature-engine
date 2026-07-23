# GitHub Release workflow design

ABFE releases are currently distributed through GitHub Releases only. PyPI
publication remains deferred until additional student testing and separate
approval.

The active workflow is stored as `.github/workflows/release.yml`. It has two
strictly separated entry paths:

1. A manually triggered dry run uses `contents: read`, reads the candidate
   version from `pyproject.toml`, builds the wheel and source distribution,
   validates their contents and metadata, installs each artifact into its own
   fresh environment, runs the installed-package smoke test from outside the
   checkout, and generates and verifies SHA-256 hashes. It can validate a
   development version and cannot upload artifacts, create a tag, or create a
   release.
2. A push of a `v*` tag runs the publishing job. That job checks that the tag
   exactly equals `v` plus a stable three-component project version, rebuilds
   and validates the distributions, fresh-installs and smoke-tests both
   artifacts, writes SHA-256 hashes, and attaches the distributions and hash
   manifest to the corresponding GitHub Release. Development, alpha, beta, and
   release-candidate versions cannot enter this publication path. It does not
   publish to PyPI.

The v0.1.0 manual dry run completed successfully on 2026-07-23, and GitHub
confirmed that the publishing job was skipped. The generalized v0.2.0
development-candidate dry run remains pending until the reviewed workflow is
pushed with separate approval. The repository-level permission remains
`contents: read`; only the tag-gated publishing job requests `contents: write`.
Creating or pushing a version tag always requires separate explicit approval.
