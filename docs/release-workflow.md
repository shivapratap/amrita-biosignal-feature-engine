# GitHub Release workflow design

ABFE v0.1.0 will initially be distributed through GitHub Releases only. PyPI
publication remains deferred until additional student testing is complete.

The active workflow is stored as `.github/workflows/release.yml`. It has two
strictly separated entry paths:

1. A manually triggered dry run uses `contents: read`, requires the frozen
   package version to be `0.1.0`, builds the wheel and source distribution,
   validates their contents and metadata, and generates and verifies SHA-256
   hashes. It does not upload artifacts, create a tag, or create a release.
2. A push of a `v*` tag runs the publishing job. That job checks that the tag
   exactly equals `v` plus the project version, rebuilds and validates the
   distributions, writes SHA-256 hashes, and attaches the distributions and
   hash manifest to the corresponding GitHub Release. It does not publish to
   PyPI.

The manual dry run completed successfully on 2026-07-23, and GitHub confirmed
that the publishing job was skipped. The repository-level permission remains
`contents: read`; only the tag-gated publishing job requests `contents: write`.
Creating or pushing `v0.1.0` still requires separate explicit approval.
