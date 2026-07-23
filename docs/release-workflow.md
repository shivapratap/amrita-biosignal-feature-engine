# GitHub Release workflow design

ABFE v0.1.0 will initially be distributed through GitHub Releases only. PyPI
publication remains deferred until additional student testing is complete.

The proposed tag workflow is stored as
`.github/workflows/release.yml.disabled`. GitHub does not execute files with
that suffix. It must remain disabled until the user separately approves:

1. the final `0.1.0` version and release date;
2. the reviewed initial Git history and remote push;
3. activation of the workflow;
4. creation and push of annotated tag `v0.1.0`; and
5. publication of the corresponding GitHub Release.

Once activated, the workflow checks that the tag exactly equals `v` plus the
project version, builds wheel and source distribution from that tag, validates
their contents and metadata, writes SHA-256 hashes, and attaches only those
artifacts and the hash manifest to an immutable GitHub Release. It does not
publish to PyPI.

The workflow requests `contents: write` only for its release job. Ordinary CI
retains default read permissions and never publishes artifacts.
