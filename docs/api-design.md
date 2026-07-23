# Public API design

ABFE keeps its top-level namespace deliberately small. The top level exports
package versioning and core configuration/computation types:

- `__version__`
- `WelchPSDConfig`
- `MultitaperPSDConfig`
- `PSDResult`
- `compute_psd`

Step 6 promotes its orchestration models to the top level:

- `ExtractorConfig`
- `BandPowerRequest`
- `BandPowerRatioRequest`
- `ExtractionProvenance`
- `ExtractionResult`
- `BatchExtractionResult`

`FeatureExtractor` is also top-level orchestration API. Registry inspection
remains in `feature_registry`; structured diagnostic types remain in
`diagnostics`.

Individual scientific feature functions remain organized in their domain
submodules:

```python
from amrita_biosignal_feature_engine.time_domain import root_mean_square
from amrita_biosignal_feature_engine.frequency_domain import band_power
from amrita_biosignal_feature_engine.entropy import sample_entropy_profile
```

They are not flattened into the package top level. This avoids name collisions,
keeps provenance clear, and prevents a growing feature catalog from turning the
root namespace into an unstable registry.

The scalar registry is an immutable mapping of frozen `FeatureSpec` objects.
It intentionally excludes `sample_entropy_profile`, whose structured array
output should not be collapsed into an invented scalar. Caller-defined band
power and ratio outputs use frozen request objects with explicit output names
and band direction; ABFE ships no default physiological band presets.

`FeatureExtractor.extract` validates one pre-windowed signal and returns an
immutable `ExtractionResult`. Signal-only requests do not compute a PSD;
frequency and band requests share exactly one `PSDResult`. Structural request
or single-window input errors raise before computation. Numerical undefined
results and per-feature exceptions become distinct diagnostics with `NaN`
values. `extract_batch` retains structurally invalid rows as all-`NaN` results
with `INVALID_WINDOW` diagnostics so input alignment is never lost.
