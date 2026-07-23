# Power spectral density

ABFE uses method-specific immutable configurations. Parameters that do not
belong to the selected method cannot be supplied and are never silently
ignored.

## Welch

```python
from amrita_biosignal_feature_engine import WelchPSDConfig, compute_psd

config = WelchPSDConfig(
    window_length=2.0,
    overlap=1.0,
    window="hann",
    detrend="constant",
    scaling="density",
)
result = compute_psd(window, sampling_frequency=200.0, config=config)
```

## Multitaper

```python
from amrita_biosignal_feature_engine import MultitaperPSDConfig, compute_psd

config = MultitaperPSDConfig(
    window_length=2.0,
    overlap=1.0,
    bandwidth=4.0,
    low_bias=True,
    multitaper_weighting="eigen",
    detrend="constant",
    scaling="density",
)
result = compute_psd(window, sampling_frequency=200.0, config=config)
```

For both methods, seconds are multiplied by sampling frequency and converted
to sample counts with Python's round-half-to-even rule. For example, at 100 Hz,
0.025 seconds becomes 2 samples, 0.035 becomes 4, and 0.045 remains 4.

## Scaling and resolution

- `density` has units of signal-unit squared per Hz.
- `bin_power` is density multiplied by FFT-bin spacing and has units of
  signal-unit squared per bin.

`bin_power` is not SciPy's `scaling="spectrum"`. For a Hann window, SciPy's
spectrum normalization is 1.5 times ABFE's density-times-bin-width quantity.
The distinct name prevents those quantities from being mistaken for one
another.

Every result records:

- `bin_spacing`: the uniform FFT frequency-grid spacing;
- `effective_bandwidth`: the configured DPSS bandwidth for multitaper, or
  `None` for Welch because ABFE does not currently claim one universal
  window-dependent effective bandwidth.

Band-resolution warnings use bin spacing for Welch and the larger of bin
spacing and effective bandwidth for multitaper.
