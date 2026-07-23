"""Small self-checking example for the public ABFE API."""

from __future__ import annotations

import math

import numpy as np

from amrita_biosignal_feature_engine import (
    BandPowerRatioRequest,
    BandPowerRequest,
    ExtractorConfig,
    FeatureExtractor,
    WelchPSDConfig,
    __version__,
    compute_psd,
)
from amrita_biosignal_feature_engine import frequency_domain as frequency
from amrita_biosignal_feature_engine import time_domain as time_features
from amrita_biosignal_feature_engine.entropy import (
    approximate_entropy,
    permutation_entropy,
    sample_entropy_profile,
)
from amrita_biosignal_feature_engine.feature_registry import DEFAULT_FEATURE_NAMES


def main() -> None:
    """Exercise representative public APIs and fail if a result is invalid."""
    assert __version__ != "0+unknown", "Install ABFE before running this script"
    sampling_frequency = 200.0
    seconds = 2.0
    sample_times = np.arange(int(sampling_frequency * seconds)) / sampling_frequency
    signal = np.sin(2.0 * np.pi * 10.0 * sample_times) + 0.25 * np.sin(
        2.0 * np.pi * 25.0 * sample_times
    )

    rms = time_features.root_mean_square(signal)
    psd_config = WelchPSDConfig(window_length=2.0, overlap=1.0)
    psd = compute_psd(signal, sampling_frequency, psd_config)
    peak_hz = frequency.peak_frequency(psd)
    power_8_12_hz = frequency.band_power(psd, band=(8.0, 12.0))
    approximate = approximate_entropy(signal)
    permutation = permutation_entropy(signal, normalize=True)
    profile = sample_entropy_profile(signal)

    extractor = FeatureExtractor(ExtractorConfig(sampling_frequency, psd_config))
    requests = (
        "root_mean_square",
        "approximate_entropy",
        "peak_frequency",
        "spectral_entropy",
        BandPowerRequest("power_8_12_hz", (8.0, 12.0)),
        BandPowerRatioRequest("power_8_12_over_20_30", (8.0, 12.0), (20.0, 30.0)),
    )
    result = extractor.extract(signal, features=requests)
    batch = extractor.extract_batch((signal, signal * 0.5), features=requests)

    direct_values = (rms, peak_hz, power_8_12_hz, approximate, permutation)
    assert all(math.isfinite(value) for value in direct_values)
    assert abs(peak_hz - 10.0) <= psd.bin_spacing
    assert profile.point_count > 0
    assert len(DEFAULT_FEATURE_NAMES) == 26
    assert not result.failed_features
    assert len(batch.rows) == 2
    assert all(not row.failed_features for row in batch.rows)

    print(f"ABFE version: {__version__}")
    print(f"Registered scalar features: {len(DEFAULT_FEATURE_NAMES)}")
    print(f"Direct RMS: {rms:.6f}")
    print(f"Direct peak frequency: {peak_hz:.2f} Hz")
    print("Extracted values:")
    for name, value in result.values.items():
        print(f"  {name}: {value:.6g}")
    print("ABFE API smoke test passed.")


if __name__ == "__main__":
    main()
