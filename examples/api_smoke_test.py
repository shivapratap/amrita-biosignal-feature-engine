"""Small self-checking example for the public ABFE API."""

from __future__ import annotations

import math

import numpy as np

from amrita_biosignal_feature_engine import (
    BandPowerRatioRequest,
    BandPowerRequest,
    ExtractorConfig,
    FeatureExtractor,
    LargestLyapunovRequest,
    WelchPSDConfig,
    __version__,
    compute_psd,
)
from amrita_biosignal_feature_engine import frequency_domain as frequency
from amrita_biosignal_feature_engine import time_domain as time_features
from amrita_biosignal_feature_engine.complexity import (
    detrended_fluctuation_analysis,
    fisher_information,
    higuchi_fractal_dimension,
    hjorth_mobility,
    hjorth_complexity,
    katz_fractal_dimension,
    largest_lyapunov_exponent,
    lempel_ziv_complexity,
    petrosian_fractal_dimension,
)
from amrita_biosignal_feature_engine.entropy import (
    approximate_entropy,
    distribution_entropy,
    fuzzy_entropy,
    permutation_entropy,
    sample_entropy_profile,
    svd_entropy,
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
    spectral_entropy = frequency.spectral_entropy(psd)
    power_8_12_hz = frequency.band_power(psd, band=(8.0, 12.0))
    approximate = approximate_entropy(signal)
    permutation = permutation_entropy(signal, normalize=True)
    fuzzy = fuzzy_entropy(signal)
    distribution = distribution_entropy(signal)
    svd = svd_entropy(signal)
    mobility = hjorth_mobility(signal)
    hjorth = hjorth_complexity(signal)
    lz_complexity = lempel_ziv_complexity(signal)
    fisher = fisher_information(signal)
    petrosian = petrosian_fractal_dimension(signal)
    katz = katz_fractal_dimension(signal)
    higuchi = higuchi_fractal_dimension(signal)
    dfa = detrended_fluctuation_analysis(signal)
    lyapunov = largest_lyapunov_exponent(
        signal,
        sampling_frequency=sampling_frequency,
        embedding_dimension=3,
        delay_samples=2,
        minimum_separation_samples=10,
        fit_start=0,
        fit_end=6,
    )
    profile = sample_entropy_profile(signal)

    extractor = FeatureExtractor(ExtractorConfig(sampling_frequency, psd_config))
    requests = (
        "root_mean_square",
        "approximate_entropy",
        "peak_frequency",
        "spectral_entropy",
        BandPowerRequest("power_8_12_hz", (8.0, 12.0)),
        BandPowerRatioRequest("power_8_12_over_20_30", (8.0, 12.0), (20.0, 30.0)),
        LargestLyapunovRequest("largest_lyapunov_s_inverse", 3, 2, 10, 0, 6),
    )
    result = extractor.extract(signal, features=requests)
    batch = extractor.extract_batch((signal, signal * 0.5), features=requests)

    direct_values = (
        rms,
        peak_hz,
        spectral_entropy,
        power_8_12_hz,
        approximate,
        permutation,
        fuzzy,
        distribution,
        svd,
        mobility,
        hjorth,
        lz_complexity,
        fisher,
        petrosian,
        katz,
        higuchi,
        dfa,
        lyapunov,
    )
    assert all(math.isfinite(value) for value in direct_values)
    assert abs(peak_hz - 10.0) <= psd.bin_spacing
    assert profile.point_count > 0
    assert len(DEFAULT_FEATURE_NAMES) == 34
    assert not result.failed_features
    assert len(batch.rows) == 2
    assert all(not row.failed_features for row in batch.rows)

    print(f"ABFE version: {__version__}")
    print(f"Registered scalar features: {len(DEFAULT_FEATURE_NAMES)}")
    print(f"Direct RMS: {rms:.6f}")
    print(f"Direct peak frequency: {peak_hz:.2f} Hz")
    print(f"Direct spectral entropy: {spectral_entropy:.6f}")
    print(f"Direct Hjorth complexity: {hjorth:.6f}")
    print("Extracted values:")
    for name, value in result.values.items():
        print(f"  {name}: {value:.6g}")
    print(
        "Largest Lyapunov estimates are parameter-sensitive; "
        "a positive value alone does not establish deterministic chaos."
    )
    print("ABFE API smoke test passed.")


if __name__ == "__main__":
    main()
