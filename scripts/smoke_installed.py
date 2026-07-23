"""Exercise the installed ABFE distribution outside a source checkout."""

from __future__ import annotations

import argparse

import numpy as np

import amrita_biosignal_feature_engine as abfe
from amrita_biosignal_feature_engine.complexity import (
    hjorth_complexity,
    hjorth_mobility,
    katz_fractal_dimension,
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
from amrita_biosignal_feature_engine.frequency_domain import (
    peak_frequency,
    spectral_entropy,
)
from amrita_biosignal_feature_engine.time_domain import root_mean_square


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    return parser.parse_args()


def main() -> None:
    """Run representative direct, PSD, entropy, single, and batch operations."""
    arguments = _arguments()
    if abfe.__version__ != arguments.expected_version:
        raise AssertionError(
            f"installed version {abfe.__version__!r} != {arguments.expected_version!r}"
        )
    sampling_frequency = 100.0
    time = np.arange(400, dtype=np.float64) / sampling_frequency
    signal = np.sin(2.0 * np.pi * 10.0 * time) + 0.5 * np.sin(
        2.0 * np.pi * 25.0 * time
    )
    if not root_mean_square(signal) > 0.0:
        raise AssertionError("time-domain smoke calculation failed")

    config = abfe.WelchPSDConfig(window_length=2.0, overlap=1.0)
    psd = abfe.compute_psd(signal, sampling_frequency, config)
    if peak_frequency(psd) != 10.0 or not np.isfinite(spectral_entropy(psd)):
        raise AssertionError("PSD/frequency smoke calculation failed")

    short_signal = signal[:80]
    entropy_values = (
        approximate_entropy(short_signal, order=2),
        distribution_entropy(short_signal, order=2),
        fuzzy_entropy(short_signal, order=2),
        permutation_entropy(short_signal, order=3),
        svd_entropy(short_signal, order=3),
    )
    if not all(np.isfinite(value) for value in entropy_values):
        raise AssertionError("scalar entropy smoke calculation failed")
    if sample_entropy_profile(short_signal, order=2).point_count <= 0:
        raise AssertionError("sample-entropy profile smoke calculation failed")
    complexity_values = (
        hjorth_mobility(short_signal),
        hjorth_complexity(short_signal),
        petrosian_fractal_dimension(short_signal),
        katz_fractal_dimension(short_signal),
        lempel_ziv_complexity(short_signal),
    )
    if not all(np.isfinite(value) for value in complexity_values):
        raise AssertionError("complexity smoke calculation failed")

    extractor = abfe.FeatureExtractor(abfe.ExtractorConfig(sampling_frequency, config))
    extracted = extractor.extract(
        signal,
        features=("root_mean_square", "approximate_entropy", "spectral_entropy"),
    )
    batch = extractor.extract_batch(
        (signal, signal),
        features=("mean", "peak_frequency"),
    )
    if extracted.failed_features or any(row.failed_features for row in batch.rows):
        raise AssertionError("extractor smoke calculation failed")
    if len(batch.rows) != 2 or batch.feature_names != ("mean", "peak_frequency"):
        raise AssertionError("batch extraction shape/schema failed")
    if extracted.provenance.package_version != arguments.expected_version:
        raise AssertionError("extraction provenance version does not match installed package")
    print(f"ABFE {abfe.__version__} installed-package smoke test passed")


if __name__ == "__main__":
    main()
