from __future__ import annotations

import os

import numpy as np
import pytest

from amrita_biosignal_feature_engine import MultitaperPSDConfig, compute_psd

if os.environ.get("ABFE_RUN_REFERENCE") != "1":
    pytest.skip("set ABFE_RUN_REFERENCE=1 to run external references", allow_module_level=True)

mne = pytest.importorskip("mne")

pytestmark = pytest.mark.reference


def test_default_low_bias_multitaper_matches_mne() -> None:
    from mne.time_frequency import psd_array_multitaper

    sampling_frequency = 200.0
    signal = np.random.default_rng(123).normal(size=4000)
    config = MultitaperPSDConfig(
        window_length=4.0,
        overlap=2.0,
        bandwidth=2.0,
        low_bias=True,
        multitaper_weighting="eigen",
        detrend="constant",
        scaling="density",
    )
    actual = compute_psd(signal, sampling_frequency, config)

    window_samples = 800
    step_samples = 400
    segments = np.array(
        [
            signal[start : start + window_samples]
            for start in range(0, signal.size - window_samples + 1, step_samples)
        ]
    )
    expected_values, expected_frequencies = psd_array_multitaper(
        segments,
        sfreq=sampling_frequency,
        fmin=0.0,
        fmax=sampling_frequency / 2.0,
        bandwidth=2.0,
        adaptive=False,
        low_bias=True,
        normalization="full",
        remove_dc=True,
        output="power",
        verbose=False,
    )
    np.testing.assert_array_equal(actual.frequencies, expected_frequencies)
    np.testing.assert_allclose(actual.values, expected_values.mean(axis=0), rtol=2e-12, atol=2e-14)
