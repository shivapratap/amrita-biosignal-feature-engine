"""Dependency-free timing, peak-memory, and profiling baseline for ABFE."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import multiprocessing as mp
import os
import platform
import pstats
import resource
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray

import amrita_biosignal_feature_engine as abfe
from amrita_biosignal_feature_engine.complexity import lempel_ziv_complexity
from amrita_biosignal_feature_engine.entropy import (
    approximate_entropy,
    distribution_entropy,
    fuzzy_entropy,
    permutation_entropy,
    sample_entropy_profile,
    svd_entropy,
)
from amrita_biosignal_feature_engine.feature_registry import DEFAULT_FEATURE_NAMES
from amrita_biosignal_feature_engine.frequency_domain import peak_frequency
from amrita_biosignal_feature_engine.time_domain import (
    integrated_absolute_value,
    root_mean_square,
    slope_sign_change_count,
    waveform_length,
    zero_crossing_count,
)

SAMPLING_FREQUENCY = 256.0
BENCHMARK_SEED = 20260723
SIGNAL_FEATURES = (
    "minimum",
    "maximum",
    "mean",
    "standard_deviation",
    "root_mean_square",
    "waveform_length",
    "zero_crossing_count",
    "slope_sign_change_count",
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One named deterministic workload."""

    name: str
    category: str
    size: int
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Serializable measurements for one workload."""

    name: str
    category: str
    size: int
    batch_size: int
    iterations_per_repeat: int
    repeat_count: int
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    peak_rss_mib: float
    checksum: float


CASES = (
    BenchmarkCase("time_direct_n256", "time_direct", 256),
    BenchmarkCase("time_direct_n1024", "time_direct", 1024),
    BenchmarkCase("time_direct_n4096", "time_direct", 4096),
    BenchmarkCase("signal_extractor_n256", "signal_extractor", 256),
    BenchmarkCase("signal_extractor_n1024", "signal_extractor", 1024),
    BenchmarkCase("welch_psd_n1024", "welch_psd", 1024),
    BenchmarkCase("welch_psd_n4096", "welch_psd", 4096),
    BenchmarkCase("multitaper_psd_n1024", "multitaper_psd", 1024),
    BenchmarkCase("multitaper_psd_n4096", "multitaper_psd", 4096),
    BenchmarkCase("entropy_linear_n256", "entropy_linear", 256),
    BenchmarkCase("entropy_linear_n1024", "entropy_linear", 1024),
    BenchmarkCase("lempel_ziv_n512", "lempel_ziv", 512),
    BenchmarkCase("lempel_ziv_n1024", "lempel_ziv", 1024),
    BenchmarkCase("lempel_ziv_n2048", "lempel_ziv", 2048),
    BenchmarkCase("lempel_ziv_n4096", "lempel_ziv", 4096),
    BenchmarkCase("entropy_quadratic_n256", "entropy_quadratic", 256),
    BenchmarkCase("entropy_quadratic_n1024", "entropy_quadratic", 1024),
    BenchmarkCase("sample_entropy_profile_n256", "sample_entropy_profile", 256),
    BenchmarkCase("sample_entropy_profile_n1024", "sample_entropy_profile", 1024),
    BenchmarkCase("mixed_extractor_n256", "mixed_extractor", 256),
    BenchmarkCase("mixed_extractor_n1024", "mixed_extractor", 1024),
    BenchmarkCase("batch_16x256", "batch_extractor", 256, 16),
)


def _signal(size: int, *, seed_offset: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(BENCHMARK_SEED + size + seed_offset)
    time_axis = np.arange(size, dtype=np.float64) / SAMPLING_FREQUENCY
    phase = np.pi * 8.0 * time_axis**2
    return np.asarray(
        np.sin(2.0 * np.pi * 10.0 * time_axis)
        + 0.35 * np.sin(2.0 * np.pi * 25.0 * time_axis)
        + 0.2 * np.sin(phase)
        + 0.05 * rng.standard_normal(size),
        dtype=np.float64,
    )


def _extractor(*, multitaper: bool = False) -> abfe.FeatureExtractor:
    psd_config: abfe.WelchPSDConfig | abfe.MultitaperPSDConfig
    if multitaper:
        psd_config = abfe.MultitaperPSDConfig(
            window_length=1.0,
            overlap=0.5,
            bandwidth=4.0,
        )
    else:
        psd_config = abfe.WelchPSDConfig(window_length=1.0, overlap=0.5)
    return abfe.FeatureExtractor(abfe.ExtractorConfig(SAMPLING_FREQUENCY, psd_config))


def _prepare_case(case: BenchmarkCase) -> Callable[[], float]:
    signal = _signal(case.size)
    if case.category == "time_direct":
        def compute() -> float:
            values = (
                root_mean_square(signal),
                waveform_length(signal),
                integrated_absolute_value(signal),
                float(zero_crossing_count(signal)),
                float(slope_sign_change_count(signal)),
            )
            return float(sum(values))

    elif case.category == "signal_extractor":
        extractor = _extractor()

        def compute() -> float:
            result = extractor.extract(signal, features=SIGNAL_FEATURES)
            if result.failed_features:
                raise AssertionError(f"unexpected failed features: {result.failed_features}")
            return float(sum(result.values.values()))

    elif case.category == "welch_psd":
        welch_config = abfe.WelchPSDConfig(window_length=1.0, overlap=0.5)

        def compute() -> float:
            psd = abfe.compute_psd(signal, SAMPLING_FREQUENCY, welch_config)
            return float(np.sum(psd.values) + peak_frequency(psd))

    elif case.category == "multitaper_psd":
        multitaper_config = abfe.MultitaperPSDConfig(
            window_length=1.0,
            overlap=0.5,
            bandwidth=4.0,
        )

        def compute() -> float:
            psd = abfe.compute_psd(signal, SAMPLING_FREQUENCY, multitaper_config)
            return float(np.sum(psd.values) + peak_frequency(psd))

    elif case.category == "entropy_linear":
        def compute() -> float:
            return float(
                permutation_entropy(signal, order=3)
                + svd_entropy(signal, order=3)
            )

    elif case.category == "lempel_ziv":
        def compute() -> float:
            return lempel_ziv_complexity(signal)

    elif case.category == "entropy_quadratic":
        def compute() -> float:
            return float(
                approximate_entropy(signal, order=2)
                + fuzzy_entropy(signal, order=2)
                + distribution_entropy(signal, order=2, n_bins=500)
            )

    elif case.category == "sample_entropy_profile":
        def compute() -> float:
            profile = sample_entropy_profile(signal, order=2)
            return float(
                profile.point_count
                + np.sum(profile.r_values)
                + np.sum(profile.se_profile)
            )

    elif case.category == "mixed_extractor":
        extractor = _extractor()

        def compute() -> float:
            result = extractor.extract(signal)
            if result.failed_features or len(result.values) != len(DEFAULT_FEATURE_NAMES):
                raise AssertionError(
                    "mixed extraction failed: "
                    f"{result.failed_features}, {len(result.values)} values"
                )
            return float(sum(result.values.values()))

    elif case.category == "batch_extractor":
        signals = tuple(
            _signal(case.size, seed_offset=index + 1) for index in range(case.batch_size)
        )
        extractor = _extractor()

        def compute() -> float:
            result = extractor.extract_batch(signals, features=SIGNAL_FEATURES)
            if len(result.rows) != case.batch_size or any(
                row.failed_features for row in result.rows
            ):
                raise AssertionError("batch extraction did not produce complete aligned rows")
            return float(sum(sum(row.values.values()) for row in result.rows))

    else:  # pragma: no cover - CASES is fixed above
        raise RuntimeError(f"unknown benchmark category: {case.category}")

    def checked_operation() -> float:
        checksum = compute()
        if not np.isfinite(checksum):
            raise AssertionError(f"benchmark {case.name} produced a nonfinite checksum")
        return checksum

    return checked_operation


def _peak_rss_mib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return peak / divisor


def _measure_case(
    case: BenchmarkCase,
    *,
    repeat_count: int,
    target_repeat_seconds: float,
    maximum_iterations: int,
) -> BenchmarkResult:
    operation = _prepare_case(case)
    checksum = operation()
    start = time.perf_counter()
    second_checksum = operation()
    single_seconds = max(time.perf_counter() - start, 1e-9)
    if not np.isclose(checksum, second_checksum, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"benchmark {case.name} is not deterministic")
    iterations = max(
        1,
        min(maximum_iterations, int(target_repeat_seconds / single_seconds)),
    )
    samples: list[float] = []
    for _ in range(repeat_count):
        started = time.perf_counter()
        for _ in range(iterations):
            checksum = operation()
        samples.append((time.perf_counter() - started) / iterations)
    return BenchmarkResult(
        name=case.name,
        category=case.category,
        size=case.size,
        batch_size=case.batch_size,
        iterations_per_repeat=iterations,
        repeat_count=repeat_count,
        median_seconds=statistics.median(samples),
        minimum_seconds=min(samples),
        maximum_seconds=max(samples),
        peak_rss_mib=_peak_rss_mib(),
        checksum=checksum,
    )


def _worker(
    queue: Any,
    case: BenchmarkCase,
    repeat_count: int,
    target_repeat_seconds: float,
    maximum_iterations: int,
) -> None:
    try:
        queue.put(
            (
                True,
                asdict(
                    _measure_case(
                        case,
                        repeat_count=repeat_count,
                        target_repeat_seconds=target_repeat_seconds,
                        maximum_iterations=maximum_iterations,
                    )
                ),
            )
        )
    except BaseException as exc:
        queue.put((False, f"{type(exc).__name__}: {exc}"))


def _isolated_measurement(
    case: BenchmarkCase,
    *,
    repeat_count: int,
    target_repeat_seconds: float,
    maximum_iterations: int,
) -> BenchmarkResult:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(queue, case, repeat_count, target_repeat_seconds, maximum_iterations),
    )
    process.start()
    process.join()
    try:
        succeeded, payload = queue.get(timeout=2.0)
    except Empty as exc:  # pragma: no cover - abnormal child termination
        raise RuntimeError(
            f"benchmark worker {case.name} exited with code {process.exitcode}"
        ) from exc
    finally:
        queue.close()
    if not succeeded:
        raise RuntimeError(f"benchmark {case.name} failed: {payload}")
    return BenchmarkResult(**payload)


def _profile_cases(cases: Sequence[BenchmarkCase]) -> str:
    sections: list[str] = []
    for case in cases:
        operation = _prepare_case(case)
        profiler = cProfile.Profile()
        profiler.enable()
        checksum = operation()
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
        sections.append(
            f"## {case.name}\n\nchecksum: {checksum:.17g}\n\n```text\n"
            f"{stream.getvalue().rstrip()}\n```\n"
        )
    return "# ABFE performance profiles\n\n" + "\n".join(sections)


def _environment() -> dict[str, str | int]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count() or 0,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "abfe": abfe.__version__,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/baseline.json"),
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("benchmarks/results/profiles.md"),
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--target-repeat-seconds", type=float, default=0.2)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use two repeats and at most two iterations for a fast harness check.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all cases and write machine-readable results and selected profiles."""
    arguments = _arguments()
    repeat_count = 2 if arguments.quick else arguments.repeat
    maximum_iterations = 2 if arguments.quick else arguments.maximum_iterations
    if repeat_count < 1 or arguments.target_repeat_seconds <= 0 or maximum_iterations < 1:
        raise ValueError("repeat and iteration counts must be positive")
    results: list[BenchmarkResult] = []
    for case in CASES:
        result = _isolated_measurement(
            case,
            repeat_count=repeat_count,
            target_repeat_seconds=arguments.target_repeat_seconds,
            maximum_iterations=maximum_iterations,
        )
        results.append(result)
        print(
            f"{result.name:30} {result.median_seconds * 1000:10.3f} ms "
            f"{result.peak_rss_mib:10.1f} MiB"
        )
    payload = {
        "schema_version": 1,
        "measurement_notes": {
            "timing": "median wall time per operation after one untimed warm-up",
            "memory": "isolated-process peak resident set; includes interpreter and libraries",
            "thresholds": "observational only; no machine-specific pass/fail limits",
            "seed": BENCHMARK_SEED,
        },
        "environment": _environment(),
        "results": [asdict(result) for result in results],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    slowest_names = {
        result.name
        for result in sorted(results, key=lambda item: item.median_seconds, reverse=True)[:3]
    }
    slowest_names.add("batch_16x256")
    profile_cases = [case for case in CASES if case.name in slowest_names]
    arguments.profile_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.profile_output.write_text(_profile_cases(profile_cases), encoding="utf-8")


if __name__ == "__main__":
    main()
