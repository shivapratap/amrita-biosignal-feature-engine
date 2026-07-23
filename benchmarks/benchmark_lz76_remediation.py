"""Reproduce LZ76 parser before/after timing and allocation evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import amrita_biosignal_feature_engine as abfe
from amrita_biosignal_feature_engine.complexity import _lz76_phrase_count

BASE_SEED = 20260723
PREVIOUS_COMMIT = "de44ea7de5948fabe1c696e8eb3d0d6ecdac2b10"
REMEDIATION_COMMIT = "e02f88e84959f016702c2f210110af64478a2d8c"
DEFAULT_SIZES = (512, 1024, 2048, 4096)


@dataclass(frozen=True, slots=True)
class LZ76Comparison:
    """One deterministic before/after LZ76 measurement."""

    size: int
    binary_sha256: str
    phrase_count: int
    previous_median_seconds: float
    suffix_automaton_median_seconds: float
    speedup: float
    previous_peak_allocated_bytes: int
    suffix_automaton_peak_allocated_bytes: int


def _previous_lz76_phrase_count(sequence: NDArray[np.uint8]) -> int:
    """Preserve the exact pre-remediation parser for comparison only."""
    complexity = 1
    prefix_length = 1
    substring_length = 1
    maximum_substring_length = 1
    pointer = 0
    while prefix_length + substring_length <= sequence.size:
        if (
            sequence[pointer + substring_length - 1]
            == sequence[prefix_length + substring_length - 1]
        ):
            substring_length += 1
        else:
            maximum_substring_length = max(
                substring_length, maximum_substring_length
            )
            pointer += 1
            if pointer == prefix_length:
                complexity += 1
                prefix_length += maximum_substring_length
                pointer = 0
                maximum_substring_length = 1
            substring_length = 1
    if substring_length != 1:
        complexity += 1
    return complexity


def _binary_sequence(size: int) -> NDArray[np.uint8]:
    signal = np.random.default_rng(BASE_SEED + size).standard_normal(size)
    return np.asarray(signal >= np.median(signal), dtype=np.uint8)


def _median_seconds(
    function: Callable[[NDArray[np.uint8]], int],
    sequence: NDArray[np.uint8],
    *,
    repeats: int,
) -> float:
    function(sequence)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(sequence)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _peak_allocated_bytes(
    function: Callable[[NDArray[np.uint8]], int],
    sequence: NDArray[np.uint8],
) -> int:
    gc.collect()
    tracemalloc.start()
    function(sequence)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _measure(size: int, *, repeats: int) -> LZ76Comparison:
    sequence = _binary_sequence(size)
    previous_count = _previous_lz76_phrase_count(sequence)
    current_count = _lz76_phrase_count(sequence)
    if current_count != previous_count:
        raise AssertionError(
            f"LZ76 count changed at size {size}: {previous_count} != {current_count}"
        )
    previous_seconds = _median_seconds(
        _previous_lz76_phrase_count, sequence, repeats=repeats
    )
    current_seconds = _median_seconds(
        _lz76_phrase_count, sequence, repeats=repeats
    )
    return LZ76Comparison(
        size=size,
        binary_sha256=hashlib.sha256(sequence.tobytes()).hexdigest(),
        phrase_count=current_count,
        previous_median_seconds=previous_seconds,
        suffix_automaton_median_seconds=current_seconds,
        speedup=previous_seconds / current_seconds,
        previous_peak_allocated_bytes=_peak_allocated_bytes(
            _previous_lz76_phrase_count, sequence
        ),
        suffix_automaton_peak_allocated_bytes=_peak_allocated_bytes(
            _lz76_phrase_count, sequence
        ),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/lz76-remediation.json"),
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
    )
    return parser.parse_args()


def _validate_arguments(*, repeats: int, sizes: Sequence[int]) -> None:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("sizes must contain integers of at least two")
    if len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be unique")


def main() -> None:
    """Measure both parsers and write provenance-complete JSON evidence."""
    arguments = _arguments()
    _validate_arguments(repeats=arguments.repeats, sizes=arguments.sizes)
    results = [
        _measure(size, repeats=arguments.repeats) for size in arguments.sizes
    ]
    clock = time.get_clock_info("perf_counter")
    payload = {
        "schema_version": 2,
        "source_commits": {
            "previous_parser": PREVIOUS_COMMIT,
            "suffix_automaton_remediation": REMEDIATION_COMMIT,
        },
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "abfe": abfe.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "method": {
            "seed_rule": f"{BASE_SEED} + signal length",
            "signal_generator": "NumPy default_rng standard_normal",
            "binarization": "signal >= NumPy median; uint8",
            "binary_digest": "SHA-256 of contiguous uint8 bytes",
            "clock": "time.perf_counter",
            "clock_resolution_seconds": clock.resolution,
            "warmups_per_parser_and_size": 1,
            "repeats": arguments.repeats,
            "statistic": "median wall time of one parser call",
            "measurement_order": "previous parser, then suffix automaton",
            "compatibility": "exact phrase-count equality required before timing",
            "peak_memory": (
                "tracemalloc peak Python allocated bytes for one parser call "
                "after gc.collect; excludes input sequence and native RSS"
            ),
        },
        "results": [asdict(result) for result in results],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        print(
            f"n={result.size:5d} count={result.phrase_count:4d} "
            f"speedup={result.speedup:7.2f}x "
            f"new_peak={result.suffix_automaton_peak_allocated_bytes / 1024:9.1f} KiB"
        )


if __name__ == "__main__":
    main()
