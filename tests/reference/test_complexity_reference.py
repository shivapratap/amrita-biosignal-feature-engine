from __future__ import annotations

import json
import os
from importlib import util
from importlib.metadata import distribution
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlparse

import numpy as np
import pytest
from numpy.typing import NDArray

from amrita_biosignal_feature_engine.complexity import (
    fisher_information,
    higuchi_fractal_dimension,
    hjorth_complexity,
    hjorth_mobility,
    katz_fractal_dimension,
    lempel_ziv_complexity,
    petrosian_fractal_dimension,
)

if os.environ.get("ABFE_RUN_REFERENCE") != "1":
    pytest.skip("set ABFE_RUN_REFERENCE=1 to run external references", allow_module_level=True)

antropy = pytest.importorskip("antropy")


def _load_pinned_dihc_complexity() -> ModuleType:
    explicit = os.environ.get("ABFE_DIHC_COMPLEXITY_FILE")
    if explicit is None:
        installed = distribution("dihc-feature-manager")
        path = Path(
            str(
                installed.locate_file(
                    "dihc_feature_manager/features/fractal_complexity.py"
                )
            )
        )
        if not path.is_file():
            direct_url_text = installed.read_text("direct_url.json")
            if direct_url_text is None:
                raise FileNotFoundError(path)
            direct_url = json.loads(direct_url_text)
            repository = Path(unquote(urlparse(str(direct_url["url"])).path))
            path = (
                repository
                / "src/dihc_feature_manager/features/fractal_complexity.py"
            )
    else:
        path = Path(explicit)
    spec = util.spec_from_file_location("_abfe_pinned_dihc_complexity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned DIHC reference from {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dihc_complexity = _load_pinned_dihc_complexity()

pytestmark = pytest.mark.reference


def required_signals() -> dict[str, NDArray[np.float64]]:
    time = np.linspace(0.0, 1.0, 128, endpoint=False)
    return {
        "periodic": np.sin(2.0 * np.pi * 7.0 * time),
        "white_noise": np.random.default_rng(321).normal(size=128),
        "chirp": np.sin(2.0 * np.pi * (2.0 * time + 10.0 * time**2)),
    }


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("normalize", [False, True])
def test_lempel_ziv_matches_antropy_on_abfe_median_binary_sequence(
    name: str, normalize: bool
) -> None:
    signal = required_signals()[name]
    binary = (signal >= np.median(signal)).astype(np.uint8)
    expected = antropy.lziv_complexity(binary, normalize=normalize)
    assert lempel_ziv_complexity(signal, normalize=normalize) == pytest.approx(
        expected, abs=2e-15
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_raw_lempel_ziv_matches_pinned_dihc_commit(name: str) -> None:
    signal = required_signals()[name]
    assert lempel_ziv_complexity(signal, normalize=False) == pytest.approx(
        dihc_complexity.lempel_ziv_complexity(signal)
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("order,delay", [(2, 1), (3, 1), (4, 2)])
def test_fisher_matches_pinned_dihc_on_centered_signals(
    name: str, order: int, delay: int
) -> None:
    signal = required_signals()[name]
    centered = signal - np.mean(signal)
    assert fisher_information(centered, order=order, delay=delay) == pytest.approx(
        dihc_complexity.fisher_information(centered, tau=delay, de=order),
        abs=2e-15,
    )


def test_fisher_translation_and_degeneracy_are_intentional_dihc_improvements() -> None:
    signal = required_signals()["white_noise"]
    assert fisher_information(signal + 100.0) == pytest.approx(
        fisher_information(signal), abs=2e-15
    )
    assert dihc_complexity.fisher_information(signal + 100.0) != pytest.approx(
        dihc_complexity.fisher_information(signal)
    )
    assert np.isnan(fisher_information(np.ones(20)))
    assert dihc_complexity.fisher_information(np.ones(20)) > 0.9


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("k_max", [2, 5, 10])
def test_higuchi_matches_antropy(name: str, k_max: int) -> None:
    signal = required_signals()[name]
    # AntroPy adds a fixed 1e-9 epsilon to the OLS denominator. ABFE uses the
    # unmodified OLS equation, producing a small and bounded reference delta.
    assert higuchi_fractal_dimension(signal, k_max=k_max) == pytest.approx(
        antropy.higuchi_fd(signal, kmax=k_max), abs=5e-9
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_default_higuchi_matches_pinned_dihc(name: str) -> None:
    signal = required_signals()[name]
    assert higuchi_fractal_dimension(signal) == pytest.approx(
        dihc_complexity.higuchi_fd_feature(signal), abs=5e-9
    )


def test_constant_higuchi_degeneracy_matches_references() -> None:
    signal = np.ones(80)
    assert np.isnan(higuchi_fractal_dimension(signal))
    assert np.isnan(antropy.higuchi_fd(signal, kmax=10))
    assert np.isnan(dihc_complexity.higuchi_fd_feature(signal))


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (3.0, 10.0), (-2.0, -4.0)])
def test_hjorth_and_katz_match_antropy(
    name: str, scale: float, offset: float
) -> None:
    signal = scale * required_signals()[name] + offset
    expected_mobility, expected_complexity = antropy.hjorth_params(signal)
    assert hjorth_mobility(signal) == pytest.approx(expected_mobility, abs=2e-15)
    assert hjorth_complexity(signal) == pytest.approx(expected_complexity, abs=2e-15)
    assert katz_fractal_dimension(signal) == pytest.approx(
        antropy.katz_fd(signal), abs=2e-15
    )


@pytest.mark.parametrize("name", ["periodic", "white_noise", "chirp"])
def test_petrosian_matches_antropy_without_zero_derivative_plateaus(
    name: str,
) -> None:
    signal = required_signals()[name]
    assert np.all(np.diff(signal) != 0.0)
    assert petrosian_fractal_dimension(signal) == pytest.approx(
        antropy.petrosian_fd(signal), abs=2e-15
    )


def test_petrosian_plateau_policy_is_an_intentional_reference_difference() -> None:
    signal = np.array([0.0, -1.0, -1.0, -2.0])
    assert petrosian_fractal_dimension(signal) == 1.0
    assert antropy.petrosian_fd(signal) > 1.0
