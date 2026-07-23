"""Amrita BioSignal Feature Engine public package API."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .extractor import (
    BandPowerRatioRequest,
    BandPowerRequest,
    BatchExtractionResult,
    ExtractionProvenance,
    ExtractionResult,
    ExtractorConfig,
    FeatureExtractor,
)
from .psd import MultitaperPSDConfig, PSDResult, WelchPSDConfig, compute_psd

try:
    __version__ = version("amrita-biosignal-feature-engine")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "BandPowerRatioRequest",
    "BandPowerRequest",
    "BatchExtractionResult",
    "ExtractionProvenance",
    "ExtractionResult",
    "ExtractorConfig",
    "FeatureExtractor",
    "MultitaperPSDConfig",
    "PSDResult",
    "WelchPSDConfig",
    "__version__",
    "compute_psd",
]
