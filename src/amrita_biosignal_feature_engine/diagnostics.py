"""Warnings and immutable extraction diagnostics emitted by ABFE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ABFEWarning(UserWarning):
    """Base class for non-fatal ABFE numerical diagnostics."""


class FrequencyResolutionWarning(ABFEWarning):
    """A requested frequency band is narrower than the PSD resolution."""


class DiagnosticSeverity(str, Enum):
    """Severity of one structured extraction diagnostic."""

    WARNING = "warning"
    UNDEFINED = "undefined"
    ERROR = "error"


class DiagnosticCode(str, Enum):
    """Stable machine-readable extraction diagnostic identifier."""

    UNDEFINED_RESULT = "undefined_result"
    COMPUTATION_ERROR = "computation_error"
    PSD_COMPUTATION_ERROR = "psd_computation_error"
    FREQUENCY_RESOLUTION = "frequency_resolution"
    INVALID_WINDOW = "invalid_window"


@dataclass(frozen=True, slots=True)
class FeatureDiagnostic:
    """Structured diagnostic for one requested output."""

    feature_name: str
    severity: DiagnosticSeverity
    code: DiagnosticCode
    message: str
    exception_type: str | None = None

    def __post_init__(self) -> None:
        if not self.feature_name:
            raise ValueError("feature_name must be nonempty")
        if not self.message:
            raise ValueError("diagnostic message must be nonempty")


__all__ = [
    "ABFEWarning",
    "DiagnosticCode",
    "DiagnosticSeverity",
    "FeatureDiagnostic",
    "FrequencyResolutionWarning",
]
