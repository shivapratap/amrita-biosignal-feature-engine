"""Validate ABFE wheel and source-distribution metadata and contents."""

from __future__ import annotations

import argparse
import ast
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

PACKAGE = "amrita_biosignal_feature_engine"
SOURCE_MODULES = {
    "__init__.py",
    "complexity.py",
    "diagnostics.py",
    "extractor.py",
    "feature_registry.py",
    "frequency_domain.py",
    "psd.py",
    "py.typed",
    "time_domain.py",
    "validation.py",
    "entropy/__init__.py",
    "entropy/approximate.py",
    "entropy/distribution.py",
    "entropy/fuzzy.py",
    "entropy/permutation.py",
    "entropy/sample_entropy_profile.py",
    "entropy/svd.py",
}
SDIST_ROOT_FILES = {
    "CHANGES.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
}
FORBIDDEN_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".DS_Store",
}


def _project_version() -> str:
    section = ""
    for raw_line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif section == "project" and line.startswith("version"):
            key, separator, value = line.partition("=")
            if key.strip() == "version" and separator:
                parsed = ast.literal_eval(value.strip())
                if isinstance(parsed, str) and parsed:
                    return parsed
    raise AssertionError("pyproject.toml has no literal [project] version")


def _assert_clean_members(members: set[str]) -> None:
    for member in members:
        path = PurePosixPath(member)
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            raise AssertionError(f"generated/cache artifact present: {member}")
        if path.suffix in {".pyc", ".pyo"}:
            raise AssertionError(f"compiled Python artifact present: {member}")
        if "Audit_Report" in path.name or "ReAudit_Report" in path.name:
            raise AssertionError(f"workspace audit report present: {member}")


def _metadata_version(metadata: bytes) -> str:
    parsed = BytesParser().parsebytes(metadata)
    version = parsed.get("Version")
    if version is None:
        raise AssertionError("artifact metadata has no Version field")
    return version


def _check_wheel(path: Path, expected_version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        _assert_clean_members(members)
        expected_sources = {f"{PACKAGE}/{name}" for name in SOURCE_MODULES}
        missing = expected_sources - members
        if missing:
            raise AssertionError(f"wheel is missing package files: {sorted(missing)}")
        unexpected_roots = {
            PurePosixPath(member).parts[0]
            for member in members
            if not (
                member.startswith(f"{PACKAGE}/")
                or ".dist-info/" in member
            )
        }
        if unexpected_roots:
            raise AssertionError(f"wheel has unexpected roots: {sorted(unexpected_roots)}")
        metadata_names = [member for member in members if member.endswith(".dist-info/METADATA")]
        license_names = [member for member in members if ".dist-info/licenses/LICENSE" in member]
        if len(metadata_names) != 1 or len(license_names) != 1:
            raise AssertionError("wheel must contain exactly one METADATA and packaged LICENSE")
        if _metadata_version(archive.read(metadata_names[0])) != expected_version:
            raise AssertionError("wheel metadata version does not match pyproject.toml")


def _check_sdist(path: Path, expected_version: str) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
        _assert_clean_members(members)
        roots = {PurePosixPath(member).parts[0] for member in members}
        if len(roots) != 1:
            raise AssertionError(f"sdist must contain one versioned root: {sorted(roots)}")
        root = next(iter(roots))
        required = {f"{root}/{name}" for name in SDIST_ROOT_FILES}
        required.update(f"{root}/src/{PACKAGE}/{name}" for name in SOURCE_MODULES)
        missing = required - members
        if missing:
            raise AssertionError(f"sdist is missing required files: {sorted(missing)}")
        pkg_info = archive.extractfile(f"{root}/PKG-INFO")
        if pkg_info is None:
            raise AssertionError("sdist has no readable PKG-INFO")
        if _metadata_version(pkg_info.read()) != expected_version:
            raise AssertionError("sdist metadata version does not match pyproject.toml")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    return parser.parse_args()


def main() -> None:
    """Validate the supplied release artifacts."""
    arguments = _arguments()
    expected_version = _project_version()
    _check_wheel(arguments.wheel, expected_version)
    _check_sdist(arguments.sdist, expected_version)
    print(f"validated wheel and sdist for ABFE {expected_version}")


if __name__ == "__main__":
    main()
