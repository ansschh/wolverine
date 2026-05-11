"""Loader for the sealed antibiotic-case registry."""

from __future__ import annotations

from pathlib import Path

import yaml

from rasyn.antibiotic.schemas import ABXSealedCase, ABXSealedCaseRegistry

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "sealed_case_registry.yaml"


def load_abx_sealed_case_registry(path: Path | str | None = None) -> ABXSealedCaseRegistry:
    """Load + validate the sealed ABX-case registry YAML."""
    p = Path(path) if path else DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(p.read_text())
    cases = [ABXSealedCase(**c) for c in raw.get("cases", [])]
    return ABXSealedCaseRegistry(
        version=raw.get("version", "0.0.0"),
        locked_at_utc=raw.get("locked_at_utc"),
        spec_refs=raw.get("spec_refs", []),
        cases=cases,
    )
