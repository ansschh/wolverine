"""Load + validate the sealed-case registry YAML.

Usage:
    from rasyn.data.registry.loader import load_sealed_case_registry
    registry = load_sealed_case_registry()  # returns SealedCaseRegistry
    print(registry.cases[0].case_id)        # 'ADMET-001'

The default path resolves relative to this file so that imports work from
any working directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rasyn.schemas.registry import SealedCaseRegistry

REGISTRY_DEFAULT_PATH = Path(__file__).with_name("sealed_case_registry.yaml")


def load_sealed_case_registry(path: Path | str | None = None) -> SealedCaseRegistry:
    """Load and Pydantic-validate the sealed-case registry YAML.

    Raises ValidationError if the YAML doesn't conform to SealedCaseRegistry.
    """
    p = Path(path) if path is not None else REGISTRY_DEFAULT_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return SealedCaseRegistry.model_validate(raw)
