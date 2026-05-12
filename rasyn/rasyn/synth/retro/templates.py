"""RDChiral template extraction + application (RETRO_PLAN R-1 + R-2 Channel 1).

Templates are SMARTS-encoded retrosynthetic rules of the form
"product_pattern>>reactants_pattern" extracted from atom-mapped reactions.
Once extracted, they can be applied to a new product to enumerate
candidate precursor sets (template proposer).

This module wraps rdchiral (https://github.com/connorcoley/rdchiral) and
adds a frequency filter + a stable hash for dedup + a registry-level
decontamination interface.

The actual rdchiral package may not be installed in every environment
(e.g., test machines without C++ toolchain); we import lazily.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class RetroTemplate:
    """One extracted retrosynthetic template."""

    template_smarts: str
    template_hash: str
    extracted_count: int
    source_reaction_ids: tuple[str, ...] = field(default_factory=tuple)


def template_hash(template_smarts: str) -> str:
    """Stable SHA1 hash of a SMARTS template string (16 hex chars)."""
    return hashlib.sha1(template_smarts.encode("utf-8")).hexdigest()[:16]


def extract_template(mapped_rxn_smiles: str) -> str | None:
    """Extract a single template from an atom-mapped reaction.

    Returns the SMARTS string (product>>reactants direction) or None on failure.

    Uses rdchiral.template_extractor.extract_from_reaction; falls back to
    a heuristic stub when rdchiral is unavailable.
    """
    try:
        from rdchiral import template_extractor  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        rxn_dict = {"reactants": mapped_rxn_smiles.split(">>")[0],
                    "products": mapped_rxn_smiles.split(">>")[-1],
                    "_id": "x"}
        result = template_extractor.extract_from_reaction(rxn_dict)
        return result.get("reaction_smarts")
    except Exception:
        return None


def apply_template(template_smarts: str, product_smiles: str) -> list[list[str]]:
    """Apply a retro template to a product SMILES.

    Returns a list of precursor sets; each set is a list of SMILES.
    Empty list if template doesn't fire.
    """
    try:
        from rdchiral.main import rdchiralReactants, rdchiralReaction, rdchiralRun  # type: ignore
    except ImportError:
        return []
    try:
        rxn = rdchiralReaction(template_smarts)
        prod = rdchiralReactants(product_smiles)
        outcomes = rdchiralRun(rxn, prod)
        # outcomes is a list of SMILES strings of dot-separated precursors
        return [o.split(".") for o in outcomes]
    except Exception:
        return []


def extract_templates_bulk(
    mapped_rxn_smiles_iter: Iterable[tuple[str, str]],
    min_frequency: int = 5,
) -> list[RetroTemplate]:
    """Extract templates from many mapped reactions; return frequency-filtered list.

    Input is an iterable of (mapped_rxn_smiles, source_reaction_id) pairs.
    Output is a list of RetroTemplate sorted by extracted_count desc.

    A template with extracted_count < min_frequency is dropped. The
    source_reaction_ids tuple captures up to 16 source IDs for the
    template-level decontamination step in R-1.
    """
    counts: Counter = Counter()
    sources: dict[str, list[str]] = {}
    for mapped, rxn_id in mapped_rxn_smiles_iter:
        smarts = extract_template(mapped)
        if smarts is None:
            continue
        counts[smarts] += 1
        bucket = sources.setdefault(smarts, [])
        if len(bucket) < 16:
            bucket.append(rxn_id)

    templates: list[RetroTemplate] = []
    for smarts, n in counts.most_common():
        if n < min_frequency:
            continue
        templates.append(RetroTemplate(
            template_smarts=smarts,
            template_hash=template_hash(smarts),
            extracted_count=n,
            source_reaction_ids=tuple(sources.get(smarts, [])),
        ))
    return templates
