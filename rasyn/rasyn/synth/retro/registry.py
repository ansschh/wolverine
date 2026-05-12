"""Loader for the sealed retrosynthesis-case registry."""

from __future__ import annotations

from pathlib import Path

import yaml

from rasyn.synth.retro.schemas import (
    RetroForbiddenDocuments,
    RetroForbiddenIdentifiers,
    RetroForbiddenNeighborhood,
    RetroHiddenSolution,
    RetroNamedIntermediate,
    RetroSealedCase,
    RetroSealedCaseRegistry,
    RetroSuccessCriteria,
)

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "sealed_case_registry.yaml"


def _build_case(raw: dict) -> RetroSealedCase:
    hidden = raw.get("hidden_solution", {})
    named_intermediates = [
        RetroNamedIntermediate(**ni) for ni in hidden.get("named_intermediates", [])
    ]
    hidden_solution = RetroHiddenSolution(
        name=hidden.get("name", "TBD"),
        canonical_smiles=hidden.get("canonical_smiles"),
        inchi_key=hidden.get("inchi_key"),
        paper_doi=hidden.get("paper_doi"),
        paper_pmid=hidden.get("paper_pmid"),
        reference_route_step_count=hidden.get("reference_route_step_count"),
        reference_reaction_class_sequence=hidden.get("reference_reaction_class_sequence", []),
        named_intermediates=named_intermediates,
        notes=hidden.get("notes"),
    )

    fid = raw.get("forbidden_identifiers", {}) or {}
    forbidden_identifiers = RetroForbiddenIdentifiers(
        synonyms=fid.get("synonyms", []),
        pubchem_cids=fid.get("pubchem_cids", []),
        chembl_ids=fid.get("chembl_ids", []),
        drugbank_ids=fid.get("drugbank_ids", []),
        cas_numbers=fid.get("cas_numbers", []),
    )

    fdocs = raw.get("forbidden_documents", {}) or {}
    forbidden_documents = RetroForbiddenDocuments(
        dois=fdocs.get("dois", []),
        pmids=fdocs.get("pmids", []),
        patent_numbers=fdocs.get("patent_numbers", []),
        title_fragments=fdocs.get("title_fragments", []),
    )

    fnbr = raw.get("forbidden_neighborhood", {}) or {}
    forbidden_neighborhood = RetroForbiddenNeighborhood(
        tanimoto_to_answer=fnbr.get("tanimoto_to_answer", 0.85),
        tanimoto_to_intermediates=fnbr.get("tanimoto_to_intermediates", 0.85),
        template_source_tanimoto=fnbr.get("template_source_tanimoto", 0.85),
        require_same_murcko_for_context=fnbr.get("require_same_murcko_for_context", True),
    )

    sc = raw.get("success_criteria", {}) or {}
    success_criteria = RetroSuccessCriteria(
        max_steps=sc.get("max_steps", 8),
        min_forward_validation_rate=sc.get("min_forward_validation_rate", 0.8),
        require_all_leaves_buyable=sc.get("require_all_leaves_buyable", True),
        verdict_buckets=sc.get("verdict_buckets", []),
        literature_recovery_step_tolerance=sc.get("literature_recovery_step_tolerance", 2),
    )

    return RetroSealedCase(
        case_id=raw["case_id"],
        description=raw["description"],
        target_name=raw["target_name"],
        target_canonical_smiles=raw.get("target_canonical_smiles"),
        target_inchi_key=raw.get("target_inchi_key"),
        max_steps=raw.get("max_steps", 8),
        must_terminate_in=raw.get("must_terminate_in", "commercially_available_building_blocks"),
        buyables_snapshot_date=raw.get("buyables_snapshot_date"),
        scale_hint=raw.get("scale_hint", "10-100 mg"),
        require_condition_prediction=raw.get("require_condition_prediction", True),
        require_forward_validation=raw.get("require_forward_validation", True),
        hidden_solution=hidden_solution,
        forbidden_identifiers=forbidden_identifiers,
        forbidden_documents=forbidden_documents,
        forbidden_neighborhood=forbidden_neighborhood,
        success_criteria=success_criteria,
        notes=raw.get("notes"),
    )


def load_retro_sealed_case_registry(path: Path | str | None = None) -> RetroSealedCaseRegistry:
    """Load + validate the sealed retro-case registry YAML."""
    p = Path(path) if path else DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    cases = [_build_case(c) for c in raw.get("cases", [])]
    return RetroSealedCaseRegistry(
        version=raw.get("version", "0.0.0"),
        locked_at_utc=raw.get("locked_at_utc"),
        spec_refs=raw.get("spec_refs", []),
        cases=cases,
    )
