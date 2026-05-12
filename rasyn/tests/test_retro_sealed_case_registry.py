"""Sealed-retro-case registry: schema validation + business invariants (RETRO_PLAN R-0)."""

from __future__ import annotations

import pytest

from rasyn.schemas.hashing import hash_model
from rasyn.synth.retro.registry import load_retro_sealed_case_registry
from rasyn.synth.retro.schemas import RetroSealedCaseRegistry


def test_loads_and_validates():
    reg = load_retro_sealed_case_registry()
    assert isinstance(reg, RetroSealedCaseRegistry)


def test_three_retro_cases_present():
    reg = load_retro_sealed_case_registry()
    ids = {c.case_id for c in reg.cases}
    assert ids == {"RETRO-001", "RETRO-002", "RETRO-003"}


def test_no_duplicate_case_ids():
    reg = load_retro_sealed_case_registry()
    ids = [c.case_id for c in reg.cases]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "case_id, expected_target_name",
    [
        ("RETRO-001", "oseltamivir"),
        ("RETRO-002", "nirmatrelvir"),
        ("RETRO-003", "TBD_rasyn_designed"),
    ],
)
def test_case_targets(case_id, expected_target_name):
    reg = load_retro_sealed_case_registry()
    case = next(c for c in reg.cases if c.case_id == case_id)
    assert case.target_name == expected_target_name


def test_oseltamivir_synonyms_quarantined():
    reg = load_retro_sealed_case_registry()
    case = next(c for c in reg.cases if c.case_id == "RETRO-001")
    syns = case.forbidden_identifiers.synonyms
    assert "oseltamivir" in syns
    assert "Tamiflu" in syns
    assert "GS-4071" in syns


def test_oseltamivir_routes_quarantined():
    reg = load_retro_sealed_case_registry()
    case = next(c for c in reg.cases if c.case_id == "RETRO-001")
    title_frags = case.forbidden_documents.title_fragments
    assert any("oseltamivir" in t.lower() for t in title_frags)
    assert any("shikimic" in t.lower() for t in title_frags)


def test_nirmatrelvir_pfizer_quarantined():
    reg = load_retro_sealed_case_registry()
    case = next(c for c in reg.cases if c.case_id == "RETRO-002")
    assert "nirmatrelvir" in case.forbidden_identifiers.synonyms
    assert "PF-07321332" in case.forbidden_identifiers.synonyms
    assert "Paxlovid" in case.forbidden_identifiers.synonyms
    assert "WO2021250648" in case.forbidden_documents.patent_numbers


def test_retro_003_has_no_literature_baseline():
    """RETRO-003 is the Rasyn-designed case — no literature answer to leak."""
    reg = load_retro_sealed_case_registry()
    case = next(c for c in reg.cases if c.case_id == "RETRO-003")
    assert case.target_canonical_smiles is None  # TBD until populated post-R-2
    assert case.target_inchi_key is None
    assert case.hidden_solution.paper_doi is None
    assert "route_proposed_no_literature_baseline" in case.success_criteria.verdict_buckets


def test_decontamination_thresholds_match_plan():
    """RETRO_PLAN §1 L6: default Tanimoto threshold is 0.85 for sealed-case decontam."""
    reg = load_retro_sealed_case_registry()
    for case in reg.cases:
        if case.case_id == "RETRO-003":
            continue  # RETRO-003 has no literature to decontaminate against
        assert case.forbidden_neighborhood.tanimoto_to_answer == 0.85
        assert case.forbidden_neighborhood.tanimoto_to_intermediates == 0.85
        assert case.forbidden_neighborhood.template_source_tanimoto == 0.85


def test_locked_prediction_protocol_required_fields():
    reg = load_retro_sealed_case_registry()
    for case in reg.cases:
        assert case.require_forward_validation is True
        assert case.require_condition_prediction is True
        assert case.must_terminate_in in {
            "commercially_available_building_blocks",
            "tier1_buyables_only",
            "any_buyables",
        }


def test_max_steps_enforced_in_success_criteria():
    reg = load_retro_sealed_case_registry()
    for case in reg.cases:
        assert case.success_criteria.max_steps == case.max_steps


def test_registry_hash_stable_across_loads():
    reg1 = load_retro_sealed_case_registry()
    reg2 = load_retro_sealed_case_registry()
    assert hash_model(reg1) == hash_model(reg2)


def test_spec_refs_present():
    reg = load_retro_sealed_case_registry()
    refs_concat = " ".join(reg.spec_refs)
    assert "RETRO.md" in refs_concat
    assert "RETRO_PLAN.md" in refs_concat
