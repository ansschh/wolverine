"""Sealed-case registry: schema validation + business invariants."""

from __future__ import annotations

import pytest

from rasyn.data.registry.loader import load_sealed_case_registry
from rasyn.schemas.hashing import hash_model
from rasyn.schemas.registry import SealedCaseRegistry


def test_loads_and_validates():
    reg = load_sealed_case_registry()
    assert isinstance(reg, SealedCaseRegistry)


def test_three_admet_cases_present():
    reg = load_sealed_case_registry()
    case_ids = {c.case_id for c in reg.cases}
    assert case_ids == {"ADMET-001", "ADMET-002", "ADMET-003"}


@pytest.mark.parametrize(
    "case_id, expected_liability, expected_mode",
    [
        ("ADMET-001", "hERG", "active_metabolite_safety_rescue"),
        ("ADMET-002", "oral_exposure", "prodrug_exposure_rescue"),
        ("ADMET-003", "solubility", "polarity_solubility_rescue"),
    ],
)
def test_case_lockdowns(case_id, expected_liability, expected_mode):
    reg = load_sealed_case_registry()
    case = next(c for c in reg.cases if c.case_id == case_id)
    assert case.liability_type == expected_liability
    assert case.rescue_mode == expected_mode


def test_oxs_doi_is_quarantined():
    """The OXS lead-optimization paper must be in ADMET-003's forbidden documents."""
    reg = load_sealed_case_registry()
    case = next(c for c in reg.cases if c.case_id == "ADMET-003")
    assert "10.1039/d4md00275j" in case.forbidden_documents.dois


def test_default_quarantine_thresholds_match_spec():
    """Spec defaults: tanimoto_to_answer >= 0.85; tanimoto_with_context >= 0.65."""
    reg = load_sealed_case_registry()
    for case in reg.cases:
        assert case.quarantine.tanimoto_to_answer == 0.85
        assert case.quarantine.tanimoto_with_context == 0.65


def test_registry_has_known_synonyms():
    reg = load_sealed_case_registry()
    a1 = next(c for c in reg.cases if c.case_id == "ADMET-001")
    assert "fexofenadine" in a1.forbidden_identifiers.synonyms
    assert "terfenadine" in a1.forbidden_identifiers.synonyms

    a2 = next(c for c in reg.cases if c.case_id == "ADMET-002")
    assert "valacyclovir" in a2.forbidden_identifiers.synonyms
    assert "acyclovir" in a2.forbidden_identifiers.synonyms

    a3 = next(c for c in reg.cases if c.case_id == "ADMET-003")
    assert "OXS007570" in a3.forbidden_identifiers.synonyms
    assert "OXS008474" in a3.forbidden_identifiers.synonyms


def test_registry_hash_stable_across_loads():
    reg1 = load_sealed_case_registry()
    reg2 = load_sealed_case_registry()
    assert hash_model(reg1) == hash_model(reg2)


def test_no_duplicate_case_ids():
    reg = load_sealed_case_registry()
    ids = [c.case_id for c in reg.cases]
    assert len(ids) == len(set(ids))
