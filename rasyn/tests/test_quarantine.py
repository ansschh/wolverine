"""Pass-0 quarantine logic tests (no RDKit needed for the scrub-by-id paths)."""

from __future__ import annotations

from rasyn.data.decontam.quarantine import build_forbidden_index, scrub_rows
from rasyn.data.registry.loader import load_sealed_case_registry


def test_build_forbidden_index_seeds_synonyms():
    reg = load_sealed_case_registry()
    idx = build_forbidden_index(reg)
    assert "fexofenadine" in idx.synonyms_lower
    assert "valacyclovir" in idx.synonyms_lower
    assert "oxs008474" in idx.synonyms_lower


def test_oxs_doi_is_indexed():
    reg = load_sealed_case_registry()
    idx = build_forbidden_index(reg)
    assert "10.1039/d4md00275j" in idx.dois


def test_scrub_removes_synonym_match():
    reg = load_sealed_case_registry()
    rows = [
        {"smiles": "CCO", "synonyms": ["fexofenadine"]},
        {"smiles": "CCN", "synonyms": ["something_clean"]},
    ]
    kept, report = scrub_rows(rows, reg, canonicalize=False)
    assert report.removed_by_synonym == 1
    assert len(kept) == 1
    assert kept[0]["synonyms"] == ["something_clean"]


def test_scrub_removes_doi_match():
    reg = load_sealed_case_registry()
    rows = [
        {"smiles": "CCO", "doi": "10.1039/d4md00275j"},
        {"smiles": "CCN", "doi": "10.1234/some-other"},
    ]
    kept, report = scrub_rows(rows, reg, canonicalize=False)
    assert report.removed_by_doi == 1
    assert len(kept) == 1


def test_scrub_keeps_clean_rows():
    reg = load_sealed_case_registry()
    rows = [
        {"smiles": "CCO", "synonyms": ["alcohol"], "doi": "10.1234/clean"},
        {"smiles": "CCN", "synonyms": ["amine"], "doi": "10.1234/also-clean"},
    ]
    kept, report = scrub_rows(rows, reg, canonicalize=False)
    assert report.total_input == 2
    assert report.total_kept == 2
    assert len(kept) == 2


def test_scrub_report_total_input_equals_kept_plus_removed():
    reg = load_sealed_case_registry()
    rows = [
        {"smiles": "CCO", "synonyms": ["fexofenadine"]},
        {"smiles": "CCN", "synonyms": ["valacyclovir"]},
        {"smiles": "CCC", "synonyms": ["clean"]},
    ]
    kept, r = scrub_rows(rows, reg, canonicalize=False)
    removed = (
        r.removed_by_smiles + r.removed_by_inchi_key + r.removed_by_synonym + r.removed_by_chembl_id
        + r.removed_by_pubchem_cid + r.removed_by_cas + r.removed_by_doi + r.removed_by_pmid
        + r.removed_by_chembl_doc_id + r.removed_by_assay_id + r.removed_by_neighbor_to_answer
        + r.removed_by_neighbor_with_context
    )
    assert r.total_input == r.total_kept + removed
    assert r.total_kept == len(kept)
