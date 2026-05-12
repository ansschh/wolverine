"""Tests for retro reaction + template decontamination."""

from __future__ import annotations

import pytest

from rasyn.data.decontam.retro_quarantine import (
    RetroQuarantineReport,
    build_retro_forbidden_index,
    scrub_reactions,
    scrub_templates,
)
from rasyn.synth.retro.registry import load_retro_sealed_case_registry
from rasyn.synth.retro.templates import RetroTemplate


@pytest.fixture(scope="module")
def fidx():
    reg = load_retro_sealed_case_registry()
    return build_retro_forbidden_index(reg)


def test_forbidden_index_has_oseltamivir(fidx):
    assert "VSZGPKBBMSAYNT-UHFFFAOYSA-N" in fidx.sealed_target_inchi_keys


def test_forbidden_index_has_nirmatrelvir(fidx):
    assert "BFHAYPLBUQVNNJ-UHFFFAOYSA-N" in fidx.sealed_target_inchi_keys


def test_forbidden_index_has_oseltamivir_dois(fidx):
    # Any oseltamivir DOI; we pick a known one from the registry.
    assert "10.1021/op0501390" in fidx.dois


def test_forbidden_index_has_pfizer_patent(fidx):
    assert "WO2021250648" in fidx.patents


def test_scrub_reactions_drops_oseltamivir_product(fidx):
    rows = [
        {
            "product_inchi_key": "VSZGPKBBMSAYNT-UHFFFAOYSA-N",
            "product_smiles": "CCC(CC)OC1C=C(CC(N)C1NC(C)=O)C(=O)OCC",
            "document_id": "10.9999/clean",
        },
        {
            "product_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "product_smiles": "CCO",
            "document_id": "10.9999/clean",
        },
    ]
    report = RetroQuarantineReport()
    kept = list(scrub_reactions(rows, fidx, report=report))
    assert len(kept) == 1
    assert kept[0]["product_smiles"] == "CCO"
    assert report.removed_by_product_inchi_key == 1


def test_scrub_reactions_drops_quarantined_doi(fidx):
    rows = [
        {
            "product_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "product_smiles": "CCO",
            "document_id": "https://doi.org/10.1126/science.abl4784",  # Owen Science 2021
        },
    ]
    report = RetroQuarantineReport()
    kept = list(scrub_reactions(rows, fidx, report=report))
    assert kept == []
    assert report.removed_by_doi == 1


def test_scrub_reactions_drops_pfizer_patent(fidx):
    rows = [
        {
            "product_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "product_smiles": "CCO",
            "document_id": "WO2021250648",
        },
    ]
    report = RetroQuarantineReport()
    kept = list(scrub_reactions(rows, fidx, report=report))
    assert kept == []
    assert report.removed_by_patent == 1


def test_scrub_reactions_keeps_clean_row(fidx):
    rows = [
        {
            "product_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "product_smiles": "CCO",
            "document_id": "10.9999/clean",
        },
    ]
    report = RetroQuarantineReport()
    kept = list(scrub_reactions(rows, fidx, report=report))
    assert len(kept) == 1
    assert report.total_kept_reactions == 1


def test_scrub_reactions_total_input_equals_kept_plus_removed(fidx):
    rows = [
        {
            "product_inchi_key": "VSZGPKBBMSAYNT-UHFFFAOYSA-N",
            "product_smiles": "CCC(CC)OC1C=C(CC(N)C1NC(C)=O)C(=O)OCC",
            "document_id": "10.9999/clean",
        },
        {
            "product_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "product_smiles": "CCO",
            "document_id": "10.9999/clean",
        },
        {
            "product_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
            "product_smiles": "CCN",
            "document_id": "WO2021250648",
        },
    ]
    report = RetroQuarantineReport()
    list(scrub_reactions(rows, fidx, report=report))
    removed = (
        report.removed_by_product_inchi_key + report.removed_by_product_smiles
        + report.removed_by_intermediate_inchi_key + report.removed_by_intermediate_smiles
        + report.removed_by_doi + report.removed_by_patent
        + report.removed_by_product_tanimoto + report.removed_by_intermediate_tanimoto
    )
    assert report.total_input_reactions == report.total_kept_reactions + removed


def test_scrub_templates_drops_oseltamivir_source(fidx):
    """A template extracted from a quarantined source reaction must be dropped."""
    template = RetroTemplate(
        template_smarts="[C:1][O:2]>>[C:1][Cl:3].[H][O:2]",  # dummy
        template_hash="abc123",
        extracted_count=10,
        source_reaction_ids=("clean1", "osel_source"),
    )
    source_lookup = {
        "clean1": "CCO",
        "osel_source": "CCC(CC)OC1C=C(CC(N)C1NC(C)=O)C(=O)OCC",
    }
    report = RetroQuarantineReport()
    kept = scrub_templates([template], source_lookup, fidx, report=report)
    assert kept == []
    assert report.total_input_templates == 1
    assert report.total_kept_templates == 0


def test_scrub_templates_keeps_clean_template(fidx):
    template = RetroTemplate(
        template_smarts="[C:1][O:2]>>[C:1][Cl:3].[H][O:2]",
        template_hash="def456",
        extracted_count=10,
        source_reaction_ids=("clean1", "clean2"),
    )
    source_lookup = {"clean1": "CCO", "clean2": "CCN"}
    report = RetroQuarantineReport()
    kept = scrub_templates([template], source_lookup, fidx, report=report)
    assert len(kept) == 1
