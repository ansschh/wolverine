"""Identifier validators."""

from __future__ import annotations

from rasyn.utils.identifiers import (
    is_valid_cas,
    is_valid_chembl_id,
    is_valid_doi,
    is_valid_inchi_key,
    is_valid_pmid,
    is_valid_pubchem_cid,
)


def test_doi_validator():
    assert is_valid_doi("10.1039/d4md00275j")
    assert is_valid_doi("10.1234/abc.def")
    assert not is_valid_doi("not-a-doi")
    assert not is_valid_doi("doi:10.1039/d4md00275j")


def test_chembl_id_validator():
    assert is_valid_chembl_id("CHEMBL110")
    assert is_valid_chembl_id("CHEMBL999999999")
    assert not is_valid_chembl_id("chembl110")
    assert not is_valid_chembl_id("110")


def test_pubchem_cid_validator():
    assert is_valid_pubchem_cid("5405")
    assert not is_valid_pubchem_cid("CID5405")


def test_pmid_validator():
    assert is_valid_pmid("12345678")
    assert not is_valid_pmid("PMID:12345678")


def test_cas_validator_shape_only():
    assert is_valid_cas("50679-08-8")
    assert is_valid_cas("124832-26-4")
    assert not is_valid_cas("not-a-cas")


def test_inchi_key_validator():
    assert is_valid_inchi_key("AAAAAAAAAAAAAA-BBBBBBBBBB-N")
    assert not is_valid_inchi_key("AAAAAAAAAAAAAA-BBBBBBBBBBN")  # missing dash
    assert not is_valid_inchi_key("aaaaaaaaaaaaaa-bbbbbbbbbb-n")  # lowercase
