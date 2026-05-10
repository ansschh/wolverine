"""Canary generator + canary audit invariants."""

from __future__ import annotations

from rasyn.data.decontam.canary_audit import audit_against_rows
from rasyn.data.registry.canary_generator import LAYERS, generate_canaries_for_registry
from rasyn.data.registry.loader import load_sealed_case_registry


def test_generates_per_layer_per_case():
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=4)
    assert len(canaries) == len(reg.cases) * len(LAYERS) * 4

    by_case = {c.case_id: 0 for c in reg.cases}
    for canary in canaries:
        by_case[canary.case_id] += 1
    for case_id, n in by_case.items():
        assert n == len(LAYERS) * 4, f"case {case_id} got {n}"


def test_canary_ids_unique():
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=4)
    ids = [c.canary_id for c in canaries]
    assert len(set(ids)) == len(ids)


def test_canary_inchi_keys_valid_shape():
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=2)
    ik_canaries = [c for c in canaries if c.layer == "inchi_key"]
    for c in ik_canaries:
        s = c.payload
        assert len(s) == 27 and s[14] == "-" and s[25] == "-", f"bad inchi-key shape: {s!r}"


def test_audit_passes_on_clean_rows():
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=2)
    rows = [{"smiles": "CCO", "inchi_key": None, "doi": None, "synonyms": []}]
    res = audit_against_rows(canaries, rows)
    assert res.passed is True
    assert res.survivors == []


def test_audit_catches_smiles_canary_survivor():
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=2)
    payload = next(c for c in canaries if c.layer == "smiles").payload
    rows = [{"smiles": payload}]
    res = audit_against_rows(canaries, rows)
    assert res.passed is False
    assert any(s.payload == payload for s in res.survivors)


def test_audit_catches_doi_canary_survivor():
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=2)
    payload = next(c for c in canaries if c.layer == "doi").payload
    rows = [{"smiles": "CCO", "doi": payload}]
    res = audit_against_rows(canaries, rows)
    assert res.passed is False


def test_audit_catches_title_text_canary_substring():
    reg = load_sealed_case_registry()
    canaries = generate_canaries_for_registry(reg, per_layer=2)
    payload = next(c for c in canaries if c.layer == "title_text").payload
    rows = [{"title": f"some preamble {payload} suffix", "smiles": "CCO"}]
    res = audit_against_rows(canaries, rows)
    assert res.passed is False
