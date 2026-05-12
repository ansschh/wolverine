"""Tests for retro reaction utilities (canonicalization, classification, hashing)."""

from __future__ import annotations

import pytest

from rasyn.synth.retro.reactions import (
    bucketize_class_name,
    canonicalize_reaction,
    canonicalize_smiles,
    inchi_key_from_smiles,
)
from rasyn.synth.retro.templates import template_hash


# ===== Canonicalization =====

def test_canonicalize_smiles_idempotent_ethanol():
    canon = canonicalize_smiles("CCO")
    assert canon is not None
    assert canonicalize_smiles(canon) == canon


def test_canonicalize_smiles_invalid_returns_none():
    """RDKit returns None for nonsense; pure-string fallback returns the input."""
    out = canonicalize_smiles("not_a_valid_molecule_!@#")
    # In the no-RDKit fallback path we get the input back. Either is acceptable.
    assert out is None or out == "not_a_valid_molecule_!@#"


def test_inchi_key_format():
    ik = inchi_key_from_smiles("CCO")
    if ik is None:
        pytest.skip("rdkit not installed in this env")
    assert len(ik) == 27
    assert ik[14] == "-"
    assert ik[25] == "-"


def test_canonicalize_reaction_ok():
    out = canonicalize_reaction(["CCO", "CC(=O)Cl"], "CCOC(C)=O")
    if out is None:
        pytest.skip("rdkit not installed")
    reactants, product = out
    assert len(reactants) == 2
    assert product is not None


# ===== Class bucketing =====

@pytest.mark.parametrize("raw, bucket", [
    ("amide bond formation", "amide_coupling"),
    ("Peptide_coupling", "amide_coupling"),
    ("Suzuki coupling reaction", "suzuki_coupling"),
    ("buchwald_hartwig amination", "buchwald_hartwig"),
    ("reductive_amination", "reductive_amination"),
    ("SN2 displacement", "sn2"),
    ("SNAr reaction", "sn_ar"),
    ("Negishi cross-coupling", "negishi"),  # has 'Negishi'
    ("Wittig olefination", "wittig"),
    ("CuAAC click reaction", "click"),
    ("Boc protection step", "protection_deprotection"),
    ("some random cross coupling", "other_cross_coupling"),
    (None, "unclassified"),
    ("", "unclassified"),
    ("totally unknown reaction type", "unclassified"),
])
def test_bucketize_class_name(raw, bucket):
    assert bucketize_class_name(raw) == bucket


# ===== Template hash =====

def test_template_hash_stable():
    h1 = template_hash("[C:1][O:2]>>[C:1][Cl:3]")
    h2 = template_hash("[C:1][O:2]>>[C:1][Cl:3]")
    assert h1 == h2
    assert len(h1) == 16


def test_template_hash_changes_with_smarts():
    h1 = template_hash("[C:1][O:2]>>[C:1][Cl:3]")
    h2 = template_hash("[C:1][N:2]>>[C:1][Cl:3]")
    assert h1 != h2
