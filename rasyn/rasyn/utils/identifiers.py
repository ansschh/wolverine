"""Validators for cross-source identifiers (CAS, ChEMBL, PubChem CID, DOI, PMID)."""

from __future__ import annotations

import re

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
CHEMBL_RE = re.compile(r"^CHEMBL\d+$")
PUBCHEM_CID_RE = re.compile(r"^\d+$")
PUBCHEM_AID_RE = re.compile(r"^\d+$")
DOI_RE = re.compile(r"^10\.\d{4,9}/[^\s]+$")
PMID_RE = re.compile(r"^\d{1,9}$")
PMCID_RE = re.compile(r"^PMC\d+$")
INCHI_KEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def is_valid_cas(s: str) -> bool:
    """Shape check only; does NOT validate CAS checksum digit."""
    return bool(CAS_RE.match(s))


def is_valid_chembl_id(s: str) -> bool:
    return bool(CHEMBL_RE.match(s))


def is_valid_pubchem_cid(s: str) -> bool:
    return bool(PUBCHEM_CID_RE.match(s))


def is_valid_doi(s: str) -> bool:
    return bool(DOI_RE.match(s))


def is_valid_pmid(s: str) -> bool:
    return bool(PMID_RE.match(s))


def is_valid_inchi_key(s: str) -> bool:
    return bool(INCHI_KEY_RE.match(s))
