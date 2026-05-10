"""Molecule identifier types.

`MoleculeRef` is the cross-source canonical reference: a canonical SMILES (RDKit
canonicalised, with `chembl_structure_pipeline` standardisation: desalted,
neutralised, tautomer-normalised) plus the corresponding 27-char InChIKey.
The InChIKey is the join key across data sources.

Optional database-specific identifiers may be attached. They are NEVER used as
the primary identity — the (canonical_smiles, inchi_key) pair is the truth.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MoleculeRef(BaseModel):
    """A canonical reference to a molecule.

    `canonical_smiles` and `inchi_key` are optional to support stub
    registry entries (Phase A-0) where the populator script will fill them
    in via PubChem/ChEMBL lookup + RDKit canonicalisation. Runtime code
    that depends on these MUST validate they are present (see
    `MoleculeRef.is_populated`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = Field(default=None, description="Common or generic name (display only, not identity).")
    canonical_smiles: str | None = Field(default=None, description="RDKit canonical SMILES post-standardisation.")
    inchi_key: str | None = Field(default=None, description="27-character InChIKey, e.g. 'XYZABC...-N'.")
    chembl_id: str | None = None
    pubchem_cid: str | None = None
    cas_number: str | None = None
    drugbank_id: str | None = None
    iupac_name: str | None = None

    @field_validator("inchi_key")
    @classmethod
    def _check_inchi_key_shape(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 27 or v[14] != "-" or v[25] != "-":
            raise ValueError(f"InChIKey must match 14-10-1 char layout, got: {v!r}")
        return v.upper()

    @property
    def is_populated(self) -> bool:
        """True iff the populator has filled in canonical_smiles + inchi_key."""
        return self.canonical_smiles is not None and self.inchi_key is not None
