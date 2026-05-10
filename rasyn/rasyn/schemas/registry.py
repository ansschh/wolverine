"""Sealed-case registry + forbidden-entities schemas (Phase A-0).

The sealed-case registry is the single source of truth for what must be
quarantined from training. It is HASHED and FROZEN before any data mining
begins. If a sealed case changes, the registry version bumps and every
downstream artifact (dataset manifest, training manifest, locked predictions)
must be regenerated and re-hashed.

Sealed cases are kept here as a stub at Phase A-0; the canonical SMILES,
InChIKey, ChEMBL/PubChem IDs are populated by `rasyn.data.registry.populate`
which hits public APIs and round-trips through RDKit canonicalisation. We
do NOT hand-write SMILES strings into the YAML — that is the most common
source of silent corruption.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rasyn.schemas.molecule import MoleculeRef

LiabilityType = Literal[
    "hERG",
    "solubility",
    "metabolic_stability",
    "oral_exposure",
    "cyp_inhibition",
    "permeability",
    "cytotoxicity",
]

RescueMode = Literal[
    "direct_analog_safety_rescue",
    "polarity_solubility_rescue",
    "metabolic_soft_spot_rescue",
    "prodrug_exposure_rescue",
    "active_metabolite_safety_rescue",
]


class ForbiddenIdentifiers(BaseModel):
    """Identifiers tied to a sealed case that must be quarantined from training.

    All fields are lists; empty list means "nothing of this kind known yet".
    The populator script fills these in from public APIs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    smiles_variants: list[str] = Field(default_factory=list)
    inchi_keys: list[str] = Field(default_factory=list)
    cas_numbers: list[str] = Field(default_factory=list)
    chembl_ids: list[str] = Field(default_factory=list)
    pubchem_cids: list[str] = Field(default_factory=list)
    drugbank_ids: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    iupac_names: list[str] = Field(default_factory=list)


class ForbiddenDocuments(BaseModel):
    """Documents tied to a sealed case that must be quarantined.

    `title_fragments` and `author_fragments` are case-insensitive substring
    matches used for fuzzy text scrubbing (used to catch papers re-cited
    via slightly different titles).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dois: list[str] = Field(default_factory=list)
    pmids: list[str] = Field(default_factory=list)
    pmcids: list[str] = Field(default_factory=list)
    chembl_doc_ids: list[str] = Field(default_factory=list)
    patent_numbers: list[str] = Field(default_factory=list)
    title_fragments: list[str] = Field(default_factory=list)
    author_fragments: list[str] = Field(default_factory=list)


class ForbiddenAssays(BaseModel):
    """Assay records tied to a sealed case that must be quarantined."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chembl_assay_ids: list[str] = Field(default_factory=list)
    pubchem_aids: list[str] = Field(default_factory=list)


class QuarantineConfig(BaseModel):
    """Decontamination radii (per-case overridable; spec defaults shown)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tanimoto_to_answer: float = Field(default=0.85, ge=0.0, le=1.0)
    tanimoto_with_context: float = Field(default=0.65, ge=0.0, le=1.0)
    require_same_murcko_for_context: bool = True
    require_same_target_for_context: bool = True


class SealedCase(BaseModel):
    """One sealed ADMET-rescue case.

    `parent` and `answer` start with name + database IDs; the populator script
    fills `canonical_smiles` and `inchi_key` via API lookup + RDKit
    canonicalisation. Everything in `forbidden_*` is what we explicitly want
    to scrub; the populator extends these with discovered synonyms / analogs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(..., pattern=r"^[A-Z]+-\d{3}$", description="e.g. 'ADMET-001'")
    description: str
    parent: MoleculeRef
    answer: MoleculeRef
    liability_type: LiabilityType
    rescue_mode: RescueMode
    forbidden_identifiers: ForbiddenIdentifiers = Field(default_factory=ForbiddenIdentifiers)
    forbidden_documents: ForbiddenDocuments = Field(default_factory=ForbiddenDocuments)
    forbidden_assays: ForbiddenAssays = Field(default_factory=ForbiddenAssays)
    quarantine: QuarantineConfig = Field(default_factory=QuarantineConfig)

    @field_validator("case_id")
    @classmethod
    def _normalise_case_id(cls, v: str) -> str:
        return v.upper()


class Canary(BaseModel):
    """A synthetic leakage tracer inserted into raw data before decontamination.

    If a canary survives decontamination, the pipeline halts: the cleaning
    rules missed something. Each canary is a fake SMILES OR a fake document ID
    OR a fake synonym, tagged with the case it traces and the layer it tests.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    canary_id: str
    case_id: str
    layer: Literal["smiles", "inchi_key", "synonym", "doi", "pmid", "chembl_id", "patent", "title_text"]
    payload: str
    inserted_into: list[str] = Field(default_factory=list, description="raw source files / tables this was injected into")


class SealedCaseRegistry(BaseModel):
    """Frozen registry of all sealed cases. Hashed and audit-tracked.

    Bump `version` (semver) on any change. The `locked_at_utc` timestamp is
    the freeze marker — anything trained against this registry must reference
    its hash in its training manifest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    locked_at_utc: str = Field(..., description="ISO 8601 UTC timestamp")
    cases: list[SealedCase]
    canaries: list[Canary] = Field(default_factory=list)
    spec_refs: list[str] = Field(
        default_factory=lambda: [
            "proposer_system_test_cases.md §1007-1049",
            "rasyn_admet_conditioning_architecture_benchmark_spec.md §0.6, §8.5",
            "rasyn_heldout_discovery_demo_context.md §65-76",
        ]
    )

    @field_validator("cases")
    @classmethod
    def _unique_case_ids(cls, v: list[SealedCase]) -> list[SealedCase]:
        ids = [c.case_id for c in v]
        if len(set(ids)) != len(ids):
            dups = [i for i in ids if ids.count(i) > 1]
            raise ValueError(f"Duplicate case_ids: {sorted(set(dups))}")
        return v
