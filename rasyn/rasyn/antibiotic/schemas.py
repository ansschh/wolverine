"""Pydantic schemas for antibiotic discovery (per spec §8, §12, §14).

All schemas are frozen + extra-forbidden so the audit trail is strict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ===== Vocabularies =====

# v1 organisms (per spec §14 + §15 — sealed cases + benchmark targets)
Organism = Literal[
    "E.coli", "S.aureus", "MRSA", "K.pneumoniae", "A.baumannii", "P.aeruginosa",
    "N.gonorrhoeae", "M.tuberculosis", "C.difficile", "broad_spectrum", "unknown",
]

GramType = Literal["Gram-positive", "Gram-negative", "atypical", "unknown"]

SpectrumGoal = Literal[
    "broad_spectrum_or_general_antibacterial",
    "pathogen_specific",
    "target_pathogen_specific_or_selective",
    "unknown",
]

AssayType = Literal["MIC", "growth_inhibition", "binary_active_inactive", "zone_of_inhibition", "other"]

CounterScreenType = Literal[
    "cytotoxicity", "hemolysis", "aggregation", "artifact_pattern",
    "mitochondrial_toxicity", "off_target_kinase", "other",
]

HardNegativeType = Literal[
    "active_but_cytotoxic",
    "artifact_risk",
    "wrong_organism",
    "activity_cliff",
    "known_control_only",
    "physicochemical_shortcut",
]

# Multi-head ranker label categories
AntibacterialLabel = Literal["active", "inactive", "weak", "unknown"]
SelectivityLabel = Literal["selective", "broadly_active", "cytotoxic", "unknown"]
DiscoveryLabel = Literal[
    "novel_functional",   # active + selective + novel scaffold
    "active_known",       # active but similar to known antibiotics
    "active_toxic",       # active but cytotoxic / hemolytic
    "artifact",           # likely false positive
    "inactive",
    "unknown",
]


# ===== Tables (spec §8) =====

class ABXMoleculeRef(BaseModel):
    """One canonical molecule, mirrors ADMET MoleculeRef but adds antibiotic-relevant fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_smiles: str | None = None
    inchi_key: str | None = None
    name: str | None = None

    chembl_id: str | None = None
    pubchem_cid: str | None = None
    coadd_id: str | None = None
    drugbank_id: str | None = None
    repurposing_hub_id: str | None = None

    tautomer_hash: str | None = None
    murcko_scaffold: str | None = None

    # Antibiotic-specific descriptor flags
    is_known_antibiotic: bool = False
    known_antibiotic_class: str | None = None  # e.g., "beta_lactam", "fluoroquinolone"

    @field_validator("inchi_key")
    @classmethod
    def _check_inchi_key(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 27 or v[14] != "-" or v[25] != "-":
            raise ValueError(f"InChIKey must be 14-10-1 chars, got: {v!r}")
        return v.upper()


class AntibacterialAssayFact(BaseModel):
    """One measured antibacterial outcome.

    Per spec §8.2: molecule × organism × assay → activity label, with quality flags.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fact_id: str
    molecule_inchi_key: str
    organism: Organism
    gram_type: GramType = "unknown"
    assay_type: AssayType
    endpoint: str  # e.g., "MIC_ug_per_mL", "growth_inhibition_%_at_10uM", "active_inactive"
    standard_value: float | None = None
    standard_units: str | None = None
    activity_label: AntibacterialLabel
    source: str  # "co_add", "chembl", "pubchem", "literature", "internal"
    source_record_id: str | None = None
    document_id: str | None = None  # paper DOI / patent
    quality_flags: dict[str, bool] = Field(default_factory=dict)


class CounterScreenFact(BaseModel):
    """One counter-screen measurement (cytotoxicity, hemolysis, etc.). Spec §8.3."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fact_id: str
    molecule_inchi_key: str
    counter_screen_type: CounterScreenType
    cell_line_or_system: str  # e.g., "HepG2", "RBC_human", "PAINS_alert"
    endpoint: str  # e.g., "IC50_uM", "%_lysis_at_20uM", "pattern_match"
    standard_value: float | None = None
    standard_units: str | None = None
    label: Literal["cytotoxic", "non_cytotoxic", "hemolytic", "aggregating", "artifact_alert", "clean", "unknown"]
    source: str
    source_record_id: str | None = None


class AntibioticRankingTask(BaseModel):
    """One organism-conditioned ranking task: parent context + candidate list with labels.

    Spec §8.4 — analogous to ADMET's candidate_sets but with organism conditioning
    and antibiotic-specific labels (selectivity, hard-negative types).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    organism: Organism
    gram_type: GramType
    spectrum_goal: SpectrumGoal
    candidate_inchi_keys: list[str]
    antibacterial_labels: list[AntibacterialLabel]
    selectivity_labels: list[SelectivityLabel]
    discovery_labels: list[DiscoveryLabel]
    hard_negative_types: list[HardNegativeType | None]
    source_fact_ids: list[str] = Field(default_factory=list)


class GenerativeTrainingExample(BaseModel):
    """One generative training example for Channels E (fragment) or F (edit).

    Spec §8.5 — input to the conditional diffusion training stages.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    full_molecule_inchi_key: str
    fragment_inchi_key: str | None = None  # for fragment-conditioned generation
    seed_inchi_key: str | None = None       # for edit-style generation
    organism_context: Organism
    activity_label: AntibacterialLabel
    selectivity_label: SelectivityLabel
    conditioning_tags: list[str] = Field(default_factory=list)  # e.g., ["antibacterial=active", "tox=clean"]


# ===== Challenge packet (per spec §7) =====

class ABXOrganismContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    organism: Organism
    gram_type: GramType = "unknown"
    spectrum_goal: SpectrumGoal = "unknown"


class ABXSelectivityContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    avoid_mammalian_cytotoxicity: bool = True
    avoid_hemolysis: bool = True
    avoid_known_antibiotic_near_duplicates: bool = True
    avoid_broad_nonspecific_membrane_disruption: bool = False


class ABXCandidateContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    library: str  # "anonymized repurposing-like library", "screening library", "de novo", etc.
    library_size_hint: int | None = None
    pool_size_target: int = 5000


class ABXChallengePacket(BaseModel):
    """Per-case input the system receives at inference. Hash goes into LockedPrediction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    organism_context: ABXOrganismContext
    selectivity_context: ABXSelectivityContext
    candidate_context: ABXCandidateContext
    schema_version: str = "0.1.0"


# ===== Ranker output (per spec §12) =====

class FailureModeProbabilities(BaseModel):
    """Per spec §12.6 — multi-class failure-mode head output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inactive: float
    cytotoxic: float
    artifact: float
    organism_mismatch: float
    known_control_only: float


class ABXRankerOutput(BaseModel):
    """Per-candidate multi-head ranker output. Spec §12."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_inchi_key: str
    candidate_smiles: str

    antibacterial_score: float       # organism-conditioned activity probability
    organism_specific_score: float   # selectivity for the requested organism vs others
    selectivity_score: float         # selectivity vs mammalian / off-targets
    cytotoxicity_risk: float
    hemolysis_risk: float
    artifact_risk: float

    known_antibiotic_similarity_penalty: float  # high if too close to known antibiotics
    training_active_similarity_penalty: float   # high if too close to training actives (memorization)
    novelty_score: float                         # high if novel chemotype
    synthesizability_score: float

    uncertainty_score: float
    failure_mode_probabilities: FailureModeProbabilities

    final_discovery_score: float    # composite per spec §12 formula


# ===== Sealed case registry (per spec §14, §19) =====

class ABXSealedCase(BaseModel):
    """One sealed antibiotic discovery case with forbidden-entity lists."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    description: str
    organism_context: ABXOrganismContext
    selectivity_context: ABXSelectivityContext
    candidate_context: ABXCandidateContext

    hidden_solution: dict = Field(default_factory=dict)
    forbidden_identifiers: dict = Field(default_factory=dict)
    forbidden_documents: dict = Field(default_factory=dict)
    forbidden_assays: dict = Field(default_factory=dict)
    forbidden_neighborhoods: dict = Field(default_factory=dict)

    success_criteria: dict = Field(default_factory=dict)
    notes: str | None = None


class ABXSealedCaseRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str = "0.1.0"
    locked_at_utc: str | None = None
    spec_refs: list[str] = Field(default_factory=list)
    cases: list[ABXSealedCase]
