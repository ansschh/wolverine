"""Pydantic schemas for Rasyn-Retro (RETRO.md + RETRO_PLAN.md).

All schemas are frozen + extra-forbidden so the audit trail is strict.
Pattern matches rasyn/antibiotic/schemas.py.

Data contracts (RETRO_PLAN §3 phase R-0):
    Molecule                  - canonical molecule reference (InChIKey-keyed)
    Reaction                  - one atom-mapped reaction fact (RETRO.md §9 reaction fact table)
    RetroStep                 - one retrosynthetic disconnection (RETRO.md §9 retrosynthesis step)
    RouteTree                 - AND-OR tree of retro steps terminating at buyables
    CandidateRoute            - one model output: route tree + scores + rationale
    RouteRationale            - structured rationale fields (RETRO.md §11 Component 5)
    ProposerOutput            - unified output of each single-step proposer channel
    ForwardValidationResult   - output of forward-reaction validator
    ConditionPrediction       - reagent/solvent/temperature/catalyst class predictions
    BuyabilityRecord          - one row in the buyables index
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ===== Vocabularies =====

# Coarse reaction-class buckets used for FiLM conditioning. The full RXNFP /
# Schneider class hierarchy has ~1000 templates; v1 uses these 12 high-level
# buckets for the proposer / value / condition models.
ReactionClass = Literal[
    "amide_coupling",
    "suzuki_coupling",
    "buchwald_hartwig",
    "reductive_amination",
    "sn2",
    "sn_ar",
    "negishi",
    "wittig",
    "click",
    "protection_deprotection",
    "other_cross_coupling",
    "unclassified",
]

# Solvent / catalyst / temperature buckets per RETRO_PLAN.md R-3 lock.
SolventClass = Literal[
    "DMSO", "DMF", "DMAc", "NMP", "THF", "2-MeTHF", "dioxane", "MeCN", "DCM", "DCE",
    "EtOH", "MeOH", "iPrOH", "water", "toluene", "xylene", "ether", "EtOAc",
    "acetone", "HFIP", "TFA_neat", "pyridine", "hexane", "heptane",
    "ionic_liquid", "supercritical_CO2", "neat", "solvent_free", "unknown", "other",
]

CatalystClass = Literal[
    "Pd_phosphine", "Pd_NHC", "Ni_phosphine", "Cu", "Ru", "Rh", "Ir", "Pt",
    "organocat_amine", "organocat_acid", "Lewis_acid", "Bronsted_acid",
    "Bronsted_base", "phase_transfer", "photocat", "enzymatic",
    "none", "unknown", "other",
]

TemperatureBin = Literal["cryo", "rt", "warm", "reflux", "high_T", "unknown"]

ReagentClass = Literal[
    "carbodiimide", "HATU_HBTU_family", "T3P", "boronic_acid", "boronate_ester",
    "aryl_halide", "alkyl_halide", "amine_primary", "amine_secondary",
    "carbonyl_protect", "amine_protect", "alcohol_protect",
    "reductant_NaBH4_class", "reductant_LiAlH4_class", "reductant_H2",
    "oxidant_DMP_class", "oxidant_Swern_class",
    "base_NaH", "base_K2CO3_class", "base_TEA_DIPEA", "base_DBU",
    "acid_HCl", "acid_TFA", "acid_sulfonic",
    "azide_source", "alkyne", "halogenation_NBS_class",
    "no_extra_reagent", "unknown", "other",
]

ProposerChannel = Literal["template", "graphedit", "seq2seq", "retrieval", "diffusion"]

QualityTier = Literal["gold", "silver", "bronze"]


# ===== Core molecule / reaction tables =====

class Molecule(BaseModel):
    """Canonical molecule reference for the retro module.

    Mirrors `rasyn.schemas.molecule.MoleculeRef` but adds retro-specific
    fields: commercial availability, cost tier, inventory source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_smiles: str
    inchi_key: str
    name: str | None = None

    # Retro-specific
    commercial_availability: bool = False
    inventory_source: str | None = None  # "ZINC22", "Enamine_REAL_BB", "eMolecules", None
    cost_tier: Literal["tier1", "tier2", "tier3", "unknown"] = "unknown"
    cost_per_g_usd: float | None = None

    # Structural identifiers used by decontamination
    murcko_scaffold: str | None = None
    tautomer_hash: str | None = None

    @field_validator("inchi_key")
    @classmethod
    def _check_inchi_key(cls, v: str) -> str:
        if len(v) != 27 or v[14] != "-" or v[25] != "-":
            raise ValueError(f"InChIKey must be 14-10-1 chars, got: {v!r}")
        return v.upper()

    @field_validator("canonical_smiles")
    @classmethod
    def _no_whitespace(cls, v: str) -> str:
        if any(c.isspace() for c in v):
            raise ValueError(f"canonical_smiles must contain no whitespace: {v!r}")
        return v


class Reaction(BaseModel):
    """One atom-mapped reaction fact (RETRO.md §9 reaction fact table)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reaction_id: str
    reactant_smiles: list[str]  # canonical
    reagent_smiles: list[str] = Field(default_factory=list)
    catalyst_smiles: list[str] = Field(default_factory=list)
    solvent_smiles: list[str] = Field(default_factory=list)
    product_smiles: str  # canonical; sole product (multi-product reactions split upstream)

    reactant_inchi_keys: list[str]
    product_inchi_key: str

    # Atom-mapped reaction SMILES (RXNMapper output), used for template extraction.
    mapped_rxn_smiles: str | None = None

    # Coarse conditions
    reaction_class: ReactionClass = "unclassified"
    rxnfp_class_index: int | None = None  # Schneider class index, -1 if unmapped
    temperature_bin: TemperatureBin = "unknown"
    solvent_class: SolventClass = "unknown"
    catalyst_class: CatalystClass = "unknown"
    reagent_classes: list[ReagentClass] = Field(default_factory=list)

    # Outcome
    yield_pct: float | None = None
    reported_failed: bool = False  # negative example (RETRO.md §8 F)

    # Provenance
    source: Literal["uspto_full", "uspto_50k", "uspto_mit", "uspto_llm", "ord", "ord_erly", "internal", "other"]
    source_record_id: str | None = None
    document_id: str | None = None  # patent / DOI
    quality_tier: QualityTier = "bronze"
    quality_flags: dict[str, bool] = Field(default_factory=dict)

    @field_validator("product_inchi_key")
    @classmethod
    def _check_product_inchi_key(cls, v: str) -> str:
        if len(v) != 27 or v[14] != "-" or v[25] != "-":
            raise ValueError(f"product_inchi_key must be 14-10-1 chars, got: {v!r}")
        return v.upper()

    @field_validator("yield_pct")
    @classmethod
    def _yield_in_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 100.0):
            raise ValueError(f"yield_pct must be in [0, 100], got {v}")
        return v


class RetroStep(BaseModel):
    """One retrosynthetic disconnection (RETRO.md §9 retrosynthesis step table).

    A RetroStep is *backward*: given product, propose precursors.
    The forward validator checks `forward(precursors, conditions) == product`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    retro_step_id: str
    product_inchi_key: str
    precursor_inchi_keys: list[str]
    reaction_class: ReactionClass

    # Provenance
    proposed_by_channel: ProposerChannel
    proposed_by_top_k_rank: int  # which rank this step had within the channel
    confidence: float = Field(ge=0.0, le=1.0)

    # Conditions (filled by R-3 condition predictor)
    predicted_solvent_class: SolventClass = "unknown"
    predicted_catalyst_class: CatalystClass = "unknown"
    predicted_temperature_bin: TemperatureBin = "unknown"
    predicted_reagent_classes: list[ReagentClass] = Field(default_factory=list)

    # Validation (filled by R-3 forward validator)
    forward_validation_pass: bool = False
    forward_tanimoto_to_target: float | None = None

    # Optional reference to a source reaction (when retrieval channel)
    source_reaction_id: str | None = None


class ForwardValidationResult(BaseModel):
    """Output of the forward-reaction validator for one RetroStep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retro_step_id: str
    forward_predicted_product_smiles: str
    forward_predicted_inchi_key: str
    tanimoto_to_target: float = Field(ge=0.0, le=1.0)
    canonical_smiles_match: bool
    pass_rule: Literal["exact_match", "tanimoto>=0.95", "fail"]


class ConditionPrediction(BaseModel):
    """Output of the conditions predictor for one (reactants, product) pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reactant_inchi_keys: list[str]
    product_inchi_key: str
    reaction_class: ReactionClass

    solvent_class: SolventClass
    solvent_logits: dict[str, float] = Field(default_factory=dict)

    catalyst_class: CatalystClass
    catalyst_logits: dict[str, float] = Field(default_factory=dict)

    temperature_bin: TemperatureBin
    temperature_logits: dict[str, float] = Field(default_factory=dict)

    reagent_classes: list[ReagentClass]
    reagent_logits: dict[str, float] = Field(default_factory=dict)

    overall_confidence: float = Field(ge=0.0, le=1.0)


class ProposerOutput(BaseModel):
    """Unified output of one single-step proposer channel.

    Each of the 5 channels (template/graphedit/seq2seq/retrieval/diffusion)
    returns one ProposerOutput per target. The top-K precursor sets are
    in `candidates`, sorted by `confidence` descending.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    channel: ProposerChannel
    target_inchi_key: str
    target_smiles: str
    reaction_class_hint: ReactionClass | None = None  # if conditioned

    # K candidate precursor sets, each a list of precursor InChIKeys
    candidates: list[list[str]]
    candidate_smiles: list[list[str]]
    confidences: list[float]
    reaction_class_predictions: list[ReactionClass]

    # Channel-specific metadata (e.g., template hash for template channel)
    channel_metadata: dict[str, list[str]] = Field(default_factory=dict)


class BuyabilityRecord(BaseModel):
    """One row in the buyables index (frozen ZINC22 + Enamine + eMolecules union).

    Per RETRO_PLAN L4: cost-tier flagged, restricted to tier1 (<=~$10/g)
    for headline reporting; full tiers retained for analysis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inchi_key: str
    canonical_smiles: str
    inventory_sources: list[Literal["ZINC22", "Enamine_REAL_BB", "eMolecules"]]
    cost_tier: Literal["tier1", "tier2", "tier3", "unknown"]
    cost_per_g_usd: float | None = None
    catalog_id: str | None = None
    snapshot_date: str  # ISO date of the frozen snapshot

    @field_validator("inchi_key")
    @classmethod
    def _check_inchi_key(cls, v: str) -> str:
        if len(v) != 27 or v[14] != "-" or v[25] != "-":
            raise ValueError(f"inchi_key must be 14-10-1 chars, got: {v!r}")
        return v.upper()


# ===== Route-level (AND-OR tree) =====

class RouteTreeNode(BaseModel):
    """One node of the AND-OR route tree.

    AND-OR semantics:
      - "OR" nodes are molecules: pick one of several possible expansions
        (disconnections) for this molecule. Children are AND nodes.
      - "AND" nodes are retrosynthetic steps: to satisfy this step, ALL
        children (precursors) must be solved (each a new OR node).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    node_type: Literal["OR_molecule", "AND_step"]

    # For OR_molecule
    molecule_inchi_key: str | None = None
    molecule_smiles: str | None = None
    is_buyable: bool = False
    buyability_record_inchi_key: str | None = None  # FK to BuyabilityRecord

    # For AND_step
    retro_step: RetroStep | None = None

    children_node_ids: list[str] = Field(default_factory=list)

    # Search bookkeeping
    depth: int = 0
    value_estimate: float | None = None  # output of R-4 value model
    expanded: bool = False


class RouteTree(BaseModel):
    """Complete AND-OR route tree rooted at the target molecule.

    Per RETRO.md §11 and RETRO_PLAN R-5. A "completed" tree has all leaves
    being buyable OR_molecule nodes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tree_id: str
    target_inchi_key: str
    target_smiles: str
    nodes: list[RouteTreeNode]
    root_node_id: str

    # Aggregate metrics
    step_count: int  # number of AND_step nodes
    longest_linear_sequence: int
    all_leaves_buyable: bool
    total_estimated_yield_pct: float | None = None
    purchasable_fraction: float = Field(ge=0.0, le=1.0)
    cost_score: float | None = None  # higher = cheaper (negated in route_score)
    risk_score: float = Field(ge=0.0, le=1.0)


class RouteRationale(BaseModel):
    """Structured rationale (RETRO.md §11 Component 5), no free-form text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key_disconnections: list[str]  # e.g. ["amide bond disconnection", "Suzuki coupling"]
    precedent_support_reaction_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    forward_model_recovered_target: bool
    condition_prediction_available: bool
    buyables_coverage_pct: float = Field(ge=0.0, le=100.0)


class CandidateRoute(BaseModel):
    """One model output: a complete RouteTree + scores + rationale.

    Per RETRO.md §9 candidate-route table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_route_id: str
    target_inchi_key: str
    target_smiles: str

    route_tree: RouteTree
    step_predictions: list[RetroStep]
    forward_validation_results: list[ForwardValidationResult]
    condition_predictions: list[ConditionPrediction]

    # Route-level score per RETRO_PLAN L6
    route_score: float
    step_plausibility_product: float = Field(ge=0.0, le=1.0)
    forward_pass_rate: float = Field(ge=0.0, le=1.0)
    step_count_norm: float = Field(ge=0.0, le=1.0)
    cost_norm: float = Field(ge=0.0, le=1.0)
    risk_flags_norm: float = Field(ge=0.0, le=1.0)

    uncertainty: float | None = None
    rationale: RouteRationale


# ===== Sealed-case registry (RETRO_PLAN §8) =====

class RetroForbiddenNeighborhood(BaseModel):
    """Decontamination thresholds for one sealed retro case.

    Per RETRO_PLAN §1 L6 / RETRO.md §13.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Molecule-level Tanimoto cutoff: reactions with product Tan >= this to
    # the sealed target or any named intermediate get removed during R-1.
    tanimoto_to_answer: float = 0.85
    tanimoto_to_intermediates: float = 0.85

    # Template-level: drop every extracted template whose source-reaction set
    # contains any sealed target/intermediate within this Tanimoto.
    template_source_tanimoto: float = 0.85

    require_same_murcko_for_context: bool = True


class RetroForbiddenDocuments(BaseModel):
    """Document quarantine for one sealed retro case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dois: list[str] = Field(default_factory=list)
    pmids: list[str] = Field(default_factory=list)
    patent_numbers: list[str] = Field(default_factory=list)
    title_fragments: list[str] = Field(default_factory=list)


class RetroForbiddenIdentifiers(BaseModel):
    """Identifier quarantine for one sealed retro case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    synonyms: list[str] = Field(default_factory=list)
    pubchem_cids: list[int] = Field(default_factory=list)
    chembl_ids: list[str] = Field(default_factory=list)
    drugbank_ids: list[str] = Field(default_factory=list)
    cas_numbers: list[str] = Field(default_factory=list)


class RetroNamedIntermediate(BaseModel):
    """One named intermediate in the literature route of a sealed case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    canonical_smiles: str | None = None
    inchi_key: str | None = None

    @field_validator("inchi_key")
    @classmethod
    def _check_inchi_key(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 27 or v[14] != "-" or v[25] != "-":
            raise ValueError(f"inchi_key must be 14-10-1 chars, got: {v!r}")
        return v.upper()


class RetroHiddenSolution(BaseModel):
    """The literature reference route for a sealed retro case (revealed only after locked prediction)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    canonical_smiles: str | None = None
    inchi_key: str | None = None
    paper_doi: str | None = None
    paper_pmid: str | None = None
    named_intermediates: list[RetroNamedIntermediate] = Field(default_factory=list)
    reference_route_step_count: int | None = None
    reference_reaction_class_sequence: list[ReactionClass] = Field(default_factory=list)
    notes: str | None = None


class RetroSuccessCriteria(BaseModel):
    """Verdict bucket thresholds for a sealed retro case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = 8
    min_forward_validation_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    require_all_leaves_buyable: bool = True
    verdict_buckets: list[Literal[
        "literature_optimal",
        "literature_competitive",
        "novel_valid",
        "route_proposed_no_literature_baseline",
        "missed",
    ]] = Field(default_factory=list)
    literature_recovery_step_tolerance: int = 2  # +/- vs reference step count


class RetroSealedCase(BaseModel):
    """One sealed retrosynthesis case with locked-prediction protocol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    description: str
    target_name: str
    target_canonical_smiles: str | None = None
    target_inchi_key: str | None = None

    # Constraints packet (RETRO.md §7) the system receives at inference
    max_steps: int = 8
    must_terminate_in: Literal[
        "commercially_available_building_blocks", "tier1_buyables_only", "any_buyables"
    ] = "commercially_available_building_blocks"
    buyables_snapshot_date: str | None = None
    scale_hint: str = "10-100 mg"
    require_condition_prediction: bool = True
    require_forward_validation: bool = True

    hidden_solution: RetroHiddenSolution
    forbidden_identifiers: RetroForbiddenIdentifiers = Field(default_factory=RetroForbiddenIdentifiers)
    forbidden_documents: RetroForbiddenDocuments = Field(default_factory=RetroForbiddenDocuments)
    forbidden_neighborhood: RetroForbiddenNeighborhood = Field(default_factory=RetroForbiddenNeighborhood)

    success_criteria: RetroSuccessCriteria = Field(default_factory=RetroSuccessCriteria)
    notes: str | None = None

    @field_validator("target_inchi_key")
    @classmethod
    def _check_target_inchi_key(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 27 or v[14] != "-" or v[25] != "-":
            raise ValueError(f"target_inchi_key must be 14-10-1 chars, got: {v!r}")
        return v.upper()


class RetroSealedCaseRegistry(BaseModel):
    """The registry of sealed retrosynthesis cases (RETRO_PLAN §8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "0.1.0"
    locked_at_utc: str | None = None
    spec_refs: list[str] = Field(default_factory=list)
    cases: list[RetroSealedCase]
