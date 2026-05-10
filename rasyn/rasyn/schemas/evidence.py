"""CandidateEvidencePacket — structured evidence handed to the rescue ranker.

All fields here are computed (descriptors, similarity) or predicted by clean
internal models. The candidate's measured hidden outcomes (potency, hERG IC50,
solubility) are NEVER allowed here. See spec §2.1 (forbidden evidence).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RetentionBucket = Literal["strong", "acceptable", "weak", "failed", "unknown"]
ImprovementCategory = Literal["large", "moderate", "minor", "none", "worse", "unknown"]


class StructuralEvidence(BaseModel):
    """Pairwise structural relationship between candidate and parent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tanimoto_to_parent: float = Field(..., ge=0.0, le=1.0, description="ECFP4 / Morgan-1024 Tanimoto.")
    murcko_scaffold_match: bool
    murcko_scaffold_smiles: str | None = None
    transformation_distance: int = Field(..., ge=0, description="Heavy-atom changes vs. parent.")
    mcs_atom_count: int | None = None
    pharmacophore_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    shape_similarity: float | None = Field(default=None, ge=0.0, le=1.0)


class DescriptorBlock(BaseModel):
    """RDKit-computed property block (absolute, candidate)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mw: float
    log_p: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    aromatic_rings: int
    fsp3: float
    formal_charge: int
    log_d_estimate: float | None = None
    pka_estimate: float | None = None


class DescriptorDeltas(BaseModel):
    """candidate - parent for each absolute descriptor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta_mw: float
    delta_log_p: float
    delta_tpsa: float
    delta_hbd: int
    delta_hba: int
    delta_rotatable_bonds: int
    delta_aromatic_rings: int
    delta_fsp3: float
    delta_formal_charge: int
    delta_log_d: float | None = None
    delta_pka: float | None = None


class ActivityRetentionEvidence(BaseModel):
    """Predicted likelihood the candidate retains the parent's pharmacology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    predicted_retention_bucket: RetentionBucket
    predicted_retention_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pharmacophore_preservation: float | None = Field(default=None, ge=0.0, le=1.0)
    auxiliary_predicted_potency_fold: float | None = Field(
        default=None,
        description="Clean auxiliary model's predicted candidate-to-parent potency fold (>1 = worse).",
    )


class LiabilityEvidence(BaseModel):
    """Mechanism-aware evidence on whether the liability is fixed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    liability_drivers_in_parent: list[str] = Field(default_factory=list)
    candidate_changes_affecting_liability: list[str] = Field(default_factory=list)
    predicted_improvement_category: ImprovementCategory
    predicted_improvement_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    auxiliary_absolute_prediction: float | None = None
    auxiliary_delta_prediction: float | None = None


class StructuredRationale(BaseModel):
    """Structured fields, NOT free-form text. See spec §4.7."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    liability_driver_features: list[str] = Field(default_factory=list)
    modified_features: list[str] = Field(default_factory=list)
    preserved_activity_features: list[str] = Field(default_factory=list)
    transformation_class: str | None = None
    expected_delta_direction: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'hERG': 'decrease', 'logP': 'decrease', 'activity': 'retain'}",
    )
    failure_mode_risks: list[str] = Field(default_factory=list)


class RiskEvidence(BaseModel):
    """New-liability + plausibility flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    new_liability_flags: list[str] = Field(default_factory=list)
    synthesizability_score: float | None = Field(default=None, description="SAScore or equivalent (1=easy, 10=hard).")
    reactive_alert_flags: list[str] = Field(default_factory=list)


class CandidateEvidencePacket(BaseModel):
    """Everything the ranker sees about one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    parent_inchi_key: str
    canonical_smiles: str
    inchi_key: str
    structural: StructuralEvidence
    descriptors: DescriptorBlock
    descriptor_deltas: DescriptorDeltas
    activity_retention: ActivityRetentionEvidence
    liability: LiabilityEvidence
    risk: RiskEvidence
    structured_rationale: StructuredRationale
    proposer_sources: list[str]
