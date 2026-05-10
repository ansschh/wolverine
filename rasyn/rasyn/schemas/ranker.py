"""Ranker input/output schemas.

The ranker's main task is pairwise rescue ranking on (parent, candidate, context).
It produces a primary `rescue_score` plus auxiliary subscores and the 7-class
rescue label distribution. Per-subclaim confidences are tracked separately for
calibration. See spec §3.7, §8.4-8.5.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rasyn.schemas.evidence import (
    ActivityRetentionEvidence,
    CandidateEvidencePacket,
    LiabilityEvidence,
    StructuredRationale,
)

RescueLabel = Literal[
    "strong_success",
    "weak_success",
    "failed_activity_loss",
    "failed_no_liability_improvement",
    "failed_wrong_liability",
    "failed_new_liability",
    "uncertain",
]

FailureMode = Literal[
    "activity_lost",
    "liability_not_fixed",
    "wrong_liability_improved",
    "new_liability_introduced",
    "implausible_chemistry",
    "uncertain",
]


class RankerInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    challenge_packet_hash: str
    parent_canonical_smiles: str
    parent_inchi_key: str
    candidate_evidence: CandidateEvidencePacket


class ConfidenceBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: float = Field(..., ge=0.0, le=1.0)
    activity_retention: float = Field(..., ge=0.0, le=1.0)
    liability_improvement: float = Field(..., ge=0.0, le=1.0)
    new_risk: float = Field(..., ge=0.0, le=1.0)
    chemical_plausibility: float = Field(..., ge=0.0, le=1.0)


class RankerOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    candidate_id: str
    rescue_score: float = Field(..., ge=0.0, le=1.0)
    rescue_label_probs: dict[str, float] = Field(
        ..., description="Probability per RescueLabel; should sum to ~1.0."
    )
    failure_mode_probs: dict[str, float] = Field(
        ..., description="Probability per FailureMode; not constrained to sum to 1 (multi-label).",
    )
    activity_retention_pred: ActivityRetentionEvidence
    liability_improvement_pred: LiabilityEvidence
    structured_rationale: StructuredRationale
    confidence: ConfidenceBlock
    rank: int | None = Field(default=None, ge=1, description="1-indexed final rank within the case.")
