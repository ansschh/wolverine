"""Proposer schemas: request → channel output → candidate annotation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProposerChannel = Literal[
    "analog_retrieval",
    "mmp_transformer",
    "liability_rules",
    "learned_inverse_delta",
    "forward_reward_optimizer",
    "learned_novelty",
]


class TransformationDescriptor(BaseModel):
    """How the candidate differs from the parent (one transformation per candidate)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transformation_class: str = Field(
        ...,
        description="e.g. 'basicity_tuning', 'bioisostere_replacement', 'prodrug_motif', 'polarity_increase'.",
    )
    summary: str | None = None
    changed_atoms: list[int] = Field(default_factory=list)
    mmp_rule_id: str | None = None
    transformation_distance: int | None = Field(
        default=None,
        description="Number of heavy-atom changes vs. parent.",
    )


class CandidateAnnotation(BaseModel):
    """One proposed candidate with provenance + transformation metadata.

    The candidate's measured ADMET / potency outcomes are NEVER carried here —
    those are forbidden evidence per the spec. Only computed/predicted properties
    appear later in `CandidateEvidencePacket`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    canonical_smiles: str
    inchi_key: str
    parent_inchi_key: str
    proposer_sources: list[ProposerChannel]
    transformation: TransformationDescriptor | None = None
    proposer_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ProposerRequest(BaseModel):
    """Input to a proposer channel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    challenge_packet_hash: str = Field(..., description="SHA256 of the ADMETChallengePacket.")
    channels: list[ProposerChannel]
    target_pool_size_min: int = 5_000
    target_pool_size_max: int = 20_000
    filtered_pool_size_max: int = 2_000


class ProposerOutput(BaseModel):
    """One proposer channel's output for one case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    channel: ProposerChannel
    candidates: list[CandidateAnnotation]
    raw_count: int = Field(..., description="Candidates generated before any filtering.")
    invalid_count: int = Field(..., description="Dropped for hard validity (valence, canonicalisation).")
    deduplicated_count: int = Field(..., description="Surviving after dedup against parent + within-channel.")
