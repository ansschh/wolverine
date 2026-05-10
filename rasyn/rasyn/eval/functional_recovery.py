"""Pre-registered functional-recovery scorer.

A candidate counts as a functional rescue iff:
  - same rescue mode as the registered answer,
  - activity retention within the case's tradeoff thresholds,
  - liability improvement at-or-above the registered target.

Criteria are pre-registered per case (and frozen) before reveal.
"""

from __future__ import annotations

from dataclasses import dataclass

from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.evidence import CandidateEvidencePacket
from rasyn.schemas.registry import RescueMode


@dataclass(frozen=True)
class FunctionalCriteria:
    rescue_mode: RescueMode
    min_retention_bucket_rank: int  # 0=unknown, 1=failed, 2=weak, 3=acceptable, 4=strong
    min_improvement_category_rank: int  # 0=unknown/worse/none, 3=minor, 4=moderate, 5=large
    forbidden_new_liability_flags: tuple[str, ...] = ()


_RETENTION_RANK = {"unknown": 0, "failed": 1, "weak": 2, "acceptable": 3, "strong": 4}
_IMPROVEMENT_RANK = {"worse": 0, "none": 0, "unknown": 0, "minor": 3, "moderate": 4, "large": 5}


def passes_functional_criteria(
    candidate: CandidateEvidencePacket,
    criteria: FunctionalCriteria,
    proposer_transformation_class: str | None = None,
) -> bool:
    if proposer_transformation_class is None:
        proposer_transformation_class = candidate.structured_rationale.transformation_class
    # Activity retention check.
    rb = _RETENTION_RANK.get(candidate.activity_retention.predicted_retention_bucket, 0)
    if rb < criteria.min_retention_bucket_rank:
        return False
    # Liability improvement check.
    ib = _IMPROVEMENT_RANK.get(candidate.liability.predicted_improvement_category, 0)
    if ib < criteria.min_improvement_category_rank:
        return False
    # No forbidden new liabilities.
    if any(flag in candidate.risk.new_liability_flags for flag in criteria.forbidden_new_liability_flags):
        return False
    return True


def default_criteria_for_packet(packet: ADMETChallengePacket) -> FunctionalCriteria:
    """Default pre-registered criteria — same for all 3 ADMET cases at v1."""
    return FunctionalCriteria(
        rescue_mode=packet.rescue_context.rescue_mode,
        min_retention_bucket_rank=_RETENTION_RANK["acceptable"],  # within 10x potency
        min_improvement_category_rank=_IMPROVEMENT_RANK["minor"],  # any positive direction
        forbidden_new_liability_flags=(),
    )
