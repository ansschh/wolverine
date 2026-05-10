"""Mode A (open proposer) and Mode B (closed hard-ranking) evaluation harness.

Mode A: system generates -> filters -> ranks; success = answer in top-k.
Mode B: system ranks a sealed pool containing answer + decoys.

The harness is baseline-symmetric: any baseline can plug in as the ranker
slot, producing apples-to-apples top-k recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rasyn.baselines.base import Baseline
from rasyn.eval.functional_recovery import (
    FunctionalCriteria,
    default_criteria_for_packet,
    passes_functional_criteria,
)
from rasyn.eval.metrics import (
    exact_recall_at_k,
    functional_recall_at_k,
    mean_reciprocal_rank,
    rank_of,
)
from rasyn.evidence.builder import build_candidate_evidence
from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.proposer.ensemble import deterministic_ensemble, run_ensemble
from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.evidence import CandidateEvidencePacket
from rasyn.schemas.proposer import CandidateAnnotation, ProposerOutput

EvidenceMap = dict[str, CandidateEvidencePacket]
ScoreFn = Callable[[str, list[CandidateEvidencePacket], str], list[tuple[str, float]]]


@dataclass
class CaseEvalResult:
    case_id: str
    mode: str  # "A" or "B"
    ranker_name: str
    pool_size: int
    rank_of_answer: int | None
    exact_recall_at_5: bool
    exact_recall_at_10: bool
    exact_recall_at_20: bool
    functional_recall_at_5: bool
    functional_recall_at_10: bool
    functional_recall_at_20: bool
    mrr: float
    invalid_rate: float
    per_channel_attribution: dict[str, int]
    notes: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__


def _build_evidence_for_pool(
    packet: ADMETChallengePacket,
    pool: list[CandidateAnnotation],
) -> EvidenceMap:
    out: EvidenceMap = {}
    for ann in pool:
        ev = build_candidate_evidence(
            parent_smiles=packet.parent_canonical_smiles,
            candidate_smiles=ann.canonical_smiles,
            liability_type=packet.liability_context.liability_type,
            candidate_id=ann.candidate_id,
            proposer_sources=ann.proposer_sources,
        )
        if ev is not None:
            out[ann.candidate_id] = ev
    return out


def _per_channel_attribution(per_channel: list[ProposerOutput]) -> dict[str, int]:
    return {o.channel: len(o.candidates) for o in per_channel}


def evaluate_mode_A(
    *,
    packet: ADMETChallengePacket,
    candidate_pool: list[str],
    ranker: Baseline | ScoreFn,
    answer_inchi_key: str,
    answer_candidate_id: str | None = None,
    functional_target_ids: set[str] | None = None,
    proposers: list[Proposer] | None = None,
    criteria: FunctionalCriteria | None = None,
) -> CaseEvalResult:
    """Mode A: full open-proposer pipeline + ranker. Answer matched by InChIKey."""
    proposers = proposers or deterministic_ensemble()
    ctx = ProposerContext(candidate_smiles_pool=candidate_pool)
    pool, per_channel = run_ensemble(packet, ctx, proposers)

    # Resolve answer to a candidate_id: by InChIKey match in the pool.
    resolved_answer_id = answer_candidate_id
    for ann in pool:
        if ann.inchi_key == answer_inchi_key:
            resolved_answer_id = ann.candidate_id
            break

    evidence_map = _build_evidence_for_pool(packet, pool)
    evidence_list = list(evidence_map.values())

    score_fn: ScoreFn
    ranker_name: str
    if isinstance(ranker, Baseline):
        ranker_name = ranker.name
        score_fn = ranker.score  # type: ignore[assignment]
    else:
        ranker_name = "callable"
        score_fn = ranker

    ranked = score_fn(packet.parent_canonical_smiles, evidence_list, packet.liability_context.liability_type)
    ranked_ids = [cid for cid, _ in ranked]

    if functional_target_ids is None and criteria is not None:
        functional_target_ids = {
            cid for cid, ev in evidence_map.items() if passes_functional_criteria(ev, criteria)
        }
    elif functional_target_ids is None:
        functional_target_ids = {resolved_answer_id} if resolved_answer_id else set()

    invalid = sum(o.invalid_count for o in per_channel)
    raw = sum(o.raw_count for o in per_channel)

    return CaseEvalResult(
        case_id=packet.case_id,
        mode="A",
        ranker_name=ranker_name,
        pool_size=len(pool),
        rank_of_answer=rank_of(resolved_answer_id, ranked_ids) if resolved_answer_id else None,
        exact_recall_at_5=resolved_answer_id is not None and exact_recall_at_k(resolved_answer_id, ranked_ids, k=5),
        exact_recall_at_10=resolved_answer_id is not None and exact_recall_at_k(resolved_answer_id, ranked_ids, k=10),
        exact_recall_at_20=resolved_answer_id is not None and exact_recall_at_k(resolved_answer_id, ranked_ids, k=20),
        functional_recall_at_5=functional_recall_at_k(functional_target_ids, ranked_ids, k=5),
        functional_recall_at_10=functional_recall_at_k(functional_target_ids, ranked_ids, k=10),
        functional_recall_at_20=functional_recall_at_k(functional_target_ids, ranked_ids, k=20),
        mrr=mean_reciprocal_rank(resolved_answer_id, ranked_ids) if resolved_answer_id else 0.0,
        invalid_rate=(invalid / raw) if raw > 0 else 0.0,
        per_channel_attribution=_per_channel_attribution(per_channel),
        notes=None if resolved_answer_id else "answer_inchi_key not present in pool",
    )


def evaluate_mode_B(
    *,
    packet: ADMETChallengePacket,
    sealed_pool_evidence: dict[str, CandidateEvidencePacket],
    answer_candidate_id: str,
    ranker: Baseline | ScoreFn,
    functional_target_ids: set[str] | None = None,
) -> CaseEvalResult:
    """Mode B: rank the (already provided) sealed candidate pool. Answer is known."""
    score_fn: ScoreFn
    ranker_name: str
    if isinstance(ranker, Baseline):
        ranker_name = ranker.name
        score_fn = ranker.score  # type: ignore[assignment]
    else:
        ranker_name = "callable"
        score_fn = ranker

    ranked = score_fn(
        packet.parent_canonical_smiles,
        list(sealed_pool_evidence.values()),
        packet.liability_context.liability_type,
    )
    ranked_ids = [cid for cid, _ in ranked]
    if functional_target_ids is None:
        functional_target_ids = {answer_candidate_id}

    return CaseEvalResult(
        case_id=packet.case_id,
        mode="B",
        ranker_name=ranker_name,
        pool_size=len(sealed_pool_evidence),
        rank_of_answer=rank_of(answer_candidate_id, ranked_ids),
        exact_recall_at_5=exact_recall_at_k(answer_candidate_id, ranked_ids, k=5),
        exact_recall_at_10=exact_recall_at_k(answer_candidate_id, ranked_ids, k=10),
        exact_recall_at_20=exact_recall_at_k(answer_candidate_id, ranked_ids, k=20),
        functional_recall_at_5=functional_recall_at_k(functional_target_ids, ranked_ids, k=5),
        functional_recall_at_10=functional_recall_at_k(functional_target_ids, ranked_ids, k=10),
        functional_recall_at_20=functional_recall_at_k(functional_target_ids, ranked_ids, k=20),
        mrr=mean_reciprocal_rank(answer_candidate_id, ranked_ids),
        invalid_rate=0.0,
        per_channel_attribution={},
    )
