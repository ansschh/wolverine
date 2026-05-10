"""Channel 1: analog retrieval via Morgan-fingerprint nearest neighbours.

Pulls structurally similar molecules from a pre-decontaminated pool.
Pool typically = ChEMBL + same-target analogs + curated paper analogs (post-decontam).
"""

from __future__ import annotations

from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.proposer import (
    CandidateAnnotation,
    ProposerOutput,
    TransformationDescriptor,
)
from rasyn.utils.canonicalize import smiles_to_inchi_key
from rasyn.utils.similarity import morgan_bits, tanimoto


class AnalogRetrievalProposer(Proposer):
    channel = "analog_retrieval"

    def __init__(self, *, min_tanimoto: float = 0.4, max_candidates: int = 5000):
        self.min_tanimoto = min_tanimoto
        self.max_candidates = max_candidates

    def propose(self, packet: ADMETChallengePacket, ctx: ProposerContext) -> ProposerOutput:
        parent_fp = morgan_bits(packet.parent_canonical_smiles)
        if parent_fp is None:
            return ProposerOutput(
                case_id=packet.case_id,
                channel=self.channel,
                candidates=[],
                raw_count=0,
                invalid_count=1,
                deduplicated_count=0,
            )

        scored: list[tuple[float, str]] = []
        invalid = 0
        for cand_smi in ctx.candidate_smiles_pool:
            if cand_smi == packet.parent_canonical_smiles:
                continue
            cand_fp = morgan_bits(cand_smi)
            if cand_fp is None:
                invalid += 1
                continue
            sim = tanimoto(parent_fp, cand_fp)
            if sim >= self.min_tanimoto:
                scored.append((sim, cand_smi))

        scored.sort(reverse=True)
        scored = scored[: self.max_candidates]

        annotations: list[CandidateAnnotation] = []
        seen: set[str] = set()
        for sim, smi in scored:
            ik = smiles_to_inchi_key(smi)
            if not ik or ik in seen:
                continue
            seen.add(ik)
            annotations.append(
                CandidateAnnotation(
                    candidate_id=f"analog-{packet.case_id}-{ik[:14]}",
                    canonical_smiles=smi,
                    inchi_key=ik,
                    parent_inchi_key=packet.parent_inchi_key,
                    proposer_sources=[self.channel],
                    transformation=TransformationDescriptor(
                        transformation_class="analog_retrieval",
                        summary=f"nearest-neighbour analog (Tanimoto={sim:.3f})",
                    ),
                    proposer_confidence=float(sim),
                )
            )

        return ProposerOutput(
            case_id=packet.case_id,
            channel=self.channel,
            candidates=annotations,
            raw_count=len(scored),
            invalid_count=invalid,
            deduplicated_count=len(annotations),
        )
