"""Channel 1: analog retrieval via Morgan-fingerprint nearest neighbours.

Pulls structurally similar molecules from a pre-decontaminated pool.
Pool typically = ChEMBL + same-target analogs + curated paper analogs (post-decontam).

Tanimoto threshold is rescue-mode-aware: prodrugs and active-metabolite
rescues add atoms (e.g. valyl ester) and need looser thresholds, while
direct-analog rescues stay tight.
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

# Per-rescue-mode minimum Tanimoto for retaining a candidate as an analog.
# Rationale (chemistry-aware, NOT learned — should eventually be replaced
# by data-driven percentiles or a learned ranker):
#   prodrug_exposure_rescue : adds an ester group, ~6+ heavy atoms → low threshold
#   polarity_solubility     : N-insertion / OH addition, modest change
#   metabolic_soft_spot     : single-atom swaps, similar size
#   active_metabolite       : oxidation / single group change → moderate
#   direct_analog           : closest analogs only → tight threshold
TANIMOTO_BY_MODE: dict[str, float] = {
    "prodrug_exposure_rescue": 0.20,
    "polarity_solubility_rescue": 0.30,
    "metabolic_soft_spot_rescue": 0.35,
    "active_metabolite_safety_rescue": 0.40,
    "direct_analog_safety_rescue": 0.40,
}
DEFAULT_TANIMOTO = 0.35


class AnalogRetrievalProposer(Proposer):
    channel = "analog_retrieval"

    def __init__(
        self,
        *,
        min_tanimoto: float | None = None,
        max_candidates: int = 5000,
        per_mode_thresholds: dict[str, float] | None = None,
    ):
        # If `min_tanimoto` is given explicitly, it overrides the per-mode lookup.
        # Otherwise the threshold is selected at propose-time from the packet's rescue mode.
        self.override_min_tanimoto = min_tanimoto
        self.max_candidates = max_candidates
        self.per_mode_thresholds = {**TANIMOTO_BY_MODE, **(per_mode_thresholds or {})}

    def _resolve_threshold(self, packet: ADMETChallengePacket) -> float:
        if self.override_min_tanimoto is not None:
            return self.override_min_tanimoto
        return self.per_mode_thresholds.get(packet.rescue_context.rescue_mode, DEFAULT_TANIMOTO)

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

        threshold = self._resolve_threshold(packet)

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
            if sim >= threshold:
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
