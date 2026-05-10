"""Compose the 6 proposer channels, dedupe + filter the unioned pool."""

from __future__ import annotations

from collections import defaultdict

from rasyn.proposer.analog import AnalogRetrievalProposer
from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.proposer.forward_opt import ForwardRewardOptimizerProposer
from rasyn.proposer.inverse_delta import LearnedInverseDeltaProposer
from rasyn.proposer.liability_rules import LiabilityRulesProposer
from rasyn.proposer.mmp import MMPTransformerProposer
from rasyn.proposer.novelty import LearnedNoveltyProposer
from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.proposer import CandidateAnnotation, ProposerOutput


def default_ensemble() -> list[Proposer]:
    """The full 6-channel default."""
    return [
        AnalogRetrievalProposer(),
        MMPTransformerProposer(),
        LiabilityRulesProposer(),
        LearnedInverseDeltaProposer(),
        ForwardRewardOptimizerProposer(),
        LearnedNoveltyProposer(),
    ]


def deterministic_ensemble() -> list[Proposer]:
    """Channels 1-3 only (deterministic; usable before any training)."""
    return [
        AnalogRetrievalProposer(),
        MMPTransformerProposer(),
        LiabilityRulesProposer(),
    ]


def run_ensemble(
    packet: ADMETChallengePacket,
    ctx: ProposerContext,
    proposers: list[Proposer] | None = None,
    *,
    max_pool_size: int = 2_000,
) -> tuple[list[CandidateAnnotation], list[ProposerOutput]]:
    """Run all proposer channels, union + dedupe by inchi_key, cap pool size.

    Returns (unioned_filtered_pool, per_channel_outputs). Per-channel outputs
    are kept for attribution metrics.
    """
    proposers = proposers or default_ensemble()
    per_channel: list[ProposerOutput] = []
    by_inchi: dict[str, CandidateAnnotation] = {}
    sources_by_inchi: defaultdict[str, set[str]] = defaultdict(set)

    for p in proposers:
        out = p.propose(packet, ctx)
        per_channel.append(out)
        for ann in out.candidates:
            sources_by_inchi[ann.inchi_key].update(ann.proposer_sources)
            existing = by_inchi.get(ann.inchi_key)
            if existing is None:
                by_inchi[ann.inchi_key] = ann
            else:
                # Merge sources from a duplicate; keep first-seen annotation otherwise.
                pass

    merged: list[CandidateAnnotation] = []
    for ik, ann in by_inchi.items():
        merged.append(ann.model_copy(update={"proposer_sources": sorted(sources_by_inchi[ik])}))

    # Cap by pool size (keep highest proposer_confidence first; None last).
    merged.sort(key=lambda a: (a.proposer_confidence is None, -(a.proposer_confidence or 0.0)))
    if len(merged) > max_pool_size:
        merged = merged[:max_pool_size]

    return merged, per_channel
