"""Channel 6 (ML-bearing, scaffolded): pure learned novelty proposer.

Unconstrained generative channel. Moonshot — not required for benchmark pass.
v1 stub returns empty.
"""

from __future__ import annotations

from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.proposer import ProposerOutput


class LearnedNoveltyProposer(Proposer):
    channel = "learned_novelty"

    def __init__(self, *, n_samples: int = 1024, temperature: float = 1.0):
        self.n_samples = n_samples
        self.temperature = temperature

    def propose(self, packet: ADMETChallengePacket, ctx: ProposerContext) -> ProposerOutput:
        return ProposerOutput(
            case_id=packet.case_id,
            channel=self.channel,
            candidates=[],
            raw_count=0,
            invalid_count=0,
            deduplicated_count=0,
        )
