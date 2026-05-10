"""Channel 4 (ML-bearing, scaffolded): learned inverse-delta proposer.

Trained on (parent, candidate, activity_context, liability, desired_delta)
tuples to produce candidates conditioned on the requested delta. Full
implementation lands post-Layer-2 architecture lock; v1 stub returns empty.
"""

from __future__ import annotations

from rasyn.proposer.base import Proposer, ProposerContext
from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.proposer import ProposerOutput


class LearnedInverseDeltaProposer(Proposer):
    channel = "learned_inverse_delta"

    def __init__(self, *, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self._model = None  # loaded post-L2

    def propose(self, packet: ADMETChallengePacket, ctx: ProposerContext) -> ProposerOutput:
        # Stub: returns empty until the model is trained.
        return ProposerOutput(
            case_id=packet.case_id,
            channel=self.channel,
            candidates=[],
            raw_count=0,
            invalid_count=0,
            deduplicated_count=0,
        )
