"""Base class for proposer channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rasyn.schemas.challenge import ADMETChallengePacket
from rasyn.schemas.proposer import ProposerChannel, ProposerOutput


@dataclass
class ProposerContext:
    """Shared context handed to every channel: candidate universe + lookups."""

    candidate_smiles_pool: list[str]
    """Pre-filtered candidate universe (decontaminated, valid). One pool per case."""
    target_pool_size_min: int = 5_000
    target_pool_size_max: int = 20_000


class Proposer(ABC):
    channel: ProposerChannel

    @abstractmethod
    def propose(self, packet: ADMETChallengePacket, ctx: ProposerContext) -> ProposerOutput:
        """Generate candidates for one case."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "channel") or cls.channel is None:
            raise TypeError(f"{cls.__name__} must define a 'channel' class attribute.")
