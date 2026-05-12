"""Abstract base class for single-step retrosynthesis proposers.

Per RETRO_PLAN.md R-2. All five proposer channels (template, graphedit,
seq2seq, retrieval, diffusion) subclass this interface and return a
unified ProposerOutput, which the Retro* planner can consume without
caring about which channel produced it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rasyn.synth.retro.schemas import ProposerChannel, ProposerOutput, ReactionClass


class RetroProposer(ABC):
    """Abstract proposer of single-step retrosynthetic disconnections."""

    #: Channel name, set by subclass.
    channel: ProposerChannel

    @abstractmethod
    def propose(
        self,
        target_smiles: str,
        target_inchi_key: str,
        *,
        top_k: int = 10,
        reaction_class_hint: ReactionClass | None = None,
        **kwargs: Any,
    ) -> ProposerOutput:
        """Return top-K candidate precursor sets for `target_smiles`.

        Subclasses MUST canonicalize precursor SMILES and compute valid
        InChIKeys before returning. Confidences MUST be sorted descending.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} channel={self.channel!r}>"
