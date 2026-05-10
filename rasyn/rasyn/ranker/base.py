"""Abstract ranker contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from rasyn.schemas.evidence import CandidateEvidencePacket
from rasyn.schemas.ranker import RankerOutput


class Ranker(ABC):
    """A ranker scores (parent, candidate, evidence) tuples for one liability."""

    name: str

    @abstractmethod
    def rank(
        self,
        *,
        parent_smiles: str,
        candidates: Iterable[CandidateEvidencePacket],
        liability_type: str,
        case_id: str,
    ) -> list[RankerOutput]:
        """Returns a 1-indexed-ranked list of RankerOutput, descending by rescue_score."""
