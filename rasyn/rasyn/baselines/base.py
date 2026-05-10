"""Baseline interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from rasyn.schemas.evidence import CandidateEvidencePacket


class Baseline(ABC):
    name: str

    @abstractmethod
    def score(
        self,
        parent_smiles: str,
        candidates: Iterable[CandidateEvidencePacket],
        liability_type: str,
    ) -> list[tuple[str, float]]:
        """Return [(candidate_id, score), ...] in DESCENDING score order.

        Higher score = better rescue (so top of the returned list is the predicted top-1).
        """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name") or not cls.name:
            raise TypeError(f"{cls.__name__} must set a 'name' class attribute.")
