"""Non-ML composite ranker. Usable before any training.

Combines liability-improvement, activity-retention, structural plausibility,
and a small failure-mode penalty into a single rescue_score in [0, 1].
Conforms to the ranker contract AND quacks like a Baseline (so it can plug
straight into the eval harness via `score(parent, candidates, liability)`).
"""

from __future__ import annotations

from typing import Iterable

from rasyn.ranker.base import Ranker
from rasyn.schemas.evidence import CandidateEvidencePacket
from rasyn.schemas.ranker import ConfidenceBlock, RankerOutput

_RETENTION_RANK = {"unknown": 0, "failed": 0, "weak": 1, "acceptable": 2, "strong": 3}
_IMPROVEMENT_RANK = {"unknown": 0, "worse": 0, "none": 1, "minor": 2, "moderate": 3, "large": 4}


def _rescue_label_probs_from_score(score: float) -> dict[str, float]:
    """Crude calibration: split mass among the 7 labels by score band."""
    if score >= 0.8:
        return {
            "strong_success": 0.55, "weak_success": 0.25,
            "failed_activity_loss": 0.05, "failed_no_liability_improvement": 0.03,
            "failed_wrong_liability": 0.02, "failed_new_liability": 0.05,
            "uncertain": 0.05,
        }
    if score >= 0.5:
        return {
            "strong_success": 0.20, "weak_success": 0.40,
            "failed_activity_loss": 0.10, "failed_no_liability_improvement": 0.10,
            "failed_wrong_liability": 0.05, "failed_new_liability": 0.05,
            "uncertain": 0.10,
        }
    if score >= 0.3:
        return {
            "strong_success": 0.05, "weak_success": 0.20,
            "failed_activity_loss": 0.20, "failed_no_liability_improvement": 0.20,
            "failed_wrong_liability": 0.10, "failed_new_liability": 0.10,
            "uncertain": 0.15,
        }
    return {
        "strong_success": 0.02, "weak_success": 0.05,
        "failed_activity_loss": 0.30, "failed_no_liability_improvement": 0.25,
        "failed_wrong_liability": 0.10, "failed_new_liability": 0.18,
        "uncertain": 0.10,
    }


def _failure_mode_probs(ev: CandidateEvidencePacket) -> dict[str, float]:
    new_liab = 0.6 if ev.risk.new_liability_flags else 0.05
    activity = {"unknown": 0.4, "failed": 0.7, "weak": 0.4, "acceptable": 0.15, "strong": 0.05}[
        ev.activity_retention.predicted_retention_bucket
    ]
    liab_not_fixed = {"unknown": 0.3, "worse": 0.7, "none": 0.5, "minor": 0.3, "moderate": 0.15, "large": 0.05}[
        ev.liability.predicted_improvement_category
    ]
    return {
        "activity_lost": activity,
        "liability_not_fixed": liab_not_fixed,
        "wrong_liability_improved": 0.1,
        "new_liability_introduced": new_liab,
        "implausible_chemistry": 0.05,
        "uncertain": 0.2,
    }


class HeuristicRanker(Ranker):
    name = "heuristic"

    def __init__(
        self,
        *,
        w_liability: float = 0.45,
        w_retention: float = 0.35,
        w_similarity: float = 0.10,
        w_new_risk: float = 0.10,
    ):
        total = w_liability + w_retention + w_similarity + w_new_risk
        self.w_liability = w_liability / total
        self.w_retention = w_retention / total
        self.w_similarity = w_similarity / total
        self.w_new_risk = w_new_risk / total

    def _score_one(self, ev: CandidateEvidencePacket) -> float:
        liab = _IMPROVEMENT_RANK.get(ev.liability.predicted_improvement_category, 0) / 4
        ret = _RETENTION_RANK.get(ev.activity_retention.predicted_retention_bucket, 0) / 3
        sim = ev.structural.tanimoto_to_parent
        new_risk_penalty = 0.7 if ev.risk.new_liability_flags else 0.0
        score = (
            self.w_liability * liab
            + self.w_retention * ret
            + self.w_similarity * sim
            - self.w_new_risk * new_risk_penalty
        )
        return max(0.0, min(1.0, score))

    def rank(
        self,
        *,
        parent_smiles: str,
        candidates: Iterable[CandidateEvidencePacket],
        liability_type: str,
        case_id: str,
    ) -> list[RankerOutput]:
        scored = [(ev, self._score_one(ev)) for ev in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        out: list[RankerOutput] = []
        for rank, (ev, s) in enumerate(scored, start=1):
            out.append(
                RankerOutput(
                    case_id=case_id,
                    candidate_id=ev.candidate_id,
                    rescue_score=float(s),
                    rescue_label_probs=_rescue_label_probs_from_score(s),
                    failure_mode_probs=_failure_mode_probs(ev),
                    activity_retention_pred=ev.activity_retention,
                    liability_improvement_pred=ev.liability,
                    structured_rationale=ev.structured_rationale,
                    confidence=ConfidenceBlock(
                        overall=0.6,
                        activity_retention=ev.activity_retention.predicted_retention_confidence or 0.5,
                        liability_improvement=ev.liability.predicted_improvement_confidence or 0.5,
                        new_risk=0.7,
                        chemical_plausibility=0.7,
                    ),
                    rank=rank,
                )
            )
        return out

    # Baseline-shaped adapter so the harness can use the heuristic ranker
    # exactly like a baseline.
    def score(
        self,
        parent_smiles: str,
        candidates: Iterable[CandidateEvidencePacket],
        liability_type: str,
    ) -> list[tuple[str, float]]:
        return [(ev.candidate_id, self._score_one(ev)) for ev in sorted(candidates, key=self._score_one, reverse=True)]
