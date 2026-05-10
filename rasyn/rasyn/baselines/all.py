"""All 8 baselines in one module — they're each <30 lines, splitting is overkill."""

from __future__ import annotations

import random as _random
from typing import Iterable

from rasyn.baselines.base import Baseline
from rasyn.schemas.evidence import CandidateEvidencePacket


def _materialize(candidates: Iterable[CandidateEvidencePacket]) -> list[CandidateEvidencePacket]:
    return list(candidates)


# --- 1. Random -------------------------------------------------------------


class RandomBaseline(Baseline):
    name = "random"

    def __init__(self, seed: int = 42):
        self._rng = _random.Random(seed)

    def score(self, parent_smiles, candidates, liability_type):
        cands = _materialize(candidates)
        order = [(c.candidate_id, self._rng.random()) for c in cands]
        order.sort(key=lambda x: x[1], reverse=True)
        return order


# --- 2. Similarity-only ----------------------------------------------------


class SimilarityOnlyBaseline(Baseline):
    name = "similarity_only"

    def score(self, parent_smiles, candidates, liability_type):
        return sorted(
            ((c.candidate_id, c.structural.tanimoto_to_parent) for c in candidates),
            key=lambda x: x[1],
            reverse=True,
        )


# --- 3. Most-polar ---------------------------------------------------------


class MostPolarBaseline(Baseline):
    name = "most_polar"

    def score(self, parent_smiles, candidates, liability_type):
        # Higher TPSA + lower logP = more polar = "rescue" guess.
        return sorted(
            ((c.candidate_id, c.descriptors.tpsa - c.descriptors.log_p * 10) for c in candidates),
            key=lambda x: x[1],
            reverse=True,
        )


# --- 4. Liability-only-property -------------------------------------------


class LiabilityOnlyBaseline(Baseline):
    name = "liability_only_property"

    def score(self, parent_smiles, candidates, liability_type):
        # Use liability-improvement category alone, ignoring activity.
        order_map = {"large": 5, "moderate": 4, "minor": 3, "none": 2, "worse": 1, "unknown": 0}

        def s(c):
            return order_map.get(c.liability.predicted_improvement_category, 0)

        return sorted(((c.candidate_id, float(s(c))) for c in candidates), key=lambda x: x[1], reverse=True)


# --- 5. Activity-only ------------------------------------------------------


class ActivityOnlyBaseline(Baseline):
    name = "activity_only"

    def score(self, parent_smiles, candidates, liability_type):
        # Use activity-retention prediction alone.
        order_map = {"strong": 4, "acceptable": 3, "weak": 2, "failed": 1, "unknown": 0}

        def s(c):
            return order_map.get(c.activity_retention.predicted_retention_bucket, 0)

        return sorted(((c.candidate_id, float(s(c))) for c in candidates), key=lambda x: x[1], reverse=True)


# --- 6. Weighted-property formula -----------------------------------------


class WeightedPropertyBaseline(Baseline):
    name = "weighted_property"

    def score(self, parent_smiles, candidates, liability_type):
        # Hand-tuned property formula, no learning. Different per liability.
        def s(c):
            d = c.descriptor_deltas
            if liability_type == "hERG":
                return -d.delta_log_p * 0.6 + d.delta_tpsa * 0.03 - d.delta_formal_charge * 0.4
            if liability_type == "solubility":
                return -d.delta_log_p * 0.7 + d.delta_tpsa * 0.04 - d.delta_aromatic_rings * 0.3 + d.delta_fsp3 * 1.0
            if liability_type == "metabolic_stability":
                return -d.delta_log_p * 0.4 + d.delta_tpsa * 0.02
            if liability_type == "oral_exposure":
                return -d.delta_log_p * 0.2 + d.delta_tpsa * 0.01 + d.delta_mw * -0.001
            return 0.0

        return sorted(((c.candidate_id, float(s(c))) for c in candidates), key=lambda x: x[1], reverse=True)


# --- 7. MMP-frequency ------------------------------------------------------


class MMPFrequencyBaseline(Baseline):
    """Rank by how often the candidate's transformation_class appears in the
    mined MMP rule corpus for THIS liability. Higher freq = more "standard".

    Stub: until MMP mining ships, falls back on transformation_class presence
    in a small seeded freq map.
    """

    name = "mmp_frequency"
    SEED_FREQ = {
        "hERG": {"phenyl_to_pyridyl": 5, "n_demethylation": 3, "tertiary_to_secondary_amine": 2, "herg_polarity_increase": 4},
        "solubility": {"sol_phenyl_to_pyridyl": 5, "sol_methyl_to_hydroxymethyl": 3, "phenyl_to_pyridyl": 4},
        "metabolic_stability": {"metstab_methyl_to_fluoro": 4, "metstab_methoxy_to_difluoromethoxy": 3, "metstab_benzylic_fluoro_shield": 3},
        "oral_exposure": {"prodrug_l_valyl_ester": 5, "prodrug_propionate_ester": 3, "prodrug_phosphate": 2},
    }

    def score(self, parent_smiles, candidates, liability_type):
        freq = self.SEED_FREQ.get(liability_type, {})

        def s(c):
            tc = c.structured_rationale.transformation_class
            return float(freq.get(tc, 0))

        return sorted(((c.candidate_id, s(c)) for c in candidates), key=lambda x: x[1], reverse=True)


# --- 8. Medchem heuristic --------------------------------------------------


class MedChemHeuristicBaseline(Baseline):
    """Liability-aware composite of structural similarity + descriptor delta direction."""

    name = "medchem_heuristic"

    def score(self, parent_smiles, candidates, liability_type):
        def s(c):
            sim = c.structural.tanimoto_to_parent
            d = c.descriptor_deltas
            if liability_type == "hERG":
                return sim * 0.4 + max(0, -d.delta_log_p) * 0.3 + max(0, d.delta_tpsa) * 0.02
            if liability_type == "solubility":
                return sim * 0.3 + max(0, -d.delta_log_p) * 0.4 + max(0, d.delta_tpsa) * 0.03 + max(0, d.delta_fsp3) * 0.5
            if liability_type == "metabolic_stability":
                return sim * 0.4 + max(0, -d.delta_log_p) * 0.3
            if liability_type == "oral_exposure":
                # Prodrug case: similarity matters less; ester motif dominates.
                tc = c.structured_rationale.transformation_class or ""
                prodrug_bonus = 1.0 if "prodrug" in tc or "ester" in tc else 0.0
                return sim * 0.2 + prodrug_bonus * 0.6 + max(0, d.delta_tpsa) * 0.01
            return sim

        return sorted(((c.candidate_id, float(s(c))) for c in candidates), key=lambda x: x[1], reverse=True)


ALL_BASELINES: list[type[Baseline]] = [
    RandomBaseline,
    SimilarityOnlyBaseline,
    MostPolarBaseline,
    LiabilityOnlyBaseline,
    ActivityOnlyBaseline,
    WeightedPropertyBaseline,
    MMPFrequencyBaseline,
    MedChemHeuristicBaseline,
]


def get_baseline(name: str, **kwargs) -> Baseline:
    for cls in ALL_BASELINES:
        if cls.name == name:
            return cls(**kwargs)
    raise ValueError(f"Unknown baseline: {name}. Known: {[c.name for c in ALL_BASELINES]}")
