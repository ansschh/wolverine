"""Build a CandidateEvidencePacket for one (parent, candidate) pair.

This is the rule-based v1 builder. Auxiliary "clean" predictors (Phase B-3
training) plug in for the activity-retention and liability-improvement
prediction fields once trained.
"""

from __future__ import annotations

from typing import Literal

from rasyn.evidence.liability_drivers import detect_liability_drivers
from rasyn.schemas.evidence import (
    ActivityRetentionEvidence,
    CandidateEvidencePacket,
    DescriptorBlock,
    LiabilityEvidence,
    RiskEvidence,
    StructuralEvidence,
    StructuredRationale,
)
from rasyn.utils.canonicalize import smiles_to_inchi_key
from rasyn.utils.descriptors import descriptor_block_from_smiles, descriptor_deltas
from rasyn.utils.similarity import (
    heavy_atom_difference,
    morgan_bits,
    murcko_match,
    murcko_scaffold_smiles,
    tanimoto,
)

ImprovementCategory = Literal["large", "moderate", "minor", "none", "worse", "unknown"]


def _heuristic_improvement(liability: str, deltas) -> ImprovementCategory:
    """Cheap rule-based prediction; replaced by a clean aux model post-B3 training."""
    if liability == "hERG":
        # Lower logP + higher TPSA + lower formal charge tend to reduce hERG.
        score = (-deltas.delta_log_p) * 0.4 + (deltas.delta_tpsa) * 0.02 + (-deltas.delta_formal_charge) * 0.2
    elif liability == "solubility":
        # Lower logP, higher TPSA, fewer aromatic rings, higher fsp3 -> better solubility.
        score = (-deltas.delta_log_p) * 0.5 + (deltas.delta_tpsa) * 0.02 + (-deltas.delta_aromatic_rings) * 0.3 + (deltas.delta_fsp3) * 1.0
    elif liability == "metabolic_stability":
        # Lower logP often helps; blocking benzylic CH not captured in descriptors alone.
        score = (-deltas.delta_log_p) * 0.4
    elif liability == "oral_exposure":
        score = (-deltas.delta_log_p) * 0.2 + (deltas.delta_tpsa) * 0.01
    else:
        return "unknown"
    if score > 1.5:
        return "large"
    if score > 0.6:
        return "moderate"
    if score > 0.1:
        return "minor"
    if score > -0.3:
        return "none"
    return "worse"


def _heuristic_retention_bucket(deltas, structural: StructuralEvidence):
    """Cheap structural-similarity heuristic for activity retention."""
    if structural.tanimoto_to_parent >= 0.85 and structural.transformation_distance <= 3:
        return "strong", 0.6
    if structural.tanimoto_to_parent >= 0.7 and structural.transformation_distance <= 6:
        return "acceptable", 0.5
    if structural.tanimoto_to_parent >= 0.55:
        return "weak", 0.45
    return "unknown", 0.3


def build_structural_evidence(parent_smiles: str, candidate_smiles: str) -> StructuralEvidence | None:
    fp_p = morgan_bits(parent_smiles)
    fp_c = morgan_bits(candidate_smiles)
    if fp_p is None or fp_c is None:
        return None
    sim = tanimoto(fp_p, fp_c)
    diff = heavy_atom_difference(parent_smiles, candidate_smiles) or 0
    return StructuralEvidence(
        tanimoto_to_parent=float(sim),
        murcko_scaffold_match=murcko_match(parent_smiles, candidate_smiles),
        murcko_scaffold_smiles=murcko_scaffold_smiles(candidate_smiles),
        transformation_distance=int(diff),
    )


def build_candidate_evidence(
    *,
    parent_smiles: str,
    candidate_smiles: str,
    liability_type: str,
    candidate_id: str,
    proposer_sources: list[str],
    parent_descriptors: DescriptorBlock | None = None,
) -> CandidateEvidencePacket | None:
    """Compose a full CandidateEvidencePacket from a (parent, candidate) pair."""
    structural = build_structural_evidence(parent_smiles, candidate_smiles)
    if structural is None:
        return None

    cand_desc = descriptor_block_from_smiles(candidate_smiles)
    if cand_desc is None:
        return None
    parent_desc = parent_descriptors or descriptor_block_from_smiles(parent_smiles)
    if parent_desc is None:
        return None

    deltas = descriptor_deltas(parent=parent_desc, candidate=cand_desc)

    parent_drivers = detect_liability_drivers(parent_smiles, liability_type)
    candidate_drivers = detect_liability_drivers(candidate_smiles, liability_type)
    candidate_changes = sorted(set(parent_drivers) - set(candidate_drivers))
    new_liability_flags = sorted(set(candidate_drivers) - set(parent_drivers))

    improvement = _heuristic_improvement(liability_type, deltas)
    retention_bucket, retention_conf = _heuristic_retention_bucket(deltas, structural)

    activity_retention = ActivityRetentionEvidence(
        predicted_retention_bucket=retention_bucket,
        predicted_retention_confidence=retention_conf,
        pharmacophore_preservation=structural.tanimoto_to_parent,
    )

    liability = LiabilityEvidence(
        liability_drivers_in_parent=parent_drivers,
        candidate_changes_affecting_liability=candidate_changes,
        predicted_improvement_category=improvement,
        predicted_improvement_confidence=0.5,  # rule-based; calibration happens post-B3
    )

    risk = RiskEvidence(
        new_liability_flags=new_liability_flags,
        synthesizability_score=None,  # SAScore wired in B3 if needed
        reactive_alert_flags=[],
    )

    rationale = StructuredRationale(
        liability_driver_features=parent_drivers,
        modified_features=candidate_changes,
        preserved_activity_features=[],  # populated by ML proposer/ranker layer
        transformation_class=None,
        expected_delta_direction={
            liability_type: "decrease" if improvement in ("large", "moderate", "minor") else "uncertain",
        },
        failure_mode_risks=new_liability_flags,
    )

    return CandidateEvidencePacket(
        candidate_id=candidate_id,
        parent_inchi_key=smiles_to_inchi_key(parent_smiles) or "",
        canonical_smiles=candidate_smiles,
        inchi_key=smiles_to_inchi_key(candidate_smiles) or "",
        structural=structural,
        descriptors=cand_desc,
        descriptor_deltas=deltas,
        activity_retention=activity_retention,
        liability=liability,
        risk=risk,
        structured_rationale=rationale,
        proposer_sources=proposer_sources,
    )
