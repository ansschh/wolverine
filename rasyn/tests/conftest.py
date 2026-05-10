"""Shared chemistry-free fixtures + helpers for the test suite."""

from __future__ import annotations

import pytest

from rasyn.schemas.evidence import (
    ActivityRetentionEvidence,
    CandidateEvidencePacket,
    DescriptorBlock,
    DescriptorDeltas,
    LiabilityEvidence,
    RiskEvidence,
    StructuralEvidence,
    StructuredRationale,
)


def make_evidence(
    cid: str,
    *,
    sim: float = 0.5,
    log_p: float = 3.0,
    tpsa: float = 50.0,
    retention: str = "acceptable",
    improvement: str = "moderate",
    transformation_class: str | None = None,
) -> CandidateEvidencePacket:
    """Build a chemistry-free CandidateEvidencePacket for test use."""
    return CandidateEvidencePacket(
        candidate_id=cid,
        parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        canonical_smiles="CCO",
        inchi_key="CCCCCCCCCCCCCC-DDDDDDDDDD-N",
        structural=StructuralEvidence(
            tanimoto_to_parent=sim,
            murcko_scaffold_match=True,
            transformation_distance=2,
        ),
        descriptors=DescriptorBlock(
            mw=200, log_p=log_p, tpsa=tpsa, hbd=1, hba=2,
            rotatable_bonds=2, aromatic_rings=1, fsp3=0.4, formal_charge=0,
        ),
        descriptor_deltas=DescriptorDeltas(
            delta_mw=20, delta_log_p=-1.0, delta_tpsa=10, delta_hbd=1, delta_hba=1,
            delta_rotatable_bonds=0, delta_aromatic_rings=0, delta_fsp3=0.1, delta_formal_charge=0,
        ),
        activity_retention=ActivityRetentionEvidence(
            predicted_retention_bucket=retention, predicted_retention_confidence=0.6,
        ),
        liability=LiabilityEvidence(
            liability_drivers_in_parent=["high_logP"],
            candidate_changes_affecting_liability=["polarity_increase"],
            predicted_improvement_category=improvement, predicted_improvement_confidence=0.6,
        ),
        risk=RiskEvidence(),
        structured_rationale=StructuredRationale(transformation_class=transformation_class),
        proposer_sources=["analog_retrieval"],
    )


@pytest.fixture
def evidence_factory():
    """Pytest fixture: returns the make_evidence helper."""
    return make_evidence
