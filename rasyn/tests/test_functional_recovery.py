"""Functional-recovery scorer tests."""

from __future__ import annotations

from rasyn.eval.functional_recovery import (
    FunctionalCriteria,
    default_criteria_for_packet,
    passes_functional_criteria,
)
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


def _ev(cid: str, *, retention: str = "acceptable", improvement: str = "moderate") -> CandidateEvidencePacket:
    return CandidateEvidencePacket(
        candidate_id=cid,
        parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        canonical_smiles="CCO",
        inchi_key="CCCCCCCCCCCCCC-DDDDDDDDDD-N",
        structural=StructuralEvidence(tanimoto_to_parent=0.5, murcko_scaffold_match=True, transformation_distance=2),
        descriptors=DescriptorBlock(mw=200, log_p=3, tpsa=50, hbd=1, hba=2, rotatable_bonds=2, aromatic_rings=1, fsp3=0.4, formal_charge=0),
        descriptor_deltas=DescriptorDeltas(delta_mw=20, delta_log_p=-1, delta_tpsa=10, delta_hbd=1, delta_hba=1, delta_rotatable_bonds=0, delta_aromatic_rings=0, delta_fsp3=0.1, delta_formal_charge=0),
        activity_retention=ActivityRetentionEvidence(predicted_retention_bucket=retention, predicted_retention_confidence=0.6),
        liability=LiabilityEvidence(liability_drivers_in_parent=["high_logP"], candidate_changes_affecting_liability=["polarity_increase"], predicted_improvement_category=improvement, predicted_improvement_confidence=0.6),
        risk=RiskEvidence(),
        structured_rationale=StructuredRationale(),
        proposer_sources=["analog_retrieval"],
    )


def test_default_criteria_uses_packet_mode():
    from rasyn.synth.fixture import build_synthetic_fixture

    packets, _ = build_synthetic_fixture()
    p = packets["SYNTH-HERG-001"]
    crit = default_criteria_for_packet(p)
    assert crit.rescue_mode == p.rescue_context.rescue_mode


def test_passes_with_meeting_thresholds():
    crit = FunctionalCriteria(
        rescue_mode="active_metabolite_safety_rescue",
        min_retention_bucket_rank=3,
        min_improvement_category_rank=3,
    )
    ev = _ev("a", retention="strong", improvement="large")
    assert passes_functional_criteria(ev, crit) is True


def test_fails_on_low_retention():
    crit = FunctionalCriteria(
        rescue_mode="active_metabolite_safety_rescue",
        min_retention_bucket_rank=3,
        min_improvement_category_rank=3,
    )
    ev = _ev("a", retention="weak", improvement="large")
    assert passes_functional_criteria(ev, crit) is False


def test_fails_on_low_improvement():
    crit = FunctionalCriteria(
        rescue_mode="active_metabolite_safety_rescue",
        min_retention_bucket_rank=3,
        min_improvement_category_rank=3,
    )
    ev = _ev("a", retention="strong", improvement="none")
    assert passes_functional_criteria(ev, crit) is False
