"""Baseline interface + ranking-shape tests (chemistry-free).

These tests use hand-constructed CandidateEvidencePackets so they run without
RDKit. They verify ranking interface, not chemical correctness.
"""

from __future__ import annotations

from rasyn.baselines import ALL_BASELINES, get_baseline
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


def _ev(cid: str, *, sim: float = 0.5, log_p: float = 3.0, tpsa: float = 50.0,
        retention: str = "acceptable", improvement: str = "moderate", tc: str | None = None) -> CandidateEvidencePacket:
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
        structured_rationale=StructuredRationale(transformation_class=tc),
        proposer_sources=["analog_retrieval"],
    )


def test_all_eight_baselines_registered():
    names = [c.name for c in ALL_BASELINES]
    assert set(names) == {
        "random",
        "similarity_only",
        "most_polar",
        "liability_only_property",
        "activity_only",
        "weighted_property",
        "mmp_frequency",
        "medchem_heuristic",
    }


def test_get_baseline_known_and_unknown():
    import pytest

    assert get_baseline("random").name == "random"
    with pytest.raises(ValueError):
        get_baseline("nonexistent_baseline")


def test_baseline_returns_descending_score():
    cs = [_ev("a", sim=0.8), _ev("b", sim=0.5), _ev("c", sim=0.9)]
    out = get_baseline("similarity_only").score("CCO", cs, "hERG")
    ranked = [cid for cid, _ in out]
    assert ranked == ["c", "a", "b"]


def test_random_baseline_is_deterministic_with_seed():
    cs = [_ev("a"), _ev("b"), _ev("c")]
    o1 = get_baseline("random", seed=7).score("CCO", cs, "hERG")
    o2 = get_baseline("random", seed=7).score("CCO", cs, "hERG")
    assert [cid for cid, _ in o1] == [cid for cid, _ in o2]


def test_liability_only_uses_improvement_category():
    cs = [
        _ev("low", improvement="none"),
        _ev("hi", improvement="large"),
        _ev("mid", improvement="minor"),
    ]
    out = get_baseline("liability_only_property").score("CCO", cs, "hERG")
    ranked = [cid for cid, _ in out]
    assert ranked[0] == "hi"
    assert ranked[-1] == "low"


def test_activity_only_uses_retention_bucket():
    cs = [
        _ev("a", retention="failed"),
        _ev("b", retention="strong"),
        _ev("c", retention="weak"),
    ]
    out = get_baseline("activity_only").score("CCO", cs, "hERG")
    ranked = [cid for cid, _ in out]
    assert ranked[0] == "b"


def test_mmp_frequency_uses_seed_table():
    cs = [
        _ev("a", tc="phenyl_to_pyridyl"),
        _ev("b", tc="some_unknown_class"),
        _ev("c", tc=None),
    ]
    out = get_baseline("mmp_frequency").score("CCO", cs, "hERG")
    assert out[0][0] == "a"


def test_weighted_property_changes_with_liability():
    cs = [_ev("a"), _ev("b", log_p=5, tpsa=10)]
    s_herg = dict(get_baseline("weighted_property").score("CCO", cs, "hERG"))
    s_sol = dict(get_baseline("weighted_property").score("CCO", cs, "solubility"))
    # The two liabilities use different formulas, so scores must differ.
    assert s_herg != s_sol
