"""Round-trip tests for every frozen schema.

Ensures: Pydantic model -> JSON -> Pydantic model -> equal hash.
"""

from __future__ import annotations

from rasyn.schemas.challenge import (
    ActivityContext,
    ADMETChallengePacket,
    LiabilityContext,
    RescueContextPacket,
)
from rasyn.schemas.config import (
    BaselineConfig,
    DecontaminationConfig,
    ProposerConfig,
    RankerConfig,
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
from rasyn.schemas.hashing import hash_model
from rasyn.schemas.molecule import MoleculeRef
from rasyn.schemas.proposer import (
    CandidateAnnotation,
    ProposerOutput,
    ProposerRequest,
    TransformationDescriptor,
)
from rasyn.schemas.ranker import ConfidenceBlock, RankerInput, RankerOutput


def _round_trip(model):
    cls = type(model)
    j = model.model_dump_json()
    rebuilt = cls.model_validate_json(j)
    assert hash_model(model) == hash_model(rebuilt), f"{cls.__name__} hash drift after round-trip"
    return rebuilt


def test_molecule_ref_round_trip():
    m = MoleculeRef(
        name="terfenadine",
        canonical_smiles="C(C)(C)c1ccccc1",  # placeholder
        inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        pubchem_cid="5405",
    )
    _round_trip(m)


def test_molecule_ref_inchi_key_validator_rejects_bad_shape():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MoleculeRef(name="x", inchi_key="too-short")


def test_molecule_ref_unpopulated_is_allowed():
    m = MoleculeRef(name="OXS007570")
    assert not m.is_populated
    _round_trip(m)


def test_admet_challenge_packet_round_trip():
    pkt = ADMETChallengePacket(
        case_id="ADMET-001",
        parent_canonical_smiles="CC(C)(C)c1ccccc1",
        parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        activity_context=ActivityContext(
            target_name="H1 receptor",
            target_chembl_id="CHEMBL231",
            desired_pharmacology="H1 antagonism",
            parent_potency_value=10.0,
            parent_potency_unit="nM",
            parent_potency_endpoint="IC50",
        ),
        liability_context=LiabilityContext(
            liability_type="hERG",
            measurement_endpoint="hERG IC50",
            parent_value=200.0,
            parent_unit="nM",
            parent_category="high",
            target_improvement_category="low",
        ),
        rescue_context=RescueContextPacket(
            rescue_mode="active_metabolite_safety_rescue",
            constraints=["preserve H1 binding within 10x"],
        ),
    )
    _round_trip(pkt)


def test_proposer_round_trip():
    cand = CandidateAnnotation(
        candidate_id="cand-0001",
        canonical_smiles="CC(C)(C)c1ccccc1",
        inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        proposer_sources=["analog_retrieval", "mmp_transformer"],
        transformation=TransformationDescriptor(
            transformation_class="bioisostere_replacement",
            summary="phenyl -> pyridyl",
            transformation_distance=1,
        ),
        proposer_confidence=0.7,
    )
    out = ProposerOutput(
        case_id="ADMET-001",
        channel="analog_retrieval",
        candidates=[cand],
        raw_count=12_345,
        invalid_count=23,
        deduplicated_count=8_901,
    )
    _round_trip(out)
    req = ProposerRequest(
        case_id="ADMET-001",
        challenge_packet_hash="0" * 64,
        channels=["analog_retrieval", "mmp_transformer"],
    )
    _round_trip(req)


def _make_evidence_packet() -> CandidateEvidencePacket:
    return CandidateEvidencePacket(
        candidate_id="cand-0001",
        parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        canonical_smiles="CC(C)(C)c1ccccc1",
        inchi_key="CCCCCCCCCCCCCC-DDDDDDDDDD-N",
        structural=StructuralEvidence(
            tanimoto_to_parent=0.78,
            murcko_scaffold_match=True,
            transformation_distance=2,
        ),
        descriptors=DescriptorBlock(
            mw=350.4,
            log_p=4.2,
            tpsa=42.3,
            hbd=2,
            hba=4,
            rotatable_bonds=6,
            aromatic_rings=3,
            fsp3=0.2,
            formal_charge=0,
        ),
        descriptor_deltas=DescriptorDeltas(
            delta_mw=44.0,
            delta_log_p=-1.3,
            delta_tpsa=21.5,
            delta_hbd=1,
            delta_hba=1,
            delta_rotatable_bonds=0,
            delta_aromatic_rings=0,
            delta_fsp3=0.0,
            delta_formal_charge=-1,
        ),
        activity_retention=ActivityRetentionEvidence(
            predicted_retention_bucket="acceptable",
            predicted_retention_confidence=0.65,
        ),
        liability=LiabilityEvidence(
            liability_drivers_in_parent=["high_logP", "tertiary_amine"],
            candidate_changes_affecting_liability=["polarity_increase", "remove_basic_amine"],
            predicted_improvement_category="moderate",
            predicted_improvement_confidence=0.6,
        ),
        risk=RiskEvidence(
            new_liability_flags=[],
            synthesizability_score=2.4,
        ),
        structured_rationale=StructuredRationale(
            transformation_class="polarity_increase",
            expected_delta_direction={"hERG": "decrease", "logP": "decrease", "activity": "retain"},
        ),
        proposer_sources=["mmp_transformer"],
    )


def test_evidence_packet_round_trip():
    _round_trip(_make_evidence_packet())


def test_ranker_round_trip():
    ev = _make_evidence_packet()
    inp = RankerInput(
        case_id="ADMET-001",
        challenge_packet_hash="0" * 64,
        parent_canonical_smiles="CC(C)(C)c1ccccc1",
        parent_inchi_key="AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        candidate_evidence=ev,
    )
    _round_trip(inp)
    out = RankerOutput(
        case_id="ADMET-001",
        candidate_id="cand-0001",
        rescue_score=0.72,
        rescue_label_probs={
            "strong_success": 0.4,
            "weak_success": 0.3,
            "failed_activity_loss": 0.1,
            "failed_no_liability_improvement": 0.05,
            "failed_wrong_liability": 0.02,
            "failed_new_liability": 0.03,
            "uncertain": 0.1,
        },
        failure_mode_probs={
            "activity_lost": 0.1,
            "liability_not_fixed": 0.05,
            "wrong_liability_improved": 0.02,
            "new_liability_introduced": 0.03,
            "implausible_chemistry": 0.01,
            "uncertain": 0.1,
        },
        activity_retention_pred=ev.activity_retention,
        liability_improvement_pred=ev.liability,
        structured_rationale=ev.structured_rationale,
        confidence=ConfidenceBlock(
            overall=0.7,
            activity_retention=0.65,
            liability_improvement=0.6,
            new_risk=0.8,
            chemical_plausibility=0.85,
        ),
        rank=3,
    )
    _round_trip(out)


def test_configs_round_trip():
    for cfg in [DecontaminationConfig(), BaselineConfig(), ProposerConfig(), RankerConfig()]:
        _round_trip(cfg)


def test_config_hashes_are_distinct():
    a = hash_model(DecontaminationConfig())
    b = hash_model(DecontaminationConfig(strictness="loose"))
    assert a != b
