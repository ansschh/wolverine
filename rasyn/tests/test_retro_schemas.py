"""Round-trip + invariant tests for Rasyn-Retro schemas (RETRO_PLAN R-0)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rasyn.synth.retro.schemas import (
    BuyabilityRecord,
    CandidateRoute,
    ConditionPrediction,
    ForwardValidationResult,
    Molecule,
    ProposerOutput,
    Reaction,
    RetroStep,
    RouteRationale,
    RouteTree,
    RouteTreeNode,
)


# ===== Molecule =====

def test_molecule_minimal_construct():
    m = Molecule(
        canonical_smiles="CCO",
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    )
    assert m.canonical_smiles == "CCO"
    assert m.inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert m.commercial_availability is False
    assert m.cost_tier == "unknown"


def test_molecule_rejects_bad_inchi_key():
    with pytest.raises(ValidationError):
        Molecule(canonical_smiles="CCO", inchi_key="too-short")


def test_molecule_rejects_whitespace_in_smiles():
    with pytest.raises(ValidationError):
        Molecule(canonical_smiles="CCO CCN", inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N")


def test_molecule_frozen_extra_forbid():
    m = Molecule(canonical_smiles="CCO", inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    with pytest.raises(ValidationError):
        Molecule(
            canonical_smiles="CCO",
            inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            unknown_field="x",
        )
    with pytest.raises(Exception):
        m.canonical_smiles = "CCN"  # frozen


def test_molecule_inchi_key_uppercased():
    m = Molecule(canonical_smiles="CCO", inchi_key="lfqscwfljhtthz-uhfffaoysa-n")
    assert m.inchi_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


# ===== Reaction =====

def test_reaction_basic():
    r = Reaction(
        reaction_id="RXN-0001",
        reactant_smiles=["CCO", "CC(=O)Cl"],
        product_smiles="CCOC(C)=O",
        reactant_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "WETWJCDKMRHUPV-UHFFFAOYSA-N"],
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        source="uspto_full",
    )
    assert r.quality_tier == "bronze"
    assert r.reaction_class == "unclassified"
    assert r.reported_failed is False


def test_reaction_yield_must_be_in_range():
    with pytest.raises(ValidationError):
        Reaction(
            reaction_id="RXN-0002",
            reactant_smiles=["CCO"],
            product_smiles="CCN",
            reactant_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
            product_inchi_key="QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            source="uspto_full",
            yield_pct=150.0,
        )


def test_reaction_rejects_bad_product_inchi_key():
    with pytest.raises(ValidationError):
        Reaction(
            reaction_id="RXN-0003",
            reactant_smiles=["CCO"],
            product_smiles="CCN",
            reactant_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
            product_inchi_key="bad",
            source="uspto_full",
        )


# ===== RetroStep =====

def test_retro_step_confidence_clamped():
    with pytest.raises(ValidationError):
        RetroStep(
            retro_step_id="STEP-0001",
            product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
            precursor_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
            reaction_class="amide_coupling",
            proposed_by_channel="template",
            proposed_by_top_k_rank=1,
            confidence=1.5,
        )


def test_retro_step_basic_ok():
    s = RetroStep(
        retro_step_id="STEP-0002",
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        precursor_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        reaction_class="amide_coupling",
        proposed_by_channel="template",
        proposed_by_top_k_rank=1,
        confidence=0.7,
    )
    assert s.forward_validation_pass is False


# ===== ForwardValidationResult =====

def test_forward_validation_result():
    fvr = ForwardValidationResult(
        retro_step_id="STEP-0002",
        forward_predicted_product_smiles="CCOC(C)=O",
        forward_predicted_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        tanimoto_to_target=0.99,
        canonical_smiles_match=True,
        pass_rule="exact_match",
    )
    assert fvr.canonical_smiles_match is True


# ===== ConditionPrediction =====

def test_condition_prediction_ok():
    cp = ConditionPrediction(
        reactant_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        reaction_class="amide_coupling",
        solvent_class="DMF",
        catalyst_class="none",
        temperature_bin="rt",
        reagent_classes=["HATU_HBTU_family", "base_TEA_DIPEA"],
        overall_confidence=0.62,
    )
    assert cp.solvent_class == "DMF"
    assert cp.overall_confidence == 0.62


# ===== ProposerOutput =====

def test_proposer_output_unified_shape():
    po = ProposerOutput(
        channel="template",
        target_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        target_smiles="CCOC(C)=O",
        candidates=[
            ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "WETWJCDKMRHUPV-UHFFFAOYSA-N"],
            ["XEKOWRVHYACXOJ-UHFFFAOYSA-N"],
        ],
        candidate_smiles=[
            ["CCO", "CC(=O)Cl"],
            ["CCOC(C)=O"],
        ],
        confidences=[0.9, 0.05],
        reaction_class_predictions=["amide_coupling", "unclassified"],
    )
    assert po.channel == "template"
    assert len(po.candidates) == len(po.confidences) == len(po.candidate_smiles)


# ===== BuyabilityRecord =====

def test_buyability_record_basic():
    b = BuyabilityRecord(
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        canonical_smiles="CCO",
        inventory_sources=["ZINC22", "Enamine_REAL_BB"],
        cost_tier="tier1",
        cost_per_g_usd=0.5,
        catalog_id="EN300-19142",
        snapshot_date="2026-05-12",
    )
    assert b.cost_tier == "tier1"


def test_buyability_record_rejects_bad_inchi():
    with pytest.raises(ValidationError):
        BuyabilityRecord(
            inchi_key="bad",
            canonical_smiles="CCO",
            inventory_sources=["ZINC22"],
            cost_tier="tier1",
            snapshot_date="2026-05-12",
        )


# ===== RouteTree + invariants =====

def _make_one_step_tree() -> RouteTree:
    target_ikey = "XEKOWRVHYACXOJ-UHFFFAOYSA-N"
    bb1_ikey = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    bb2_ikey = "WETWJCDKMRHUPV-UHFFFAOYSA-N"
    step = RetroStep(
        retro_step_id="STEP-T1",
        product_inchi_key=target_ikey,
        precursor_inchi_keys=[bb1_ikey, bb2_ikey],
        reaction_class="amide_coupling",
        proposed_by_channel="template",
        proposed_by_top_k_rank=1,
        confidence=0.9,
        forward_validation_pass=True,
        forward_tanimoto_to_target=0.98,
    )
    root = RouteTreeNode(
        node_id="N0",
        node_type="OR_molecule",
        molecule_inchi_key=target_ikey,
        molecule_smiles="CCOC(C)=O",
        is_buyable=False,
        children_node_ids=["N1"],
        depth=0,
        expanded=True,
    )
    step_node = RouteTreeNode(
        node_id="N1",
        node_type="AND_step",
        retro_step=step,
        children_node_ids=["N2", "N3"],
        depth=1,
    )
    leaf1 = RouteTreeNode(
        node_id="N2",
        node_type="OR_molecule",
        molecule_inchi_key=bb1_ikey,
        molecule_smiles="CCO",
        is_buyable=True,
        buyability_record_inchi_key=bb1_ikey,
        depth=2,
        expanded=True,
    )
    leaf2 = RouteTreeNode(
        node_id="N3",
        node_type="OR_molecule",
        molecule_inchi_key=bb2_ikey,
        molecule_smiles="CC(=O)Cl",
        is_buyable=True,
        buyability_record_inchi_key=bb2_ikey,
        depth=2,
        expanded=True,
    )
    tree = RouteTree(
        tree_id="TREE-1",
        target_inchi_key=target_ikey,
        target_smiles="CCOC(C)=O",
        nodes=[root, step_node, leaf1, leaf2],
        root_node_id="N0",
        step_count=1,
        longest_linear_sequence=1,
        all_leaves_buyable=True,
        purchasable_fraction=1.0,
        risk_score=0.1,
    )
    return tree


def test_route_tree_one_step_invariants():
    tree = _make_one_step_tree()
    assert tree.step_count == 1
    assert tree.all_leaves_buyable is True
    # Every AND_step's precursor count matches its node children count
    for node in tree.nodes:
        if node.node_type == "AND_step":
            assert len(node.retro_step.precursor_inchi_keys) == len(node.children_node_ids)
    # Every leaf OR_molecule with is_buyable=True has a buyability_record_inchi_key set
    for node in tree.nodes:
        if node.node_type == "OR_molecule" and node.is_buyable:
            assert node.buyability_record_inchi_key is not None


def test_candidate_route_full_construct():
    tree = _make_one_step_tree()
    rationale = RouteRationale(
        key_disconnections=["amide_coupling"],
        precedent_support_reaction_ids=["RXN-1234"],
        risk_flags=[],
        forward_model_recovered_target=True,
        condition_prediction_available=True,
        buyables_coverage_pct=100.0,
    )
    cond = ConditionPrediction(
        reactant_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "WETWJCDKMRHUPV-UHFFFAOYSA-N"],
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        reaction_class="amide_coupling",
        solvent_class="DMF",
        catalyst_class="none",
        temperature_bin="rt",
        reagent_classes=["HATU_HBTU_family"],
        overall_confidence=0.7,
    )
    fvr = ForwardValidationResult(
        retro_step_id="STEP-T1",
        forward_predicted_product_smiles="CCOC(C)=O",
        forward_predicted_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        tanimoto_to_target=0.99,
        canonical_smiles_match=True,
        pass_rule="exact_match",
    )
    cr = CandidateRoute(
        candidate_route_id="CAND-001",
        target_inchi_key=tree.target_inchi_key,
        target_smiles=tree.target_smiles,
        route_tree=tree,
        step_predictions=[tree.nodes[1].retro_step],
        forward_validation_results=[fvr],
        condition_predictions=[cond],
        route_score=0.72,
        step_plausibility_product=0.9,
        forward_pass_rate=1.0,
        step_count_norm=0.125,
        cost_norm=0.1,
        risk_flags_norm=0.0,
        rationale=rationale,
    )
    assert cr.route_score == 0.72
    assert cr.rationale.forward_model_recovered_target is True


def test_route_score_components_clamped():
    tree = _make_one_step_tree()
    rationale = RouteRationale(
        key_disconnections=["amide_coupling"],
        risk_flags=[],
        forward_model_recovered_target=True,
        condition_prediction_available=True,
        buyables_coverage_pct=100.0,
    )
    fvr = ForwardValidationResult(
        retro_step_id="STEP-T1",
        forward_predicted_product_smiles="CCOC(C)=O",
        forward_predicted_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        tanimoto_to_target=0.99,
        canonical_smiles_match=True,
        pass_rule="exact_match",
    )
    with pytest.raises(ValidationError):
        CandidateRoute(
            candidate_route_id="CAND-bad",
            target_inchi_key=tree.target_inchi_key,
            target_smiles=tree.target_smiles,
            route_tree=tree,
            step_predictions=[],
            forward_validation_results=[fvr],
            condition_predictions=[],
            route_score=0.5,
            step_plausibility_product=1.5,  # out of range
            forward_pass_rate=1.0,
            step_count_norm=0.0,
            cost_norm=0.0,
            risk_flags_norm=0.0,
            rationale=rationale,
        )
