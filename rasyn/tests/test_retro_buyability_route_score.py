"""Tests for buyability index + route_score formula."""
from __future__ import annotations

import pytest

from rasyn.synth.retro.buyability import BuyabilityIndex, BuyabilityIndexConfig
from rasyn.synth.retro.route_score import (
    RouteScoreInputs,
    RouteScoreWeights,
    compute_route_score_inputs,
    route_score,
)
from rasyn.synth.retro.schemas import (
    BuyabilityRecord,
    ForwardValidationResult,
    RetroStep,
)


# ===== BuyabilityIndex =====

def test_buyability_index_empty():
    idx = BuyabilityIndex(BuyabilityIndexConfig())
    assert len(idx) == 0
    assert idx.is_buyable("AAAAAAAAAAAAAA-BBBBBBBBBB-N") is False


def test_buyability_index_add_and_lookup():
    idx = BuyabilityIndex(BuyabilityIndexConfig())
    rec = BuyabilityRecord(
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        canonical_smiles="CCO",
        inventory_sources=["ZINC22"],
        cost_tier="tier1",
        cost_per_g_usd=0.5,
        snapshot_date="2026-05-12",
    )
    idx.add_record(rec)
    assert len(idx) == 1
    assert idx.is_buyable("LFQSCWFLJHTTHZ-UHFFFAOYSA-N") is True
    assert idx.lookup("LFQSCWFLJHTTHZ-UHFFFAOYSA-N") == rec


def test_buyability_tier1_only_filters_tier2():
    idx = BuyabilityIndex(BuyabilityIndexConfig(tier1_only=True))
    tier1 = BuyabilityRecord(
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        canonical_smiles="CCO", inventory_sources=["ZINC22"],
        cost_tier="tier1", snapshot_date="2026-05-12",
    )
    tier2 = BuyabilityRecord(
        inchi_key="WETWJCDKMRHUPV-UHFFFAOYSA-N",
        canonical_smiles="CC(=O)Cl", inventory_sources=["ZINC22"],
        cost_tier="tier2", snapshot_date="2026-05-12",
    )
    idx.add_record(tier1)
    idx.add_record(tier2)
    assert idx.is_buyable("LFQSCWFLJHTTHZ-UHFFFAOYSA-N") is True
    assert idx.is_buyable("WETWJCDKMRHUPV-UHFFFAOYSA-N") is False  # filtered


# ===== RouteScoreWeights =====

def test_route_score_weights_default_match_plan_l6():
    """RETRO_PLAN L6 fixed weights: 0.4 / 0.3 / 0.1 / 0.1 / 0.1."""
    w = RouteScoreWeights()
    assert w.plausibility_product == 0.4
    assert w.forward_pass_rate == 0.3
    assert w.step_count == 0.1
    assert w.cost == 0.1
    assert w.risk_flags == 0.1


def test_route_score_perfect_route():
    """A perfect route: plausibility=1, forward_pass=1, step_count=0, cost=0, risk=0 -> score=0.7"""
    inputs = RouteScoreInputs(
        step_plausibility_product=1.0,
        forward_pass_rate=1.0,
        step_count_norm=0.0,
        cost_norm=0.0,
        risk_flags_norm=0.0,
    )
    assert route_score(inputs) == pytest.approx(0.7)


def test_route_score_penalises_step_count():
    inputs = RouteScoreInputs(
        step_plausibility_product=1.0,
        forward_pass_rate=1.0,
        step_count_norm=1.0,
        cost_norm=0.0,
        risk_flags_norm=0.0,
    )
    # 0.4 + 0.3 - 0.1 - 0 - 0 = 0.6
    assert route_score(inputs) == pytest.approx(0.6)


def test_route_score_penalises_risk_flags():
    inputs = RouteScoreInputs(
        step_plausibility_product=1.0,
        forward_pass_rate=1.0,
        step_count_norm=0.0,
        cost_norm=0.0,
        risk_flags_norm=1.0,
    )
    assert route_score(inputs) == pytest.approx(0.6)


def test_compute_route_score_inputs_simple():
    step1 = RetroStep(
        retro_step_id="S1",
        product_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        precursor_inchi_keys=["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        reaction_class="amide_coupling",
        proposed_by_channel="template",
        proposed_by_top_k_rank=1,
        confidence=0.9,
    )
    step2 = step1.model_copy(update={"retro_step_id": "S2", "confidence": 0.8})
    fvr1 = ForwardValidationResult(
        retro_step_id="S1",
        forward_predicted_product_smiles="CCOC(C)=O",
        forward_predicted_inchi_key="XEKOWRVHYACXOJ-UHFFFAOYSA-N",
        tanimoto_to_target=1.0,
        canonical_smiles_match=True,
        pass_rule="exact_match",
    )
    fvr2 = fvr1.model_copy(update={"retro_step_id": "S2", "pass_rule": "fail",
                                     "canonical_smiles_match": False, "tanimoto_to_target": 0.4})
    rec = BuyabilityRecord(
        inchi_key="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        canonical_smiles="CCO",
        inventory_sources=["ZINC22"],
        cost_tier="tier1",
        cost_per_g_usd=0.5,
        snapshot_date="2026-05-12",
    )

    inputs = compute_route_score_inputs(
        steps=[step1, step2],
        fvr_list=[fvr1, fvr2],
        buyables_records=[rec, None],
        max_steps=8,
    )
    assert inputs.step_plausibility_product == pytest.approx(0.9 * 0.8)
    assert inputs.forward_pass_rate == pytest.approx(0.5)  # 1 of 2 pass
    assert inputs.step_count_norm == pytest.approx(2 / 8)


def test_compute_route_score_inputs_no_steps():
    inputs = compute_route_score_inputs(
        steps=[],
        fvr_list=[],
        buyables_records=[],
        max_steps=8,
    )
    assert inputs.step_plausibility_product == 0.0
    assert inputs.forward_pass_rate == 0.0
