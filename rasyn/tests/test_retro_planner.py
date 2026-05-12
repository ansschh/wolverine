"""Smoke tests for the Retro* planner with synthetic proposers.

These tests don't load real checkpoints — they wire mock proposers
that return fixed precursor sets so we can exercise the AND-OR search
end-to-end.
"""
from __future__ import annotations

from typing import Any

import pytest

from rasyn.synth.retro.buyability import BuyabilityIndex, BuyabilityIndexConfig
from rasyn.synth.retro.conditions import ConditionsPredictor, ConditionsPredictorConfig
from rasyn.synth.retro.planner import PlannerConfig, RetroPlanner
from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.schemas import (
    BuyabilityRecord,
    ProposerOutput,
)
from rasyn.synth.retro.validator import ForwardValidator, ForwardValidatorConfig
from rasyn.synth.retro.value_model import ValueModel, ValueModelConfig


class _MockProposer(RetroProposer):
    """A mock proposer that returns fixed (precursors, conf) per target."""

    channel = "template"  # type: ignore[assignment]

    def __init__(self, mapping: dict[str, list[tuple[list[str], list[str], float, str]]]):
        self.mapping = mapping

    def propose(self, target_smiles, target_inchi_key, *, top_k=10, **kwargs):  # type: ignore[override]
        entries = self.mapping.get(target_smiles, [])
        candidates = [e[0] for e in entries][:top_k]
        candidate_smiles = [e[1] for e in entries][:top_k]
        confs = [e[2] for e in entries][:top_k]
        classes = [e[3] for e in entries][:top_k]
        return ProposerOutput(
            channel="template",
            target_inchi_key=target_inchi_key,
            target_smiles=target_smiles,
            candidates=candidates,
            candidate_smiles=candidate_smiles,
            confidences=confs,
            reaction_class_predictions=classes,
        )


def _make_buyability_idx(buyable_records):
    idx = BuyabilityIndex(BuyabilityIndexConfig())
    for rec in buyable_records:
        idx.add_record(rec)
    return idx


def test_planner_target_is_already_buyable():
    target_smi = "CCO"
    target_ik = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    idx = _make_buyability_idx([BuyabilityRecord(
        inchi_key=target_ik, canonical_smiles="CCO", inventory_sources=["ZINC22"],
        cost_tier="tier1", snapshot_date="2026-05-12",
    )])
    proposer = _MockProposer({})
    planner = RetroPlanner(
        proposers=[proposer],
        buyability=idx,
        value_model=ValueModel(ValueModelConfig()),
        validator=ForwardValidator(ForwardValidatorConfig()),
        conditions=ConditionsPredictor(ConditionsPredictorConfig()),
        cfg=PlannerConfig(max_steps=4, max_iterations=10, time_budget_s=5),
    )
    trees = planner.plan(target_smi, target_ik)
    assert len(trees) == 1
    assert trees[0].all_leaves_buyable is True
    assert trees[0].step_count == 0


def test_planner_one_step_to_buyable():
    """Target = CCOC(C)=O, disconnects to (CCO, CC(=O)Cl), both buyable."""
    target_smi = "CCOC(C)=O"
    target_ik = "XEKOWRVHYACXOJ-UHFFFAOYSA-N"
    bb1_ik = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    bb2_ik = "WETWJCDKMRHUPV-UHFFFAOYSA-N"

    idx = _make_buyability_idx([
        BuyabilityRecord(inchi_key=bb1_ik, canonical_smiles="CCO",
                          inventory_sources=["ZINC22"], cost_tier="tier1", snapshot_date="2026-05-12"),
        BuyabilityRecord(inchi_key=bb2_ik, canonical_smiles="CC(=O)Cl",
                          inventory_sources=["ZINC22"], cost_tier="tier1", snapshot_date="2026-05-12"),
    ])
    proposer = _MockProposer({
        target_smi: [
            ([bb1_ik, bb2_ik], ["CCO", "CC(=O)Cl"], 0.9, "amide_coupling"),
        ],
    })

    # Forward validator that passes everything (mock)
    validator = ForwardValidator(ForwardValidatorConfig())
    validator._model = lambda reactants_smiles, reaction_class_hint=None: "CCOC(C)=O"

    planner = RetroPlanner(
        proposers=[proposer],
        buyability=idx,
        value_model=ValueModel(ValueModelConfig()),
        validator=validator,
        conditions=ConditionsPredictor(ConditionsPredictorConfig()),
        cfg=PlannerConfig(max_steps=4, max_iterations=20, time_budget_s=10,
                          require_forward_validation=False),
    )
    trees = planner.plan(target_smi, target_ik)
    assert len(trees) >= 1
    tree = trees[0]
    assert tree.step_count >= 1
    # All OR_molecule leaves should be buyable
    or_leaves = [n for n in tree.nodes if n.node_type == "OR_molecule" and not n.children_node_ids]
    if or_leaves:
        for leaf in or_leaves:
            assert leaf.is_buyable, f"leaf {leaf.molecule_smiles} not buyable"


def test_planner_returns_empty_when_unreachable():
    """Target with no proposer entry + not buyable -> no route."""
    target_smi = "CC(C)(C)CCCCCCCCC"  # random hydrocarbon
    target_ik = "AAAAAAAAAAAAAA-BBBBBBBBBB-N"
    idx = _make_buyability_idx([])
    proposer = _MockProposer({})
    planner = RetroPlanner(
        proposers=[proposer],
        buyability=idx,
        value_model=ValueModel(ValueModelConfig()),
        validator=ForwardValidator(ForwardValidatorConfig()),
        conditions=ConditionsPredictor(ConditionsPredictorConfig()),
        cfg=PlannerConfig(max_steps=2, max_iterations=5, time_budget_s=2),
    )
    trees = planner.plan(target_smi, target_ik)
    assert trees == []
