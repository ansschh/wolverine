"""High-level orchestrator: given a target SMILES + constraints, return ranked CandidateRoutes.

This is the public API the sealed-case scripts and any other downstream
caller use. Internally it composes the BuyabilityIndex, ValueModel,
ForwardValidator, ConditionsPredictor, and RetroPlanner.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasyn.synth.retro.buyability import BuyabilityIndex, BuyabilityIndexConfig
from rasyn.synth.retro.conditions import ConditionsPredictor, ConditionsPredictorConfig
from rasyn.synth.retro.planner import PlannerConfig, RetroPlanner, _PlannerState
from rasyn.synth.retro.proposers import (
    DiffusionProposer, DiffusionProposerConfig,
    GraphEditProposer, GraphEditProposerConfig,
    RetrievalProposer, RetrievalProposerConfig,
    Seq2SeqProposer, Seq2SeqProposerConfig,
    TemplateProposer, TemplateProposerConfig,
)
from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.route_score import (
    RouteScoreInputs, RouteScoreWeights,
    compute_route_score_inputs,
    route_score,
)
from rasyn.synth.retro.schemas import (
    CandidateRoute,
    ForwardValidationResult,
    RetroStep,
    RouteRationale,
    RouteTree,
)
from rasyn.synth.retro.validator import ForwardValidator, ForwardValidatorConfig
from rasyn.synth.retro.value_model import ValueModel, ValueModelConfig

logger = logging.getLogger("retro.orchestrator")


@dataclass
class CheckpointPaths:
    """Where to find each component's trained checkpoint on disk.

    None means 'no checkpoint loaded; use heuristic fallback'.
    """
    template: Path | None = None
    templates_pickle: Path | None = None
    graphedit: Path | None = None
    seq2seq: Path | None = None
    retrieval_index: Path | None = None
    retrieval_metadata: Path | None = None
    diffusion: Path | None = None
    forward: Path | None = None
    conditions: Path | None = None
    value: Path | None = None
    buyables_parquet: Path | None = None


@dataclass
class OrchestratorConfig:
    planner: PlannerConfig
    ckpt: CheckpointPaths
    route_weights: RouteScoreWeights = None  # type: ignore[assignment]
    enable_proposers: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.route_weights is None:
            self.route_weights = RouteScoreWeights()
        if self.enable_proposers is None:
            self.enable_proposers = ["template", "graphedit", "seq2seq", "retrieval", "diffusion"]


class RetroOrchestrator:
    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        proposers: list[RetroProposer] = []
        if "template" in cfg.enable_proposers:
            proposers.append(TemplateProposer(TemplateProposerConfig(
                checkpoint_path=cfg.ckpt.template, templates_path=cfg.ckpt.templates_pickle,
            )))
        if "graphedit" in cfg.enable_proposers:
            proposers.append(GraphEditProposer(GraphEditProposerConfig(
                checkpoint_path=cfg.ckpt.graphedit,
            )))
        if "seq2seq" in cfg.enable_proposers:
            proposers.append(Seq2SeqProposer(Seq2SeqProposerConfig(
                checkpoint_path=cfg.ckpt.seq2seq,
            )))
        if "retrieval" in cfg.enable_proposers:
            proposers.append(RetrievalProposer(RetrievalProposerConfig(
                index_path=cfg.ckpt.retrieval_index, metadata_path=cfg.ckpt.retrieval_metadata,
            )))
        if "diffusion" in cfg.enable_proposers:
            proposers.append(DiffusionProposer(DiffusionProposerConfig(
                checkpoint_path=cfg.ckpt.diffusion,
            )))

        self.buyability = BuyabilityIndex(BuyabilityIndexConfig(
            parquet_path=cfg.ckpt.buyables_parquet,
            tier1_only=cfg.planner.tier1_only,
        ))
        self.validator = ForwardValidator(ForwardValidatorConfig(checkpoint_path=cfg.ckpt.forward))
        self.conditions = ConditionsPredictor(ConditionsPredictorConfig(checkpoint_path=cfg.ckpt.conditions))
        self.value_model = ValueModel(ValueModelConfig(checkpoint_path=cfg.ckpt.value))
        self.planner = RetroPlanner(
            proposers=proposers,
            buyability=self.buyability,
            value_model=self.value_model,
            validator=self.validator,
            conditions=self.conditions,
            cfg=cfg.planner,
        )

    def plan_routes(
        self,
        target_smiles: str,
        target_inchi_key: str,
    ) -> list[CandidateRoute]:
        trees = self.planner.plan(target_smiles, target_inchi_key)
        candidates: list[CandidateRoute] = []
        for tree in trees:
            step_predictions, fvrs, cond_preds, buyables = [], [], [], []
            for node in tree.nodes:
                if node.node_type == "AND_step" and node.retro_step is not None:
                    step_predictions.append(node.retro_step)
                    # Look up FVR / conditions from the planner's state cache
                    # which we don't expose; re-validate fresh here.
                    fvr = self.validator.validate_step(
                        node.retro_step,
                        precursor_smiles=[],  # planner already validated; placeholder
                        target_smiles=tree.target_smiles,
                    )
                    fvrs.append(fvr)
                if node.node_type == "OR_molecule" and node.is_buyable:
                    rec = self.buyability.lookup(node.molecule_inchi_key)
                    buyables.append(rec)

            inputs = compute_route_score_inputs(
                step_predictions, fvrs, buyables,
                risk_flags=[],
                max_steps=self.cfg.planner.max_steps,
            )
            score = route_score(inputs, weights=self.cfg.route_weights)

            rationale = RouteRationale(
                key_disconnections=[s.reaction_class for s in step_predictions],
                precedent_support_reaction_ids=[],
                risk_flags=[],
                forward_model_recovered_target=all(f.pass_rule != "fail" for f in fvrs) if fvrs else False,
                condition_prediction_available=any(c is not None for c in cond_preds),
                buyables_coverage_pct=100.0 * inputs.step_plausibility_product,
            )
            cand = CandidateRoute(
                candidate_route_id=f"CAND-{uuid.uuid4().hex[:8]}",
                target_inchi_key=tree.target_inchi_key,
                target_smiles=tree.target_smiles,
                route_tree=tree,
                step_predictions=step_predictions,
                forward_validation_results=fvrs,
                condition_predictions=cond_preds,
                route_score=score,
                step_plausibility_product=inputs.step_plausibility_product,
                forward_pass_rate=inputs.forward_pass_rate,
                step_count_norm=inputs.step_count_norm,
                cost_norm=inputs.cost_norm,
                risk_flags_norm=inputs.risk_flags_norm,
                rationale=rationale,
            )
            candidates.append(cand)
        candidates.sort(key=lambda c: -c.route_score)
        return candidates
