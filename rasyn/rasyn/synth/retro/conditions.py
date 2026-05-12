"""Conditions predictor runtime (RETRO_PLAN R-3 Lock 7).

Given (reactants, product, reaction_class), predict coarse conditions:
  - solvent_class:    one of 30 buckets
  - catalyst_class:   one of 20 buckets
  - temperature_bin:  cryo / rt / warm / reflux / high_T / unknown
  - reagent_classes:  multi-label over ~50 reagent buckets

The model is a multi-head classifier (4 heads, all on a shared 200M-MLM
encoder of the reactant+product SMILES concatenated as 'R>>P').

A trained checkpoint stores a callable `predict_conditions(reactant_smiles_list,
product_smiles, reaction_class_hint=None) -> ConditionPrediction`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasyn.synth.retro.schemas import (
    CatalystClass,
    ConditionPrediction,
    ReactionClass,
    ReagentClass,
    SolventClass,
    TemperatureBin,
)


@dataclass
class ConditionsPredictorConfig:
    checkpoint_path: Path | None = None
    top_k_reagents: int = 5
    device: str = "cpu"


class ConditionsPredictor:
    def __init__(self, cfg: ConditionsPredictorConfig):
        self.cfg = cfg
        self._predict_fn = None
        if cfg.checkpoint_path and cfg.checkpoint_path.exists():
            self._predict_fn = self._load(cfg.checkpoint_path)

    def _load(self, ckpt_path: Path):
        try:
            import torch
        except ImportError:
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("predict_conditions")

    def predict(
        self,
        reactant_smiles_list: list[str],
        product_smiles: str,
        reactant_inchi_keys: list[str],
        product_inchi_key: str,
        reaction_class: ReactionClass | None = None,
    ) -> ConditionPrediction:
        if self._predict_fn is not None and callable(self._predict_fn):
            try:
                return self._predict_fn(
                    reactant_smiles_list, product_smiles,
                    reactant_inchi_keys=reactant_inchi_keys,
                    product_inchi_key=product_inchi_key,
                    reaction_class_hint=reaction_class,
                )
            except Exception:
                pass
        # Fallback: empty/unknown prediction so the planner can still emit
        # CandidateRoute objects with `condition_prediction_available=False`.
        return ConditionPrediction(
            reactant_inchi_keys=reactant_inchi_keys,
            product_inchi_key=product_inchi_key,
            reaction_class=reaction_class or "unclassified",
            solvent_class="unknown",
            solvent_logits={},
            catalyst_class="unknown",
            catalyst_logits={},
            temperature_bin="unknown",
            temperature_logits={},
            reagent_classes=[],
            reagent_logits={},
            overall_confidence=0.0,
        )
