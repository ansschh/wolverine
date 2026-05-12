"""Forward reaction validator (RETRO_PLAN R-3, Lock 5 of RETRO.md).

Given a proposed RetroStep (product, precursors, reaction_class), runs
the forward model to predict the product from the precursors. Compares
against the original product:
  - exact canonical SMILES match -> pass_rule='exact_match', pass=True
  - Morgan-FP Tanimoto >= 0.95 -> pass_rule='tanimoto>=0.95', pass=True
  - otherwise -> pass_rule='fail', pass=False

The actual forward model is a SMILES seq2seq trained by
scripts/train_retro_forward.py; its checkpoint stores a callable
`forward_predict(reactants_smiles, reaction_class_hint=None)` that
returns the top-1 predicted product SMILES.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasyn.synth.retro.reactions import canonicalize_smiles, inchi_key_from_smiles
from rasyn.synth.retro.schemas import (
    ForwardValidationResult,
    ReactionClass,
    RetroStep,
)


@dataclass
class ForwardValidatorConfig:
    checkpoint_path: Path | None = None
    tanimoto_threshold: float = 0.95
    device: str = "cpu"


class ForwardValidator:
    """Wraps the forward-reaction seq2seq model + comparison logic."""

    def __init__(self, cfg: ForwardValidatorConfig):
        self.cfg = cfg
        self._model = None
        if cfg.checkpoint_path and cfg.checkpoint_path.exists():
            self._model = self._load(cfg.checkpoint_path)

    def _load(self, ckpt_path: Path):
        try:
            import torch
        except ImportError:
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("forward_predict")  # callable hook written by trainer

    def _predict_product(self, reactants_smiles: list[str], rc: ReactionClass | None) -> str | None:
        """Return predicted product SMILES, or None if no model available."""
        if self._model is None or not callable(self._model):
            return None
        try:
            return self._model(reactants_smiles, reaction_class_hint=rc)
        except Exception:
            return None

    def _tanimoto(self, smi_a: str, smi_b: str) -> float:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.DataStructs import TanimotoSimilarity
        except ImportError:
            return 0.0
        ma = Chem.MolFromSmiles(smi_a)
        mb = Chem.MolFromSmiles(smi_b)
        if ma is None or mb is None:
            return 0.0
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
        return float(TanimotoSimilarity(fa, fb))

    def validate_step(
        self,
        step: RetroStep,
        precursor_smiles: list[str],
        target_smiles: str,
    ) -> ForwardValidationResult:
        """Run forward + classify pass rule."""
        target_canon = canonicalize_smiles(target_smiles) or target_smiles
        predicted = self._predict_product(precursor_smiles, step.reaction_class)
        if predicted is None:
            return ForwardValidationResult(
                retro_step_id=step.retro_step_id,
                forward_predicted_product_smiles="",
                forward_predicted_inchi_key="A" * 14 + "-" + "B" * 10 + "-N",
                tanimoto_to_target=0.0,
                canonical_smiles_match=False,
                pass_rule="fail",
            )
        pred_canon = canonicalize_smiles(predicted) or predicted
        pred_ik = inchi_key_from_smiles(pred_canon) or ("A" * 14 + "-" + "B" * 10 + "-N")
        exact = pred_canon == target_canon
        tan = 1.0 if exact else self._tanimoto(pred_canon, target_canon)
        if exact:
            rule = "exact_match"
        elif tan >= self.cfg.tanimoto_threshold:
            rule = "tanimoto>=0.95"
        else:
            rule = "fail"
        return ForwardValidationResult(
            retro_step_id=step.retro_step_id,
            forward_predicted_product_smiles=pred_canon,
            forward_predicted_inchi_key=pred_ik,
            tanimoto_to_target=tan,
            canonical_smiles_match=exact,
            pass_rule=rule,
        )
