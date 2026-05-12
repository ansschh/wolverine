"""Graph-edit proposer (RETRO_PLAN R-2 Channel 2).

Bond-edit classifier on the product graph: for each bond in the product,
the model predicts an edit operation (break, modify, leave). Once the
edits are decoded, the resulting reactants are read off the modified
graph.

The model is a graph transformer (~30M params, trained from scratch
since input is a graph rather than a SMILES string). The training
script writes a callable `predict_edits` into the checkpoint that takes
a product SMILES and returns an ordered list of `(edit_op, atoms,
bond_type, log_prob)` predictions plus the resulting precursor SMILES.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.reactions import canonicalize_smiles, inchi_key_from_smiles
from rasyn.synth.retro.schemas import ProposerChannel, ProposerOutput, ReactionClass


@dataclass
class GraphEditProposerConfig:
    checkpoint_path: Path | None = None
    top_k_edits: int = 20
    device: str = "cpu"


class GraphEditProposer(RetroProposer):
    channel: ProposerChannel = "graphedit"

    def __init__(self, cfg: GraphEditProposerConfig):
        self.cfg = cfg
        self._predictor = None
        if cfg.checkpoint_path and cfg.checkpoint_path.exists():
            self._predictor = self._load(cfg.checkpoint_path)

    def _load(self, ckpt_path: Path):
        try:
            import torch
        except ImportError:
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("predictor")

    def propose(
        self,
        target_smiles: str,
        target_inchi_key: str,
        *,
        top_k: int = 10,
        reaction_class_hint: ReactionClass | None = None,
        **kwargs: Any,
    ) -> ProposerOutput:
        edits: list[tuple[list[str], float, str]] = []  # (precursors_smiles, log_prob, reaction_class)
        if self._predictor is not None and callable(self._predictor):
            edits = self._predictor(
                target_smiles,
                top_k=self.cfg.top_k_edits,
                reaction_class_hint=reaction_class_hint,
            )

        candidates: list[list[str]] = []
        candidate_smiles: list[list[str]] = []
        confidences: list[float] = []
        class_preds: list[ReactionClass] = []

        for precursors, logp, rc in edits:
            if len(candidates) >= top_k:
                break
            canon, ikeys = [], []
            ok = True
            for p in precursors:
                cs = canonicalize_smiles(p)
                if cs is None:
                    ok = False
                    break
                canon.append(cs)
                ik = inchi_key_from_smiles(cs)
                if ik is None:
                    ok = False
                    break
                ikeys.append(ik)
            if not ok or not canon:
                continue
            candidates.append(ikeys)
            candidate_smiles.append(canon)
            confidences.append(float(math.exp(logp)))
            class_preds.append(rc or "unclassified")

        return ProposerOutput(
            channel=self.channel,
            target_inchi_key=target_inchi_key,
            target_smiles=target_smiles,
            reaction_class_hint=reaction_class_hint,
            candidates=candidates,
            candidate_smiles=candidate_smiles,
            confidences=confidences,
            reaction_class_predictions=class_preds,
        )
