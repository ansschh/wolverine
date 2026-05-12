"""Seq2seq SMILES proposer (RETRO_PLAN R-2 Channel 3).

Product SMILES -> reactants SMILES, separated by '.', using an
encoder-decoder transformer. Encoder is initialised from
`checkpoints/smiles_lm_200m/` (200M MLM backbone, ChEMBL-pretrained);
decoder is initialised from the AR-LM `checkpoints/smiles_ar_lm_200m/`.

Inference: beam search with width K. Each beam is split on '.' to give
one candidate precursor set; canonicalisation + valid-SMILES filter
applied before emission.
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
class Seq2SeqProposerConfig:
    checkpoint_path: Path | None = None
    beam_width: int = 20
    max_decode_len: int = 128
    temperature: float = 1.0
    device: str = "cpu"


class Seq2SeqProposer(RetroProposer):
    """Encoder-decoder SMILES->SMILES retro proposer."""

    channel: ProposerChannel = "seq2seq"

    def __init__(self, cfg: Seq2SeqProposerConfig):
        self.cfg = cfg
        self._model = None
        self._vocab = None
        if cfg.checkpoint_path and cfg.checkpoint_path.exists():
            self._model, self._vocab = self._load(cfg.checkpoint_path)

    def _load(self, ckpt_path: Path):
        try:
            import torch
        except ImportError:
            return None, None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("model"), ckpt.get("vocab")

    def _beam_decode(self, target_smiles: str) -> list[tuple[str, float]]:
        """Return list of (decoded_string, log_prob) sorted by score.

        If model not loaded, returns an empty list. Concrete decoding logic
        lives in the training script which writes an `infer()` callable into
        the checkpoint; here we keep the loader hook abstract.
        """
        if self._model is None:
            return []
        if callable(self._model):
            return self._model(target_smiles, beam_width=self.cfg.beam_width,
                              max_len=self.cfg.max_decode_len)
        # state-dict-only fallback: caller can extend this class with a
        # bound decoder. Empty list signals 'no decoding available'.
        return []

    def propose(
        self,
        target_smiles: str,
        target_inchi_key: str,
        *,
        top_k: int = 10,
        reaction_class_hint: ReactionClass | None = None,
        **kwargs: Any,
    ) -> ProposerOutput:
        beams = self._beam_decode(target_smiles)
        candidates: list[list[str]] = []
        candidate_smiles: list[list[str]] = []
        confidences: list[float] = []
        class_preds: list[ReactionClass] = []

        for decoded, logp in beams[: self.cfg.beam_width]:
            if len(candidates) >= top_k:
                break
            reactants = [r for r in decoded.split(".") if r]
            canon, ikeys = [], []
            ok = True
            for r in reactants:
                cs = canonicalize_smiles(r)
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
            class_preds.append(reaction_class_hint or "unclassified")

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
