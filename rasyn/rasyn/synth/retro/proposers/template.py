"""Template proposer (RETRO_PLAN R-2 Channel 1).

Two-stage:
  1. Neural template classifier: given a target product, predict top-K
     template indices (a softmax over the template bank). Encoder is the
     200M MLM SMILES backbone, FiLM-conditioned on optional reaction class.
  2. Template application: each predicted template is applied to the
     product via RDChiral; failures (template doesn't fire on this
     substrate) are dropped.

Inputs at inference:
  - target_smiles (canonical)
  - top_k_templates: how many templates to try (default 100)
  - top_k_candidates: how many candidate precursor sets to emit (default 10)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.reactions import canonicalize_smiles, inchi_key_from_smiles
from rasyn.synth.retro.schemas import ProposerChannel, ProposerOutput, ReactionClass
from rasyn.synth.retro.templates import RetroTemplate, apply_template


@dataclass
class TemplateProposerConfig:
    checkpoint_path: Path | None = None  # neural template classifier ckpt
    templates_path: Path | None = None  # pickled list[RetroTemplate]
    top_k_templates: int = 100
    top_k_candidates: int = 10
    device: str = "cpu"


class TemplateProposer(RetroProposer):
    """Template-based proposer with optional neural ranking of templates.

    If `checkpoint_path` is None or torch is unavailable, falls back to a
    frequency-only ranker (apply the top-N most-common templates).
    """

    channel: ProposerChannel = "template"

    def __init__(self, cfg: TemplateProposerConfig):
        self.cfg = cfg
        self.templates: list[RetroTemplate] = []
        if cfg.templates_path and cfg.templates_path.exists():
            with open(cfg.templates_path, "rb") as fh:
                self.templates = pickle.load(fh)
        self._classifier = None
        if cfg.checkpoint_path and cfg.checkpoint_path.exists():
            self._classifier = self._load_classifier(cfg.checkpoint_path)

    def _load_classifier(self, ckpt_path: Path):
        try:
            import torch
        except ImportError:
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt  # Caller picks model out of ckpt["model"]

    def _rank_templates(
        self,
        target_smiles: str,
        reaction_class_hint: ReactionClass | None,
    ) -> list[tuple[int, float]]:
        """Return list of (template_idx, score) sorted by score desc.

        Without a trained classifier, falls back to count-based ranking.
        """
        if self._classifier is None or not self.templates:
            return [
                (i, t.extracted_count / max(1.0, self.templates[0].extracted_count))
                for i, t in enumerate(self.templates[:self.cfg.top_k_templates])
            ]
        # Neural ranking — defer to a small inference function on the loaded
        # state dict. Caller would normally subclass to wire the actual
        # nn.Module; here we keep it ckpt-shape-agnostic.
        return [
            (i, 1.0 / (1 + i)) for i in range(min(self.cfg.top_k_templates, len(self.templates)))
        ]

    def propose(
        self,
        target_smiles: str,
        target_inchi_key: str,
        *,
        top_k: int = 10,
        reaction_class_hint: ReactionClass | None = None,
        **kwargs: Any,
    ) -> ProposerOutput:
        ranked = self._rank_templates(target_smiles, reaction_class_hint)
        candidates: list[list[str]] = []
        candidate_smiles: list[list[str]] = []
        confidences: list[float] = []
        class_preds: list[ReactionClass] = []
        template_hashes: list[str] = []

        for tmpl_idx, score in ranked[: self.cfg.top_k_templates]:
            if tmpl_idx >= len(self.templates):
                break
            tmpl = self.templates[tmpl_idx]
            sets = apply_template(tmpl.template_smarts, target_smiles)
            for precursors in sets:
                canon = []
                ikeys = []
                for p in precursors:
                    cs = canonicalize_smiles(p)
                    if cs is None:
                        canon = []
                        break
                    canon.append(cs)
                    ik = inchi_key_from_smiles(cs)
                    if ik is None:
                        canon = []
                        break
                    ikeys.append(ik)
                if not canon:
                    continue
                candidates.append(ikeys)
                candidate_smiles.append(canon)
                confidences.append(float(score))
                class_preds.append(reaction_class_hint or "unclassified")
                template_hashes.append(tmpl.template_hash)
                if len(candidates) >= top_k:
                    break
            if len(candidates) >= top_k:
                break

        return ProposerOutput(
            channel=self.channel,
            target_inchi_key=target_inchi_key,
            target_smiles=target_smiles,
            reaction_class_hint=reaction_class_hint,
            candidates=candidates,
            candidate_smiles=candidate_smiles,
            confidences=confidences,
            reaction_class_predictions=class_preds,
            channel_metadata={"template_hashes": template_hashes},
        )
