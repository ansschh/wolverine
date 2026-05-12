"""Diffusion reactant-completion proposer (RETRO_PLAN R-2 Channel 5).

Per RETRO.md Lock 4: diffusion is a *reactant/synthon completion*
proposer only — never a planner. Input: product graph + disconnection
mask (which bonds are marked as "broken") + reaction-class hint.
Output: completed reactant subgraph(s).

Architecture: DiGress-style discrete graph diffusion, FiLM-conditioned
on reaction class. Reuses the `rasyn/antibiotic/graph_diffusion.py`
machinery from the ABX module, scaled up to ~30-50M params + 200K-500K
steps (vs the 5.4M / 60K-step ABX run that was undertrained per
MEMORY L47).

Disconnection-mask strategy: by default, randomly mask 1-2 ring or
chain bonds (the most common synthetically meaningful break). At
inference, the caller can pass an explicit `disconnection_mask` via
kwargs to fix the break site.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.reactions import canonicalize_smiles, inchi_key_from_smiles
from rasyn.synth.retro.schemas import ProposerChannel, ProposerOutput, ReactionClass


@dataclass
class DiffusionProposerConfig:
    checkpoint_path: Path | None = None
    n_samples_per_target: int = 20
    n_disconnection_proposals: int = 4
    sampling_temperature: float = 1.0
    diffusion_steps: int = 100
    device: str = "cpu"
    seed: int = 42


class DiffusionProposer(RetroProposer):
    channel: ProposerChannel = "diffusion"

    def __init__(self, cfg: DiffusionProposerConfig):
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
        return ckpt.get("model")

    def _sample_reactant_sets(
        self,
        target_smiles: str,
        reaction_class_hint: ReactionClass | None,
        disconnection_mask: list[int] | None,
    ) -> list[tuple[list[str], float]]:
        """Return [(reactants_smiles, log_prob), ...] from the diffusion model.

        Concrete sampling lives in the training-script-bound model callable.
        When no model is loaded, returns an empty list.
        """
        if self._model is None:
            return []
        if callable(self._model):
            return self._model(
                product_smiles=target_smiles,
                reaction_class_hint=reaction_class_hint,
                disconnection_mask=disconnection_mask,
                n_samples=self.cfg.n_samples_per_target,
                temperature=self.cfg.sampling_temperature,
                diffusion_steps=self.cfg.diffusion_steps,
                seed=self.cfg.seed,
            )
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
        disconnection_mask = kwargs.get("disconnection_mask")
        # Sample multiple disconnection proposals if none provided
        all_samples: list[tuple[list[str], float]] = []
        if disconnection_mask is None:
            for _ in range(self.cfg.n_disconnection_proposals):
                samples = self._sample_reactant_sets(
                    target_smiles, reaction_class_hint, disconnection_mask=None,
                )
                all_samples.extend(samples)
        else:
            all_samples = self._sample_reactant_sets(
                target_smiles, reaction_class_hint, disconnection_mask,
            )

        # Sort by log-prob desc
        all_samples.sort(key=lambda x: -x[1])

        candidates: list[list[str]] = []
        candidate_smiles: list[list[str]] = []
        confidences: list[float] = []
        class_preds: list[ReactionClass] = []
        seen_keys: set[tuple[str, ...]] = set()

        for reactants, logp in all_samples:
            if len(candidates) >= top_k:
                break
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
            key = tuple(sorted(ikeys))
            if key in seen_keys:
                continue
            seen_keys.add(key)
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
