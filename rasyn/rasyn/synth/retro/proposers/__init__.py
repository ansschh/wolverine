"""Single-step retrosynthesis proposers (5 channels).

Per RETRO_PLAN.md R-2:
    - template:   RDChiral templates + neural template classifier
    - graphedit:  bond-edit classifier on product graph
    - seq2seq:    SMILES product -> reactants encoder-decoder (200M-MLM init)
    - retrieval:  FAISS over product Morgan FP + RXNFP class metadata
    - diffusion:  DiGress-style reactant completion given disconnection mask

All channels share the RetroProposer interface and return ProposerOutput.
"""
from __future__ import annotations

from rasyn.synth.retro.proposers.base import RetroProposer
from rasyn.synth.retro.proposers.diffusion import DiffusionProposer, DiffusionProposerConfig
from rasyn.synth.retro.proposers.graphedit import GraphEditProposer, GraphEditProposerConfig
from rasyn.synth.retro.proposers.retrieval import RetrievalProposer, RetrievalProposerConfig
from rasyn.synth.retro.proposers.seq2seq import Seq2SeqProposer, Seq2SeqProposerConfig
from rasyn.synth.retro.proposers.template import TemplateProposer, TemplateProposerConfig

__all__ = [
    "RetroProposer",
    "DiffusionProposer", "DiffusionProposerConfig",
    "GraphEditProposer", "GraphEditProposerConfig",
    "RetrievalProposer", "RetrievalProposerConfig",
    "Seq2SeqProposer", "Seq2SeqProposerConfig",
    "TemplateProposer", "TemplateProposerConfig",
]
