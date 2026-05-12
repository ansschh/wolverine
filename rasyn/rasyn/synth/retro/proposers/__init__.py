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

__all__ = ["RetroProposer"]
