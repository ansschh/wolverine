"""Rasyn-Retro: bespoke retrosynthesis module.

Implementation tracks RETRO_PLAN.md at repo root. The system is a
*planning system* (RETRO.md Lock 1), not a one-step reactant predictor.

Pipeline:
    target SMILES -> Retro* AND-OR tree search ->
        5-channel proposer ensemble (template, graph-edit, seq2seq,
        retrieval, diffusion-reactant-completion) ->
        forward-reaction validator + condition predictor ->
        buyability index -> ranked CandidateRoute list.
"""

__all__ = [
    "schemas",
    "registry",
    "proposers",
]
