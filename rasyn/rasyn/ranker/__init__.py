"""Pairwise rescue ranker.

Two implementations:
  - heuristic.HeuristicRanker — non-ML composite scorer; usable BEFORE training.
                                 Drop-in compatible with the eval harness as
                                 a Baseline-shaped ranker.
  - torch_ranker.ConcatMLPRanker — PyTorch nn.Module scaffold; trains in
                                    Stage-1/Stage-2/Stage-3 of PLAN.md §8.
"""

from rasyn.ranker.heuristic import HeuristicRanker

__all__ = ["HeuristicRanker"]
