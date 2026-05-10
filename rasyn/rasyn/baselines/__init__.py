"""8 baselines: random, similarity, polarity, liability-only, activity-only,
weighted-property, MMP-frequency, medchem-heuristic.

Every baseline implements the same interface as a proposer-level scorer:
takes a parent + a list of candidate evidence packets, returns a ranking.
This keeps Mode B (closed hard-ranking) symmetric across baselines and the
ranker.
"""

from rasyn.baselines.all import ALL_BASELINES, get_baseline

__all__ = ["ALL_BASELINES", "get_baseline"]
