"""Evaluation metrics for proposer + ranker.

All metrics take simple sequences so callers can swap in any data shape.
Sealed-case answers are matched by InChIKey (canonical equality).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def rank_of(target_id: str, ranked_ids: Sequence[str]) -> int | None:
    """1-indexed rank of `target_id` in `ranked_ids`, or None if absent."""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid == target_id:
            return i
    return None


def exact_recall_at_k(target_id: str, ranked_ids: Sequence[str], *, k: int) -> bool:
    r = rank_of(target_id, ranked_ids)
    return r is not None and r <= k


def functional_recall_at_k(
    functional_target_ids: Iterable[str],
    ranked_ids: Sequence[str],
    *,
    k: int,
) -> bool:
    """True iff ANY pre-registered functional-equivalent appears in top-k."""
    target_set = set(functional_target_ids)
    return any(cid in target_set for cid in ranked_ids[:k])


def mean_reciprocal_rank(target_id: str, ranked_ids: Sequence[str]) -> float:
    r = rank_of(target_id, ranked_ids)
    return 1.0 / r if r is not None else 0.0


def invalid_rate(raw_count: int, invalid_count: int) -> float:
    if raw_count <= 0:
        return 0.0
    return invalid_count / raw_count


def novelty_rate(generated_ids: Sequence[str], known_universe: set[str]) -> float:
    """Fraction of generated candidates that are NOT in the known universe."""
    if not generated_ids:
        return 0.0
    return sum(1 for cid in generated_ids if cid not in known_universe) / len(generated_ids)


def diversity(ranked_inchi_keys: Sequence[str]) -> float:
    """Distinct-InChIKey fraction in a ranking; cheap proxy for diversity."""
    if not ranked_inchi_keys:
        return 0.0
    return len(set(ranked_inchi_keys)) / len(ranked_inchi_keys)
