"""Metric correctness tests."""

from __future__ import annotations

from rasyn.eval.metrics import (
    diversity,
    exact_recall_at_k,
    functional_recall_at_k,
    invalid_rate,
    mean_reciprocal_rank,
    novelty_rate,
    rank_of,
)


def test_rank_of_present_and_absent():
    assert rank_of("b", ["a", "b", "c"]) == 2
    assert rank_of("z", ["a", "b", "c"]) is None


def test_exact_recall_at_k():
    ranked = ["a", "b", "c", "d", "e"]
    assert exact_recall_at_k("a", ranked, k=1)
    assert exact_recall_at_k("e", ranked, k=5)
    assert not exact_recall_at_k("e", ranked, k=4)
    assert not exact_recall_at_k("x", ranked, k=5)


def test_functional_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    assert functional_recall_at_k({"c", "z"}, ranked, k=3)
    assert not functional_recall_at_k({"x", "y"}, ranked, k=4)
    assert functional_recall_at_k({"d"}, ranked, k=4)


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank("a", ["a", "b"]) == 1.0
    assert mean_reciprocal_rank("b", ["a", "b"]) == 0.5
    assert mean_reciprocal_rank("z", ["a", "b"]) == 0.0


def test_invalid_rate():
    assert invalid_rate(100, 0) == 0.0
    assert invalid_rate(100, 25) == 0.25
    assert invalid_rate(0, 0) == 0.0  # guard against div0


def test_novelty_rate():
    assert novelty_rate(["a", "b", "c"], known_universe={"a"}) == 2 / 3
    assert novelty_rate([], known_universe={"a"}) == 0.0
    assert novelty_rate(["a"], known_universe=set()) == 1.0


def test_diversity():
    assert diversity(["a", "a", "b"]) == 2 / 3
    assert diversity([]) == 0.0
    assert diversity(["a", "b", "c"]) == 1.0
