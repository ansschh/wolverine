"""Heuristic ranker tests (no chemistry dependency).

Uses the `evidence_factory` fixture defined in tests/conftest.py.
"""

from __future__ import annotations

from rasyn.ranker.heuristic import HeuristicRanker


def test_ranker_outputs_one_per_candidate(evidence_factory):
    ranker = HeuristicRanker()
    cands = [evidence_factory("a"), evidence_factory("b"), evidence_factory("c")]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    assert len(out) == 3
    assert {o.candidate_id for o in out} == {"a", "b", "c"}


def test_ranker_assigns_ranks_1_to_n(evidence_factory):
    ranker = HeuristicRanker()
    cands = [evidence_factory(f"c{i}") for i in range(5)]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    ranks = [o.rank for o in out]
    assert ranks == [1, 2, 3, 4, 5]


def test_ranker_descending_score(evidence_factory):
    ranker = HeuristicRanker()
    cands = [
        evidence_factory("strong", retention="strong", improvement="large"),
        evidence_factory("ok", retention="acceptable", improvement="moderate"),
        evidence_factory("weak", retention="weak", improvement="none"),
    ]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    ids = [o.candidate_id for o in out]
    assert ids[0] == "strong"
    assert ids[-1] == "weak"


def test_ranker_label_probs_sum_close_to_1(evidence_factory):
    ranker = HeuristicRanker()
    cands = [evidence_factory("a"), evidence_factory("b")]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    for o in out:
        s = sum(o.rescue_label_probs.values())
        assert 0.99 <= s <= 1.01, f"label probs don't sum to 1: {s}"


def test_ranker_can_act_as_baseline_via_score(evidence_factory):
    """The harness calls .score; ensure interface works."""
    ranker = HeuristicRanker()
    cands = [evidence_factory("a"), evidence_factory("b")]
    out = ranker.score("CCO", cands, "hERG")
    assert len(out) == 2
    # Returned in descending score order.
    assert out[0][1] >= out[1][1]
