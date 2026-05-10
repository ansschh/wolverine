"""Heuristic ranker tests (no chemistry dependency)."""

from __future__ import annotations

from conftest import make_evidence as _ev_safe

from rasyn.ranker.heuristic import HeuristicRanker


def test_ranker_outputs_one_per_candidate():
    ranker = HeuristicRanker()
    cands = [_ev_safe("a"), _ev_safe("b"), _ev_safe("c")]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    assert len(out) == 3
    assert {o.candidate_id for o in out} == {"a", "b", "c"}


def test_ranker_assigns_ranks_1_to_n():
    ranker = HeuristicRanker()
    cands = [_ev_safe(f"c{i}") for i in range(5)]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    ranks = [o.rank for o in out]
    assert ranks == [1, 2, 3, 4, 5]


def test_ranker_descending_score():
    ranker = HeuristicRanker()
    cands = [
        _ev_safe("strong", retention="strong", improvement="large"),
        _ev_safe("ok", retention="acceptable", improvement="moderate"),
        _ev_safe("weak", retention="weak", improvement="none"),
    ]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    ids = [o.candidate_id for o in out]
    assert ids[0] == "strong"
    assert ids[-1] == "weak"


def test_ranker_label_probs_sum_close_to_1():
    ranker = HeuristicRanker()
    cands = [_ev_safe("a"), _ev_safe("b")]
    out = ranker.rank(parent_smiles="CCO", candidates=cands, liability_type="hERG", case_id="X")
    for o in out:
        s = sum(o.rescue_label_probs.values())
        assert 0.99 <= s <= 1.01, f"label probs don't sum to 1: {s}"


def test_ranker_can_act_as_baseline_via_score():
    """The harness calls .score; ensure interface works."""
    ranker = HeuristicRanker()
    cands = [_ev_safe("a"), _ev_safe("b")]
    out = ranker.score("CCO", cands, "hERG")
    assert len(out) == 2
    # Returned in descending score order.
    assert out[0][1] >= out[1][1]
