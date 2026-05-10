"""Synthetic fixture sanity tests (no RDKit needed)."""

from __future__ import annotations

from rasyn.synth.fixture import SYNTHETIC_PARENTS, build_synthetic_fixture


def test_three_synth_cases():
    packets, pool = build_synthetic_fixture()
    assert set(packets.keys()) == set(SYNTHETIC_PARENTS.keys())
    assert len(pool) > 0


def test_each_packet_has_liability_and_mode():
    packets, _ = build_synthetic_fixture()
    for case_id, p in packets.items():
        assert p.case_id == case_id
        assert p.liability_context.liability_type
        assert p.rescue_context.rescue_mode


def test_pool_contains_decoys_and_targets():
    _, pool = build_synthetic_fixture()
    # We at least have some short-and-irrelevant entries (decoys) and some
    # plausible analog patterns. Just verify variety.
    assert len(set(pool)) == len(pool), "fixture pool should be deduplicated"
    assert any(s == "c1ccccc1" for s in pool)  # explicit decoy
