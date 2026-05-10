"""Layer-1 smoke test (PLAN.md §7).

End-to-end exercise of every box in §3 against the synthetic fixture:
    1. Load registry + generate canaries
    2. Build challenge packets from synthetic fixture
    3. Run the deterministic 3-channel proposer ensemble
    4. Build evidence for each candidate
    5. Run all 8 baselines + record per-channel attribution
    6. Run canary audit against the synthetic candidate pool
    7. Print a one-line PASS/FAIL per check

Requires RDKit. Run via:
    python scripts/layer1_smoke.py
"""

from __future__ import annotations

import sys

from rasyn.baselines import ALL_BASELINES
from rasyn.data.decontam.canary_audit import audit_against_rows
from rasyn.data.registry.canary_generator import generate_canaries_for_registry
from rasyn.data.registry.loader import load_sealed_case_registry
from rasyn.eval.harness import evaluate_mode_A
from rasyn.proposer.ensemble import deterministic_ensemble
from rasyn.synth.fixture import build_synthetic_fixture


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def main() -> int:
    failures = 0

    print("[Layer-1 smoke]")
    print("== Registry + canaries ==")
    try:
        reg = load_sealed_case_registry()
        canaries = generate_canaries_for_registry(reg, per_layer=4)
        _ok(f"loaded registry; {len(reg.cases)} cases; {len(canaries)} canaries")
    except Exception as e:
        _fail(f"registry/canaries: {e}")
        failures += 1

    print("== Synthetic fixture ==")
    try:
        packets, pool = build_synthetic_fixture()
        _ok(f"fixture has {len(packets)} packets; pool size {len(pool)}")
    except Exception as e:
        _fail(f"fixture build: {e}")
        return 1

    print("== Proposer ensemble (deterministic 3-channel) ==")
    try:
        from rasyn.proposer.base import ProposerContext
        from rasyn.proposer.ensemble import run_ensemble

        any_packet = next(iter(packets.values()))
        ctx = ProposerContext(candidate_smiles_pool=pool)
        merged, per_channel = run_ensemble(any_packet, ctx, deterministic_ensemble())
        _ok(f"proposer ran for {any_packet.case_id}; merged pool size {len(merged)}; channels {[o.channel for o in per_channel]}")
    except Exception as e:
        _fail(f"proposer ensemble: {e}")
        failures += 1
        merged = []

    print("== 8 baselines on each synthetic case ==")
    try:
        for case_id, packet in packets.items():
            for cls in ALL_BASELINES:
                baseline = cls()
                # Use a stub answer InChIKey not present in pool to exercise the "absent" path.
                res = evaluate_mode_A(
                    packet=packet,
                    candidate_pool=pool,
                    ranker=baseline,
                    answer_inchi_key="ZZZZZZZZZZZZZZ-ZZZZZZZZZZ-N",
                )
                assert res.case_id == case_id and res.ranker_name == baseline.name
            _ok(f"all 8 baselines ran on {case_id}")
    except Exception as e:
        _fail(f"baseline eval: {e}")
        failures += 1

    print("== Canary audit (clean rows) ==")
    try:
        rows = [{"smiles": s, "synonyms": []} for s in pool]
        result = audit_against_rows(canaries, rows)
        if result.passed:
            _ok(f"canary audit: 0 survivors out of {result.total_canaries}")
        else:
            _fail(f"canary audit: {len(result.survivors)} survivors")
            failures += 1
    except Exception as e:
        _fail(f"canary audit: {e}")
        failures += 1

    print(f"\n[Layer-1 smoke] {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
