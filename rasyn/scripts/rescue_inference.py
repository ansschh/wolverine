"""End-to-end ADMET rescue prediction on a sealed case.

Pipeline:
  1. Load sealed case from registry -> ADMETChallengePacket
  2. Pre-filter ChEMBL corpus to N nearest neighbours by Tanimoto-to-parent
  3. Run deterministic 3-channel proposer ensemble on that pool
  4. Build CandidateEvidencePacket per candidate (descriptors + deltas + liability drivers)
  5. Rank with HeuristicRanker
  6. Report top-20 + check if known answer is in top-K (exact recall)

Run:
    python scripts/rescue_inference.py --case ADMET-001 \\
        --candidate-pool rasyn/data/clean/molecules_canonical.parquet \\
        --top-similarity-pool 5000 --max-candidates 2000

Default case: ADMET-001 (terfenadine -> fexofenadine).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from rasyn.data.registry.loader import load_sealed_case_registry
from rasyn.evidence.builder import build_candidate_evidence
from rasyn.proposer.base import ProposerContext
from rasyn.proposer.ensemble import deterministic_ensemble, run_ensemble
from rasyn.ranker.heuristic import HeuristicRanker
from rasyn.schemas.challenge import (
    ActivityContext,
    ADMETChallengePacket,
    LiabilityContext,
    RescueContextPacket,
)
from rasyn.utils.similarity import morgan_bits, tanimoto

CASE_PACKETS = {
    "ADMET-001": dict(
        target_name="Histamine H1 receptor",
        target_chembl_id="CHEMBL231",
        desired_pharmacology="H1 antagonism",
        parent_potency_value=10.0,
        parent_potency_unit="nM",
        parent_potency_endpoint="IC50",
        liability_type="hERG",
        measurement_endpoint="hERG IC50",
        parent_value=200.0,
        parent_unit="nM",
        parent_category="high",
        target_improvement_category="low",
        rescue_mode="active_metabolite_safety_rescue",
        constraints=["preserve H1 antagonism within 10x parent potency"],
    ),
    "ADMET-002": dict(
        target_name="HSV thymidine kinase / DNA polymerase",
        target_chembl_id=None,
        desired_pharmacology="anti-herpetic",
        parent_potency_value=1.0,
        parent_potency_unit="uM",
        parent_potency_endpoint="EC50",
        liability_type="oral_exposure",
        measurement_endpoint="oral bioavailability",
        parent_value=10.0,
        parent_unit="%",
        parent_category="low",
        target_improvement_category="moderate",
        rescue_mode="prodrug_exposure_rescue",
        constraints=["deliver acyclovir as active species"],
    ),
    "ADMET-003": dict(
        target_name="HL-60 anti-leukemic",
        target_chembl_id=None,
        desired_pharmacology="anti-leukemic differentiation agent",
        parent_potency_value=2.7,
        parent_potency_unit="nM",
        parent_potency_endpoint="EC50",
        liability_type="solubility",
        measurement_endpoint="aqueous solubility",
        parent_value=15.0,
        parent_unit="uM",
        parent_category="low",
        target_improvement_category="high",
        rescue_mode="polarity_solubility_rescue",
        constraints=["preserve EC50 within 10x parent (~27 nM)"],
    ),
}


def make_packet(case) -> ADMETChallengePacket:
    info = CASE_PACKETS[case.case_id]
    return ADMETChallengePacket(
        case_id=case.case_id,
        parent_canonical_smiles=case.parent.canonical_smiles or "",
        parent_inchi_key=case.parent.inchi_key or "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
        activity_context=ActivityContext(
            target_name=info["target_name"],
            target_chembl_id=info["target_chembl_id"],
            desired_pharmacology=info["desired_pharmacology"],
            parent_potency_value=info["parent_potency_value"],
            parent_potency_unit=info["parent_potency_unit"],
            parent_potency_endpoint=info["parent_potency_endpoint"],
        ),
        liability_context=LiabilityContext(
            liability_type=info["liability_type"],
            measurement_endpoint=info["measurement_endpoint"],
            parent_value=info["parent_value"],
            parent_unit=info["parent_unit"],
            parent_category=info["parent_category"],
            target_improvement_category=info["target_improvement_category"],
        ),
        rescue_context=RescueContextPacket(
            rescue_mode=info["rescue_mode"],
            constraints=info["constraints"],
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="ADMET-001", choices=list(CASE_PACKETS.keys()))
    p.add_argument(
        "--candidate-pool",
        type=Path,
        default=Path("rasyn/data/clean/molecules_canonical.parquet"),
    )
    p.add_argument("--top-similarity-pool", type=int, default=5000)
    p.add_argument("--max-candidates", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("rasyn/data/clean/rescue_inference"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    reg = load_sealed_case_registry()
    case = next((c for c in reg.cases if c.case_id == args.case), None)
    if case is None:
        raise SystemExit(f"Case {args.case} not in registry")
    if not case.parent.canonical_smiles:
        raise SystemExit(
            f"Case {args.case} parent SMILES not populated; run scripts/populate_registry.py"
        )

    packet = make_packet(case)
    answer_smi = case.answer.canonical_smiles
    answer_ik = case.answer.inchi_key

    print(f"=== Case: {packet.case_id} ===")
    print(f"Parent: {case.parent.name} | {packet.parent_canonical_smiles}")
    print(f"Answer: {case.answer.name} | {answer_smi}")
    print(f"Liability: {packet.liability_context.liability_type}")
    print(f"Rescue mode: {packet.rescue_context.rescue_mode}")

    print(f"\nLoading candidate pool: {args.candidate_pool}")
    df = pd.read_parquet(args.candidate_pool)
    pool_smi = df["canonical_smiles"].astype(str).tolist()
    print(f"  Universe size: {len(pool_smi):,}")

    # Tanimoto pre-filter to keep K nearest neighbours of parent
    print(f"\nPre-filtering to top-{args.top_similarity_pool} by Tanimoto-to-parent...")
    parent_fp = morgan_bits(packet.parent_canonical_smiles)
    if parent_fp is None:
        raise SystemExit("Cannot fingerprint parent")

    sims: list[tuple[float, str]] = []
    answer_seen = False
    for i, smi in enumerate(pool_smi):
        if i % 200_000 == 0 and i > 0:
            print(f"  scanned {i:,}/{len(pool_smi):,}")
        fp = morgan_bits(smi)
        if fp is None:
            continue
        sim = tanimoto(parent_fp, fp)
        sims.append((sim, smi))
        if smi == answer_smi:
            answer_seen = True
    sims.sort(reverse=True)
    top_pool = [smi for _, smi in sims[: args.top_similarity_pool]]
    print(f"  Pre-filtered pool: {len(top_pool)}")
    print(f"  Answer present in raw universe (post-decontam): {answer_seen}")
    print(f"  Answer present in pre-filtered top-K: {answer_smi in top_pool}")
    if answer_smi in top_pool:
        idx = top_pool.index(answer_smi)
        print(f"  Answer rank in pre-filtered pool: {idx + 1} (Tanimoto={sims[idx][0]:.3f})")

    # Force-include the answer in the pool so we can score it (decontam removed it from training,
    # but for evaluation we want to see if the ranker would have placed it well had it been there).
    if not answer_seen and answer_smi:
        print("  (Answer not in decontaminated universe — adding for evaluation only)")
        top_pool.append(answer_smi)

    print(f"\nRunning proposer ensemble on pre-filtered pool...")
    ctx = ProposerContext(candidate_smiles_pool=top_pool)
    pool, per_channel = run_ensemble(
        packet, ctx, deterministic_ensemble(), max_pool_size=args.max_candidates,
    )
    print(f"  Pool after ensemble + dedup + cap: {len(pool)}")
    for o in per_channel:
        print(f"    {o.channel}: raw={o.raw_count} invalid={o.invalid_count} kept={len(o.candidates)}")

    # Map InChIKey -> CandidateAnnotation for answer lookup
    answer_id_in_pool = None
    for ann in pool:
        if ann.inchi_key == answer_ik:
            answer_id_in_pool = ann.candidate_id
            break

    print(f"\nAnswer in proposer ensemble pool: {answer_id_in_pool is not None}")
    if answer_id_in_pool:
        print(f"  Answer candidate_id: {answer_id_in_pool}")

    print(f"\nBuilding evidence packets ({len(pool)} candidates)...")
    evidence_packets = []
    inchi_to_id: dict[str, str] = {}
    n_evidence_failed = 0
    for ann in pool:
        ev = build_candidate_evidence(
            parent_smiles=packet.parent_canonical_smiles,
            candidate_smiles=ann.canonical_smiles,
            liability_type=packet.liability_context.liability_type,
            candidate_id=ann.candidate_id,
            proposer_sources=list(ann.proposer_sources),
        )
        if ev is None:
            n_evidence_failed += 1
            continue
        evidence_packets.append(ev)
        inchi_to_id[ann.inchi_key] = ann.candidate_id
    print(f"  Built {len(evidence_packets)} evidence packets ({n_evidence_failed} failed)")

    print(f"\nRanking with HeuristicRanker...")
    ranker = HeuristicRanker()
    ranked = ranker.rank(
        parent_smiles=packet.parent_canonical_smiles,
        candidates=evidence_packets,
        liability_type=packet.liability_context.liability_type,
        case_id=packet.case_id,
    )

    # Find rank of answer
    answer_id = inchi_to_id.get(answer_ik) if answer_ik else None
    answer_rank = None
    answer_score = None
    for r in ranked:
        if r.candidate_id == answer_id:
            answer_rank = r.rank
            answer_score = r.rescue_score
            break

    print(f"\n=== TOP 20 ===")
    for r in ranked[:20]:
        is_answer = r.candidate_id == answer_id
        marker = "  ⭐ ANSWER" if is_answer else ""
        print(
            f"  #{r.rank:>3}: score={r.rescue_score:.4f} | "
            f"{r.candidate_id[:60]}{marker}"
        )

    print(f"\n=== RESULT ===")
    print(f"Pool size (filtered): {len(pool)}")
    print(f"Evidence packets built: {len(evidence_packets)}")
    print(f"Answer rank: {answer_rank}")
    print(f"Answer score: {answer_score}")
    print(f"Exact recall@5: {answer_rank is not None and answer_rank <= 5}")
    print(f"Exact recall@10: {answer_rank is not None and answer_rank <= 10}")
    print(f"Exact recall@20: {answer_rank is not None and answer_rank <= 20}")

    out = {
        "case_id": packet.case_id,
        "parent_smiles": packet.parent_canonical_smiles,
        "parent_inchi_key": packet.parent_inchi_key,
        "answer_smiles": answer_smi,
        "answer_inchi_key": answer_ik,
        "candidate_universe_size": len(pool_smi),
        "answer_in_decontam_universe": answer_seen,
        "answer_in_prefiltered_pool": answer_smi in top_pool,
        "ensemble_pool_size": len(pool),
        "evidence_packets_built": len(evidence_packets),
        "answer_rank": answer_rank,
        "answer_score": answer_score,
        "exact_recall_at_5": bool(answer_rank is not None and answer_rank <= 5),
        "exact_recall_at_10": bool(answer_rank is not None and answer_rank <= 10),
        "exact_recall_at_20": bool(answer_rank is not None and answer_rank <= 20),
        "top_20": [
            {"rank": r.rank, "rescue_score": r.rescue_score, "candidate_id": r.candidate_id}
            for r in ranked[:20]
        ],
        "per_channel_attribution": {
            o.channel: {"raw": o.raw_count, "invalid": o.invalid_count, "kept": len(o.candidates)}
            for o in per_channel
        },
        "elapsed_seconds": time.time() - t0,
    }
    out_path = args.out / f"{packet.case_id}_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
