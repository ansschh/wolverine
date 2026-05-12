"""V4 inference orchestrator — uses the v4 FiLM ranker architecture.

Same flow as run_abx_sealed_cases.py but:
  - Loads ABXMultiHeadRankerV4 (FiLM model) instead of v3.
  - Uses multiplicative composite scoring at inference time:
      final = ab * (1 - cyto)^alpha * (1 - artifact)^beta * (1 + nov_w * novelty)

Run:
    python scripts/run_abx_sealed_cases_v4.py \\
        --ranker rasyn/data/clean/abx_ranker_v4_seed42/checkpoint.pt \\
        --library ... --facts ... \\
        --cases ABX-001,ABX-002,ABX-003 \\
        --top-k 20 \\
        --alpha 2.0 --beta 1.0 --novelty-weight 0.3 \\
        --out artifacts/abx_stage5_v5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from rasyn.antibiotic.registry import load_abx_sealed_case_registry
from rasyn.antibiotic.schemas import (
    ABXChallengePacket, ABXOrganismContext, ABXSelectivityContext, ABXCandidateContext,
)
from rasyn.antibiotic.channels import ABXProposerContext, run_abx_ensemble
from rasyn.antibiotic.eval import closed_hard_ranking, open_proposer, save_metrics
from rasyn.antibiotic.rationale import build_rationale
from rasyn.antibiotic.evidence import build_evidence_packet

from train_abx_ranker_v4 import ABXMultiHeadRankerV4  # noqa: E402
from train_abx_ranker import (  # noqa: E402
    condition_vector, ORGANISM_LIST, GRAM_LIST, SPECTRUM_LIST, FAILURE_MODES,
    tokenize as _tok,
)
from h200_smiles_lm_pretrain import VOCAB_SIZE  # noqa: E402


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def packet_from_case(case) -> ABXChallengePacket:
    return ABXChallengePacket(
        case_id=case.case_id,
        organism_context=ABXOrganismContext(**case.organism_context.model_dump()),
        selectivity_context=ABXSelectivityContext(**case.selectivity_context.model_dump()),
        candidate_context=ABXCandidateContext(**case.candidate_context.model_dump()),
    )


def load_v4_ranker(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    model = ABXMultiHeadRankerV4(
        VOCAB_SIZE,
        d_model=1024, n_heads=16, n_layers=16,
        max_len=cargs.get("max_len", 128),
    ).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def score_v4(model, candidates, *, organism, gram, spectrum, device, max_len=128, bs=32):
    out_rows = []
    cond = condition_vector(organism, gram, spectrum)
    cond_t = torch.from_numpy(cond).to(device)
    for i in range(0, len(candidates), bs):
        chunk = candidates[i:i+bs]
        ids_list, mask_list = [], []
        for c in chunk:
            ids, mask = _tok(c.get("candidate_smiles") or "C", max_len)
            ids_list.append(ids); mask_list.append(mask)
        ids_t = torch.from_numpy(np.stack(ids_list)).long().to(device)
        mask_t = torch.from_numpy(np.stack(mask_list)).bool().to(device)
        cond_b = cond_t.unsqueeze(0).expand(len(chunk), -1)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids_t, mask_t, cond_b)
        ab = out["antibacterial"].float().cpu().tolist()
        cy = out["cytotox"].float().cpu().tolist()
        ar = out["artifact"].float().cpu().tolist()
        fm = out["failure_modes"].float().softmax(-1).cpu().tolist()
        for k, c in enumerate(chunk):
            out_rows.append({
                **c,
                "antibacterial_score": float(ab[k]),
                "cytotox_risk": float(cy[k]),
                "artifact_risk": float(ar[k]),
                "failure_mode_probs": dict(zip(FAILURE_MODES, fm[k])),
            })
    return out_rows


def apply_multiplicative_composite(rows, alpha, beta, novelty_weight,
                                     memorization_threshold):
    for r in rows:
        ab = max(0.0, min(1.0, r["antibacterial_score"]))
        cy = max(0.0, min(1.0, r["cytotox_risk"]))
        ar = max(0.0, min(1.0, r["artifact_risk"]))
        # selectivity: 1 - cyto^alpha * 1 - artifact^beta
        sel = ((1 - cy) ** alpha) * ((1 - ar) ** beta)
        tan_to_active = r.get("max_tanimoto_to_organism_active") or 0.0
        novelty = 1.0 - max(0.0, min(1.0, float(tan_to_active)))
        is_memo = float(tan_to_active) >= memorization_threshold
        nov_eff = 0.0 if is_memo else novelty
        base = ab * sel
        r["v4_base_score"] = float(base)
        r["v4_novelty"] = float(nov_eff)
        r["v4_is_memorization"] = bool(is_memo)
        r["v4_selectivity_factor"] = float(sel)
        r["final_discovery_score"] = float(base * (1.0 + novelty_weight * nov_eff))
    return rows


def decontam_pool(candidates, answer_smi, thresh=0.85):
    if not answer_smi:
        return candidates
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    ans_mol = Chem.MolFromSmiles(answer_smi)
    if ans_mol is None:
        return candidates
    ans_fp = AllChem.GetMorganFingerprintAsBitVect(ans_mol, 2, nBits=2048)
    keep = []
    for c in candidates:
        m = Chem.MolFromSmiles(c.get("candidate_smiles") or "")
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        if DataStructs.TanimotoSimilarity(ans_fp, fp) < thresh:
            keep.append(c)
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranker", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--known-antibiotics", type=Path, default=None)
    p.add_argument("--ch-e-json", type=Path, default=None)
    p.add_argument("--ch-f-json", type=Path, default=None)
    p.add_argument("--cases", default="ABX-001,ABX-002,ABX-003")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--novelty-weight", type=float, default=0.3)
    p.add_argument("--memorization-threshold", type=float, default=0.95)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")
    _log(f"Loading v4 ranker {args.ranker}")
    ranker = load_v4_ranker(args.ranker, device)

    library_df = pd.read_parquet(args.library)
    library_smiles = library_df["canonical_smiles"].dropna().astype(str).tolist()
    library_ik = library_df["inchi_key"].astype(str).tolist() if "inchi_key" in library_df.columns else [None]*len(library_smiles)

    known_smiles = []
    if args.known_antibiotics and args.known_antibiotics.exists():
        known_df = pd.read_parquet(args.known_antibiotics)
        known_smiles = known_df["canonical_smiles"].dropna().astype(str).tolist()
    elif "is_known_antibiotic" in library_df.columns:
        known_smiles = library_df.loc[library_df["is_known_antibiotic"] == True, "canonical_smiles"].dropna().tolist()

    registry = load_abx_sealed_case_registry()
    cases_by_id = {c.case_id: c for c in registry.cases}

    summary = {}
    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        case = cases_by_id.get(case_id)
        if case is None:
            continue
        _log(f"\n===== {case_id} =====")
        packet = packet_from_case(case)
        organism = packet.organism_context.organism
        gram = packet.organism_context.gram_type
        spectrum = packet.organism_context.spectrum_goal

        ctx = ABXProposerContext(
            library_smiles_pool=library_smiles,
            library_inchi_keys=library_ik,
            known_antibiotic_smiles=known_smiles,
            sealed_answer_smiles_or_none=case.hidden_solution.get("canonical_smiles"),
            organism_active_pool_path=args.facts,
        )
        ans_smi = case.hidden_solution.get("canonical_smiles")
        ans_ik = case.hidden_solution.get("inchi_key")
        if ans_smi and ans_smi not in ctx.library_smiles_pool:
            import random
            insert_at = random.randint(0, min(500, len(ctx.library_smiles_pool)))
            ctx = ABXProposerContext(
                library_smiles_pool=ctx.library_smiles_pool[:insert_at] + [ans_smi] + ctx.library_smiles_pool[insert_at:],
                library_inchi_keys=ctx.library_inchi_keys[:insert_at] + [ans_ik] + ctx.library_inchi_keys[insert_at:],
                known_antibiotic_smiles=ctx.known_antibiotic_smiles,
                sealed_answer_smiles_or_none=ctx.sealed_answer_smiles_or_none,
                embeddings_path=ctx.embeddings_path,
                organism_active_pool_path=ctx.organism_active_pool_path,
            )
            _log(f"  injected hidden hit at position {insert_at}")

        candidates = run_abx_ensemble(
            packet, ctx,
            ch_e_json=args.ch_e_json, ch_f_json=args.ch_f_json,
            max_pool=packet.candidate_context.pool_size_target,
        )
        closed_candidates = candidates
        open_candidates = decontam_pool(candidates, ans_smi, thresh=0.85)
        _log(f"  closed pool: {len(closed_candidates)} | open pool: {len(open_candidates)}")
        candidates = closed_candidates

        # Score with v4 ranker
        scored = score_v4(ranker, candidates, organism=organism, gram=gram,
                            spectrum=spectrum, device=device)
        # Apply multiplicative composite + novelty
        scored = apply_multiplicative_composite(
            scored, args.alpha, args.beta, args.novelty_weight,
            args.memorization_threshold,
        )
        # Inject rationale + evidence
        for r in scored:
            rationale = build_rationale(
                organism=organism,
                antibacterial_score=r["antibacterial_score"],
                cytotox_risk=r["cytotox_risk"],
                artifact_risk=r["artifact_risk"],
                failure_mode_probs=r["failure_mode_probs"],
                nearest_known_antibiotic_similarity=r.get("max_tanimoto_to_known_antibiotic"),
                nearest_training_active_similarity=r.get("max_tanimoto_to_organism_active"),
            )
            r["structured_rationale"] = rationale.to_dict()

        scored.sort(key=lambda r: -r["final_discovery_score"])

        # Closed-mode ranking metrics
        cr = closed_hard_ranking(
            scored, case_id=case_id, organism=organism,
            hidden_hit_smiles=ans_smi, library_size=len(scored),
        )

        # Open-mode metrics
        scored_open = score_v4(ranker, open_candidates, organism=organism, gram=gram,
                                 spectrum=spectrum, device=device)
        scored_open = apply_multiplicative_composite(
            scored_open, args.alpha, args.beta, args.novelty_weight,
            args.memorization_threshold,
        )
        scored_open.sort(key=lambda r: -r["final_discovery_score"])
        op = open_proposer(
            scored_open, case_id=case_id,
            hidden_hit_smiles=ans_smi,
            active_family_smiles=[ans_smi] if ans_smi else [],
            known_antibiotic_smiles=known_smiles,
        )

        # Save artifacts
        out_pq = args.out / f"{case_id}_top_candidates.parquet"
        pd.DataFrame(scored).to_parquet(out_pq, compression="zstd", index=False)
        (args.out / f"{case_id}_locked_prediction.json").write_text(json.dumps({
            "case_id": case_id, "organism": organism, "n_pool": len(scored),
            "top_k": args.top_k, "top_candidates": scored[:args.top_k],
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str))
        save_metrics(cr, args.out / f"{case_id}_closed_metrics.json")
        save_metrics(op, args.out / f"{case_id}_open_metrics.json")

        # Card
        card_lines = [f"# {case_id} — ABX v5 (FiLM ranker + focal + multiplicative)", ""]
        card_lines.append(f"**Organism:** {organism} ({gram}, spectrum={spectrum})")
        if ans_smi:
            card_lines.append(f"**Hidden answer SMILES:** `{ans_smi}`")
        card_lines.append(f"\n## Closed ranking verdict: **{cr.verdict}**")
        card_lines.append(f"Library size: {cr.library_size}  |  Hit rank: {cr.hidden_hit_rank}  |  Top-1pct: {cr.top_1_pct}")
        card_lines.append(f"\n## Top {args.top_k} candidates\n")
        card_lines.append("| Rank | final | base | sel | nov | memo? | ab | cyto | art | channel | SMILES |")
        card_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(scored[:args.top_k], 1):
            card_lines.append(
                f"| {i} | {r['final_discovery_score']:.4f} | {r['v4_base_score']:.4f} | "
                f"{r['v4_selectivity_factor']:.3f} | {r['v4_novelty']:.2f} | "
                f"{'Y' if r['v4_is_memorization'] else 'n'} | "
                f"{r['antibacterial_score']:.3f} | {r['cytotox_risk']:.3f} | "
                f"{r['artifact_risk']:.3f} | {r.get('channel','?')} | "
                f"`{(r.get('candidate_smiles') or '')[:60]}` |"
            )
        (args.out / f"{case_id}_card.md").write_text("\n".join(card_lines))

        _log(f"  closed: rank={cr.hidden_hit_rank} verdict={cr.verdict}  open: n_valid={op.n_valid}")
        summary[case_id] = {
            "closed_verdict": cr.verdict,
            "closed_hit_rank": cr.hidden_hit_rank,
            "closed_top_1_pct": cr.top_1_pct,
            "open_exact_hit_generated": op.exact_hit_generated,
            "open_family_count": op.active_family_analog_count,
            "n_pool": len(scored),
        }

    (args.out / "_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"\nDone. Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
