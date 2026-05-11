"""Phase ABX-7: end-to-end sealed-case inference.

For each of ABX-001/002/003:
  1. Load ABXChallengePacket from registry
  2. Build ABXProposerContext (library pool, known-antibiotic seeds, embeddings)
  3. Run 7-channel proposer ensemble (A/B/C/D retrieval + E/F generative-JSON
     + G diversity)
  4. Decontaminate against the sealed answer (Tanimoto >= 0.85 dropped)
  5. Score each candidate with the trained ABX ranker
  6. Sort by final_discovery_score
  7. Run closed_hard_ranking + open_proposer eval
  8. Output LockedPrediction + per-case Markdown card + metrics JSON

Run:
    cd ~/wolverine/rasyn
    python scripts/run_abx_sealed_cases.py \\
        --ranker rasyn/data/clean/abx_ranker_seed42/checkpoint.pt \\
        --ch-e-json /tmp/abx_ch_e.json \\
        --ch-f-json /tmp/abx_ch_f.json \\
        --library rasyn/data/clean/antibiotic/abx_molecules.parquet \\
        --facts rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --out rasyn/data/clean/abx_stage5_results
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

from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE, PAD
from train_abx_ranker import (
    ABXMultiHeadRanker, condition_vector, ORGANISM_LIST, GRAM_LIST, SPECTRUM_LIST,
    FAILURE_MODES, CONDITION_DIM, tokenize as _tok,
)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def packet_from_case(case) -> ABXChallengePacket:
    return ABXChallengePacket(
        case_id=case.case_id,
        organism_context=ABXOrganismContext(**case.organism_context.model_dump()),
        selectivity_context=ABXSelectivityContext(**case.selectivity_context.model_dump()),
        candidate_context=ABXCandidateContext(**case.candidate_context.model_dump()),
    )


def decontam_pool(candidates: list[dict], answer_smi: str | None, thresh: float = 0.85) -> list[dict]:
    if not answer_smi:
        return candidates
    try:
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
            if DataStructs.TanimotoSimilarity(fp, ans_fp) < thresh:
                keep.append(c)
        return keep
    except ImportError:
        return candidates


def load_ranker(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ABXMultiHeadRanker(
        VOCAB_SIZE,
        d_model=args.get("d_model", 1024),
        n_heads=args.get("n_heads", 16),
        n_layers=args.get("n_layers", 16),
        max_len=args.get("max_len", 128),
    ).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ckpt


def score_candidates(
    model_or_models, candidates: list[dict],
    *, organism: str, gram: str, spectrum: str,
    device, max_len: int = 128, bs: int = 32,
) -> list[dict]:
    """Score candidates with one or many ranker ckpts; ensemble averages outputs.

    Returns per-candidate dicts including the standard scores plus
    `antibacterial_score_std` and `uncertainty_ensemble` when >1 ckpt was provided.
    """
    models = model_or_models if isinstance(model_or_models, list) else [model_or_models]
    out_rows: list[dict] = []
    cond = condition_vector(organism, gram, spectrum)
    cond_t = torch.from_numpy(cond).to(device)
    for i in range(0, len(candidates), bs):
        chunk = candidates[i:i+bs]
        ids_list, mask_list = [], []
        for c in chunk:
            ids, mask = _tok(c.get("candidate_smiles") or "C", max_len)
            ids_list.append(ids); mask_list.append(mask)
        ids_t = torch.from_numpy(np.stack(ids_list)).to(device)
        mask_t = torch.from_numpy(np.stack(mask_list)).to(device)
        cond_b = cond_t.unsqueeze(0).expand(len(chunk), -1)
        per_model_out = []
        for m in models:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                per_model_out.append(m(ids_t, mask_t, cond_b))
        # Stack along new ensemble axis: shape (E, B)
        def _stack(key):
            return torch.stack([o[key].float().cpu() for o in per_model_out])
        ab_stack = _stack("antibacterial")
        cyto_stack = _stack("cytotox")
        art_stack = _stack("artifact")
        fm_stack = torch.stack([o["failure_modes"].float().softmax(-1).cpu() for o in per_model_out])
        ab_mean = ab_stack.mean(0); ab_std = ab_stack.std(0) if ab_stack.size(0) > 1 else torch.zeros_like(ab_mean)
        cy_mean = cyto_stack.mean(0); art_mean = art_stack.mean(0); fm_mean = fm_stack.mean(0)
        for k, c in enumerate(chunk):
            final = float(ab_mean[k]) - 0.5 * float(cy_mean[k]) - 0.3 * float(art_mean[k])
            fm_dict = dict(zip(FAILURE_MODES, fm_mean[k].tolist()))
            rationale = build_rationale(
                organism=organism,
                antibacterial_score=float(ab_mean[k]),
                cytotox_risk=float(cy_mean[k]),
                artifact_risk=float(art_mean[k]),
                failure_mode_probs=fm_dict,
                nearest_known_antibiotic_similarity=c.get("max_tanimoto_to_known_antibiotic"),
                nearest_training_active_similarity=c.get("max_tanimoto_to_organism_active"),
                uncertainty_score=float(ab_std[k]) if len(models) > 1 else None,
            )
            evidence = build_evidence_packet(
                candidate_smiles=c.get("candidate_smiles") or "",
                organism=organism,
                antibacterial_scores_by_organism={organism: float(ab_mean[k])},
                cytotox_risk=float(cy_mean[k]),
                artifact_risk=float(art_mean[k]),
                uncertainty_score=float(ab_std[k]) if len(models) > 1 else None,
                nearest_known_antibiotic_similarity=c.get("max_tanimoto_to_known_antibiotic"),
                nearest_training_active_similarity=c.get("max_tanimoto_to_organism_active"),
                proposer_sources=[c.get("channel")] if c.get("channel") else None,
            )
            out_rows.append({
                **c,
                "antibacterial_score": float(ab_mean[k]),
                "antibacterial_score_std": float(ab_std[k]),
                "cytotox_risk": float(cy_mean[k]),
                "artifact_risk": float(art_mean[k]),
                "failure_mode_probs": fm_dict,
                "uncertainty_ensemble": float(ab_std[k]) if len(models) > 1 else None,
                "final_discovery_score": float(final),
                "structured_rationale": rationale.to_dict(),
                "evidence_packet": evidence,
            })
    return out_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranker", type=Path, required=True,
                   help="Ranker checkpoint path OR comma-separated list of paths for ensemble averaging.")
    p.add_argument("--library", type=Path, required=True,
                   help="ABX molecule pool (abx_molecules.parquet)")
    p.add_argument("--facts", type=Path, required=True,
                   help="Antibacterial assay facts (for organism-active retrieval in Ch C)")
    p.add_argument("--known-antibiotics", type=Path, default=None,
                   help="Optional parquet with column 'canonical_smiles' of known antibiotics (for Ch D)")
    p.add_argument("--ch-e-json", type=Path, default=None)
    p.add_argument("--ch-f-json", type=Path, default=None)
    p.add_argument("--cases", default="ABX-001,ABX-002,ABX-003")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    # Support comma-separated ranker checkpoints for ensemble averaging (spec §12).
    ranker_paths = [Path(p.strip()) for p in str(args.ranker).split(",") if p.strip()]
    _log(f"Loading {len(ranker_paths)} ranker ckpt(s): {[str(p) for p in ranker_paths]}")
    rankers = [load_ranker(p, device)[0] for p in ranker_paths]
    ranker = rankers[0] if len(rankers) == 1 else rankers  # downstream score_candidates handles both

    library_df = pd.read_parquet(args.library)
    library_smiles = library_df["canonical_smiles"].dropna().astype(str).tolist()
    library_ik = library_df["inchi_key"].astype(str).tolist() if "inchi_key" in library_df.columns else [None] * len(library_smiles)

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
            _log(f"  {case_id} not in registry; skipping.")
            continue
        _log(f"\n===== {case_id} =====")
        packet = packet_from_case(case)
        organism = packet.organism_context.organism
        gram = packet.organism_context.gram_type
        spectrum = packet.organism_context.spectrum_goal

        # Build proposer context
        ctx = ABXProposerContext(
            library_smiles_pool=library_smiles,
            library_inchi_keys=library_ik,
            known_antibiotic_smiles=known_smiles,
            sealed_answer_smiles_or_none=case.hidden_solution.get("canonical_smiles"),
            organism_active_pool_path=args.facts,
        )

        # If hidden answer SMILES known, INJECT it into library_smiles_pool for closed-mode eval.
        # Spec §15.1: "the system receives a fixed candidate library containing the hidden hit".
        ans_smi = case.hidden_solution.get("canonical_smiles")
        ans_ik = case.hidden_solution.get("inchi_key")
        if ans_smi and ans_smi not in ctx.library_smiles_pool:
            # Insert at random position so Channel A's slice has a chance to include it.
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
            _log(f"  injected hidden hit at position {insert_at} for closed-mode eval")

        # Run ensemble
        candidates = run_abx_ensemble(
            packet, ctx,
            ch_e_json=args.ch_e_json,
            ch_f_json=args.ch_f_json,
            max_pool=packet.candidate_context.pool_size_target,
        )
        _log(f"  pool after ensemble + diversity: {len(candidates)}")

        # CLOSED-mode candidates: pool WITH hidden hit (no decontam against answer)
        closed_candidates = candidates
        # OPEN-mode candidates: pool WITHOUT hidden hit (decontam against answer)
        open_candidates = decontam_pool(candidates, ans_smi, thresh=0.85)
        _log(f"  closed-mode pool: {len(closed_candidates)} | open-mode pool: {len(open_candidates)}")
        # Use CLOSED for scoring/ranking metrics (closed_hard_ranking expects hit IN pool)
        candidates = closed_candidates

        if not candidates:
            _log(f"  no candidates; skipping {case_id}")
            continue

        # Score
        scored = score_candidates(
            ranker, candidates,
            organism=organism, gram=gram, spectrum=spectrum,
            device=device,
        )
        scored.sort(key=lambda r: -r["final_discovery_score"])

        # Closed-ranking eval
        # (treat the full ranked list as the library; hidden_hit not in list since decontam removed it)
        # closed_hard_ranking expects the hidden hit to be IN the list — we set it None to compute baseline metrics
        cr = closed_hard_ranking(
            scored, case_id=case_id, organism=organism,
            hidden_hit_smiles=ans_smi,
            library_size=len(scored),
        )
        # Open-mode eval — use open_candidates (decontaminated)
        # Re-score the open-mode pool so 'scored' for closed and 'scored_open' for open are distinct.
        scored_open = score_candidates(
            ranker, open_candidates,
            organism=organism, gram=gram, spectrum=spectrum,
            device=device,
        )
        scored_open.sort(key=lambda r: -r["final_discovery_score"])
        op = open_proposer(
            scored_open, case_id=case_id,
            hidden_hit_smiles=ans_smi,
            active_family_smiles=[ans_smi] if ans_smi else [],
            known_antibiotic_smiles=known_smiles,
        )

        # Save per-case artifacts
        out_pq = args.out / f"{case_id}_top_candidates.parquet"
        pd.DataFrame(scored).to_parquet(out_pq, compression="zstd", index=False)

        out_locked = args.out / f"{case_id}_locked_prediction.json"
        out_locked.write_text(json.dumps({
            "case_id": case_id,
            "organism": organism,
            "n_pool": len(scored),
            "top_k": args.top_k,
            "top_candidates": scored[:args.top_k],
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str))

        save_metrics(cr, args.out / f"{case_id}_closed_metrics.json")
        save_metrics(op, args.out / f"{case_id}_open_metrics.json")

        # Markdown card
        card_lines = [f"# {case_id} — ABX sealed-case inference card", ""]
        card_lines.append(f"**Organism:** {organism} ({gram}, spectrum={spectrum})")
        if ans_smi:
            card_lines.append(f"**Hidden answer SMILES (injected for closed-mode rank, decontam'd for open-mode):** `{ans_smi}`")
        card_lines.append("")
        card_lines.append(f"## Closed ranking verdict: **{cr.verdict}**")
        card_lines.append(f"Library size: {cr.library_size}  |  Hit rank: {cr.hidden_hit_rank}  |  Top-1pct: {cr.top_1_pct}")
        card_lines.append("")
        card_lines.append(f"## Top {args.top_k} candidates")
        card_lines.append("")
        card_lines.append("| Rank | final | antibacterial | cytotox | artifact | channel | SMILES |")
        card_lines.append("|---|---|---|---|---|---|---|")
        for i, r in enumerate(scored[:args.top_k], 1):
            card_lines.append(
                f"| {i} | {r['final_discovery_score']:.3f} | {r['antibacterial_score']:.3f} | "
                f"{r['cytotox_risk']:.3f} | {r['artifact_risk']:.3f} | {r.get('channel','?')} | "
                f"`{(r.get('candidate_smiles') or '')[:80]}` |"
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
