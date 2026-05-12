"""Round 3: score the v4 ranker on a broad EXTERNAL chemistry pool.

Goal: demonstrate that the v4 ranker can surface non-trivial antibacterial-
candidate molecules from a chemistry space outside its antibac training set.

Procedure:
  1. Load chembl_all_smiles.parquet (~2.8M molecules — broader chemistry).
  2. Subtract abx_molecules.parquet (175K — already-tested) → external pool.
  3. Random-sample N_SAMPLE molecules (default 50K).
  4. Score each with the v4 ranker for organism X (default E.coli + A.baumannii).
  5. For each organism, return:
       - Top-K by ab_score (raw confidence)
       - Top-K by composite final_score (multiplicative + novelty)
       - Top-K *surprises*: high ab + low Tanimoto to nearest training-active

The "surprises" are the active-learning loop's labeling candidates. In a full
round 3 these would be pseudo-labeled by aux predictor consensus and added
to the next training round.

Run on pod:
    python scripts/abx_round3_score_external.py \\
        --ranker rasyn/data/clean/abx_ranker_v4_seed42/checkpoint.pt \\
        --all-chembl rasyn/data/clean/chembl_all_smiles.parquet \\
        --abx-mols rasyn/data/clean/antibiotic/abx_molecules.parquet \\
        --facts rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --organisms E.coli,A.baumannii \\
        --n-sample 50000 --top-k 25 \\
        --out artifacts/abx_round3_external
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from train_abx_ranker_v4 import ABXMultiHeadRankerV4  # noqa: E402
from train_abx_ranker import condition_vector, ORGANISM_LIST, tokenize as _tok  # noqa: E402
from h200_smiles_lm_pretrain import VOCAB_SIZE  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _morgan_fp(smi: str, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits) if m else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ranker", type=Path, required=True)
    p.add_argument("--all-chembl", type=Path, required=True)
    p.add_argument("--abx-mols", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--organisms", default="E.coli,A.baumannii")
    p.add_argument("--n-sample", type=int, default=50000)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _log("Loading molecule pools...")
    all_smiles = pd.read_parquet(args.all_chembl)["canonical_smiles"].astype(str).tolist()
    _log(f"  chembl_all: {len(all_smiles):,}")
    abx = pd.read_parquet(args.abx_mols)
    abx_set = set(abx["canonical_smiles"].dropna().astype(str).tolist())
    _log(f"  abx_molecules (to exclude): {len(abx_set):,}")
    external = [s for s in all_smiles if s not in abx_set]
    _log(f"  external pool: {len(external):,}")

    rng = np.random.default_rng(args.seed)
    if len(external) > args.n_sample:
        idx = rng.choice(len(external), size=args.n_sample, replace=False)
        external = [external[i] for i in idx]
    _log(f"  scoring {len(external):,} external molecules")

    _log("Loading v4 ranker...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ranker, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    model = ABXMultiHeadRankerV4(
        VOCAB_SIZE, d_model=1024, n_heads=16, n_layers=16,
        max_len=cargs.get("max_len", 128),
    ).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    _log("Loading facts for training-active Tanimoto reference...")
    facts_df = pd.read_parquet(args.facts)

    summary: dict = {"n_external_scored": len(external)}

    for organism in args.organisms.split(","):
        organism = organism.strip()
        _log(f"\n===== Scoring against organism={organism} =====")
        cond = condition_vector(organism, "Gram-negative", "broad_spectrum_or_general_antibacterial")
        cond_t = torch.from_numpy(cond).to(device)

        # Score all external molecules
        ab_all = np.zeros(len(external), dtype=np.float32)
        cy_all = np.zeros(len(external), dtype=np.float32)
        ar_all = np.zeros(len(external), dtype=np.float32)
        for i in range(0, len(external), args.bs):
            chunk = external[i:i+args.bs]
            ids_list, mask_list = [], []
            for s in chunk:
                ids, mask = _tok(s or "C", 128)
                ids_list.append(ids); mask_list.append(mask)
            ids_t = torch.from_numpy(np.stack(ids_list)).long().to(device)
            mask_t = torch.from_numpy(np.stack(mask_list)).bool().to(device)
            cond_b = cond_t.unsqueeze(0).expand(len(chunk), -1)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids_t, mask_t, cond_b)
            ab_all[i:i+len(chunk)] = out["antibacterial"].float().cpu().numpy()
            cy_all[i:i+len(chunk)] = out["cytotox"].float().cpu().numpy()
            ar_all[i:i+len(chunk)] = out["artifact"].float().cpu().numpy()
            if (i // args.bs) % 50 == 0:
                _log(f"  scored {i+len(chunk):,}/{len(external):,}")

        # Composite: ab * (1-cyto)^2 * (1-art)
        composite = ab_all * (1 - cy_all)**2 * (1 - ar_all)

        # Get top-K by raw ab_score and by composite
        top_ab_idx = np.argsort(ab_all)[-args.top_k:][::-1]
        top_comp_idx = np.argsort(composite)[-args.top_k:][::-1]

        # Compute Tanimoto to nearest training-active for "surprise" detection
        _log("  computing Tan-to-nearest-training-active for top candidates...")
        from rdkit.DataStructs import TanimotoSimilarity
        org_actives = facts_df[
            (facts_df["organism"] == organism) & (facts_df["activity_label"] == "active")
        ]
        active_smiles = org_actives["canonical_smiles"].dropna().astype(str).unique().tolist()[:5000]
        active_fps = [_morgan_fp(s) for s in active_smiles if _morgan_fp(s) is not None]
        _log(f"  built {len(active_fps)} active fingerprints for {organism}")

        candidates_with_tan = []
        # Look at top 200 by composite — compute Tan-to-actives for each
        for ci in np.argsort(composite)[-200:][::-1]:
            smi = external[ci]
            fp = _morgan_fp(smi)
            if fp is None:
                tan = 0.0
            else:
                tan = max((TanimotoSimilarity(fp, af) for af in active_fps), default=0.0)
            candidates_with_tan.append({
                "smiles": smi,
                "ab_score": float(ab_all[ci]),
                "cytotox_risk": float(cy_all[ci]),
                "artifact_risk": float(ar_all[ci]),
                "composite_score": float(composite[ci]),
                "max_tan_to_training_active": float(tan),
                "novelty": 1.0 - float(tan),
                "surprise_score": float(composite[ci]) * (1.0 - float(tan)),  # high comp × high novelty
            })
        # Top-K by surprise (high composite AND high novelty)
        candidates_with_tan.sort(key=lambda x: -x["surprise_score"])
        top_surprises = candidates_with_tan[:args.top_k]

        # Top-K by composite (no novelty bonus)
        candidates_with_tan_by_comp = sorted(candidates_with_tan, key=lambda x: -x["composite_score"])
        top_composite = candidates_with_tan_by_comp[:args.top_k]

        # Pool stats
        pool_stats = {
            "n_scored": len(external),
            "ab_score_mean": float(ab_all.mean()),
            "ab_score_p99": float(np.percentile(ab_all, 99)),
            "ab_score_max": float(ab_all.max()),
            "composite_mean": float(composite.mean()),
            "composite_p99": float(np.percentile(composite, 99)),
            "composite_max": float(composite.max()),
        }

        summary[organism] = {
            "pool_stats": pool_stats,
            "top_k_by_composite": top_composite,
            "top_k_by_surprise": top_surprises,
        }
        _log(f"  pool stats: ab_mean={pool_stats['ab_score_mean']:.3f} comp_max={pool_stats['composite_max']:.4f}")

    (args.out / "external_scoring_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"\nWrote {args.out / 'external_scoring_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
