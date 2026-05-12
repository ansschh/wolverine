"""Best-of-N: sample many SMILES from the AR LM, filter drug-like, score with v4 ranker.

This is a more stable alternative to RL fine-tuning. We:
  1. Sample N SMILES from the AR LM (temperature 1.0, top-p 0.95)
  2. Validate + filter to drug-like (no '.', 8 <= n_heavy <= 60, >= 1 ring)
  3. Canonicalize + dedupe
  4. Score each with the v4 ranker for the target organism
  5. Surface top-K by:
       - raw antibacterial score
       - multiplicative composite (ab * (1-cyto)^2 * (1-art))
       - surprise (composite * (1 - max_Tan_to_training_actives))

Output: JSON with all valid samples + their scores, plus top-K cards per metric.

Run on 1 GPU (~10 min for 10K samples):
    python scripts/sample_and_score_ar_lm.py \\
        --ar-lm rasyn/data/clean/smiles_ar_lm_200m/checkpoint.pt \\
        --ranker rasyn/data/clean/abx_ranker_v4_seed42/checkpoint.pt \\
        --facts rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --organism E.coli --gram Gram-negative \\
        --spectrum broad_spectrum_or_general_antibacterial \\
        --n-samples 10000 --batch 64 \\
        --temperature 1.0 --top-p 0.95 \\
        --top-k 50 \\
        --out artifacts/abx_lm_generate_ecoli
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

from train_smiles_ar_lm import ARSMILESLM, VOCAB_SIZE  # noqa: E402
from train_abx_ranker_v4 import ABXMultiHeadRankerV4  # noqa: E402
from train_abx_ranker import condition_vector, tokenize as ranker_tok  # noqa: E402
from h200_smiles_lm_pretrain import VOCAB_SIZE as RANKER_VOCAB_SIZE  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def filter_druglike(smi: str) -> str | None:
    if not smi or "." in smi or len(smi) > 150:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        n_heavy = m.GetNumHeavyAtoms()
        if n_heavy < 8 or n_heavy > 60:
            return None
        if rdMolDescriptors.CalcNumRings(m) < 1:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def _morgan_fp(smi: str, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits) if m else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ar-lm", type=Path, required=True)
    p.add_argument("--ranker", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--organism", required=True)
    p.add_argument("--gram", default="Gram-negative")
    p.add_argument("--spectrum", default="broad_spectrum_or_general_antibacterial")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--n-ref-actives", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    _log(f"Loading AR LM {args.ar_lm}")
    ckpt_lm = torch.load(args.ar_lm, map_location="cpu", weights_only=False)
    cargs_lm = ckpt_lm.get("args", {})
    lm = ARSMILESLM(
        d_model=cargs_lm.get("d_model", 1024),
        n_heads=cargs_lm.get("n_heads", 16),
        n_layers=cargs_lm.get("n_layers", 16),
        max_len=cargs_lm.get("max_len", 128),
    ).to(device).eval()
    sd_lm = {k.removeprefix("module."): v for k, v in ckpt_lm["model"].items()}
    lm.load_state_dict(sd_lm, strict=True)

    _log(f"Loading v4 ranker {args.ranker}")
    ckpt_r = torch.load(args.ranker, map_location="cpu", weights_only=False)
    cargs_r = ckpt_r.get("args", {})
    ranker = ABXMultiHeadRankerV4(
        RANKER_VOCAB_SIZE, d_model=1024, n_heads=16, n_layers=16,
        max_len=cargs_r.get("max_len", 128),
    ).to(device).eval()
    sd_r = {k.removeprefix("module."): v for k, v in ckpt_r["model"].items()}
    ranker.load_state_dict(sd_r, strict=False)

    _log(f"Loading training-active fingerprints (organism={args.organism})")
    facts_df = pd.read_parquet(args.facts)
    actives = facts_df[(facts_df["organism"] == args.organism) &
                       (facts_df["activity_label"] == "active")]
    active_smiles = actives["canonical_smiles"].dropna().astype(str).unique().tolist()[:args.n_ref_actives]
    active_fps = [_morgan_fp(s) for s in active_smiles if _morgan_fp(s) is not None]
    _log(f"  built {len(active_fps)} active fingerprints")

    # Sample
    _log(f"Sampling {args.n_samples} SMILES at temp={args.temperature}, top_p={args.top_p}")
    all_smiles: list[str] = []
    t_sample = time.time()
    n_done = 0
    while n_done < args.n_samples:
        bs = min(args.batch, args.n_samples - n_done)
        out = lm.sample(
            bs, max_len=args.max_len, temperature=args.temperature,
            top_p=args.top_p, device=device,
        )
        all_smiles.extend(out)
        n_done += bs
        if n_done % 1024 == 0 or n_done == args.n_samples:
            _log(f"  sampled {n_done}/{args.n_samples} ({(n_done)/(time.time()-t_sample):.0f}/sec)")
    _log(f"Sampling done in {time.time()-t_sample:.1f}s")

    # Filter validity + canonicalize
    _log("Filtering to drug-like canonicalized SMILES + deduping")
    valid_set: set[str] = set()
    for s in all_smiles:
        cs = filter_druglike(s)
        if cs:
            valid_set.add(cs)
    valid = sorted(valid_set)
    _log(f"  valid drug-like: {len(valid):,} / {len(all_smiles):,} ({100*len(valid)/len(all_smiles):.1f}%)")

    if not valid:
        _log("No valid samples; aborting.")
        return 1

    # Score with v4 ranker
    _log(f"Scoring with v4 ranker for {args.organism}...")
    cond = condition_vector(args.organism, args.gram, args.spectrum)
    cond_t = torch.from_numpy(cond).to(device)
    ab_arr = np.zeros(len(valid), dtype=np.float32)
    cy_arr = np.zeros(len(valid), dtype=np.float32)
    ar_arr = np.zeros(len(valid), dtype=np.float32)
    bs = 64
    for i in range(0, len(valid), bs):
        chunk = valid[i:i+bs]
        ids_list, mask_list = [], []
        for s in chunk:
            ids, mask = ranker_tok(s, args.max_len)
            ids_list.append(ids); mask_list.append(mask)
        ids_t = torch.from_numpy(np.stack(ids_list)).long().to(device)
        mask_t = torch.from_numpy(np.stack(mask_list)).bool().to(device)
        cond_b = cond_t.unsqueeze(0).expand(len(chunk), -1)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = ranker(ids_t, mask_t, cond_b)
        ab_arr[i:i+len(chunk)] = out["antibacterial"].float().cpu().numpy()
        cy_arr[i:i+len(chunk)] = out["cytotox"].float().cpu().numpy()
        ar_arr[i:i+len(chunk)] = out["artifact"].float().cpu().numpy()

    # Composite + Tanimoto-to-training (for top-K only — expensive)
    composite = ab_arr * (1 - cy_arr)**2 * (1 - ar_arr)
    # Top-300 by composite — compute Tanimoto for these for "surprise"
    top_n_for_tan = 300
    idx_top = np.argsort(composite)[-top_n_for_tan:][::-1]
    _log(f"Computing Tan-to-actives for top-{top_n_for_tan} composite")
    from rdkit.DataStructs import TanimotoSimilarity
    candidates = []
    for k, ci in enumerate(idx_top):
        smi = valid[ci]
        fp = _morgan_fp(smi)
        if fp is None:
            tan = 0.0
        else:
            tan = max((TanimotoSimilarity(fp, af) for af in active_fps), default=0.0)
        cand = {
            "smiles": smi,
            "ab_score": float(ab_arr[ci]),
            "cytotox_risk": float(cy_arr[ci]),
            "artifact_risk": float(ar_arr[ci]),
            "composite_score": float(composite[ci]),
            "max_tan_to_training_active": float(tan),
            "novelty": 1.0 - float(tan),
            "surprise_score": float(composite[ci]) * (1.0 - float(tan)),
        }
        candidates.append(cand)
        if k % 50 == 0:
            _log(f"  Tan {k}/{top_n_for_tan}")

    # Three rankings
    by_ab = sorted(candidates, key=lambda x: -x["ab_score"])[:args.top_k]
    by_composite = sorted(candidates, key=lambda x: -x["composite_score"])[:args.top_k]
    by_surprise = sorted(candidates, key=lambda x: -x["surprise_score"])[:args.top_k]

    pool_stats = {
        "n_sampled": len(all_smiles),
        "n_valid_unique": len(valid),
        "validity_rate": len(valid) / max(1, len(all_smiles)),
        "ab_score": {
            "mean": float(ab_arr.mean()),
            "p50": float(np.percentile(ab_arr, 50)),
            "p90": float(np.percentile(ab_arr, 90)),
            "p99": float(np.percentile(ab_arr, 99)),
            "max": float(ab_arr.max()),
        },
        "composite": {
            "mean": float(composite.mean()),
            "p50": float(np.percentile(composite, 50)),
            "p99": float(np.percentile(composite, 99)),
            "max": float(composite.max()),
        },
    }
    _log(f"Pool stats: ab_mean={pool_stats['ab_score']['mean']:.3f} ab_max={pool_stats['ab_score']['max']:.3f}")
    _log(f"            composite_mean={pool_stats['composite']['mean']:.4f} composite_max={pool_stats['composite']['max']:.4f}")

    summary = {
        "organism": args.organism,
        "n_ref_actives": len(active_fps),
        "pool_stats": pool_stats,
        "top_k_by_ab_score": by_ab,
        "top_k_by_composite": by_composite,
        "top_k_by_surprise": by_surprise,
    }
    (args.out / "lm_generate_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"Wrote {args.out / 'lm_generate_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
