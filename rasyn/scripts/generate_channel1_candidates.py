"""Channel 1: analog retrieval via Stage-1 backbone embeddings.

Encodes parent SMILES via the Stage-1 200M backbone, then performs cosine-
similarity nearest-neighbor search over the precomputed chembl_embeddings_200m
index (2.47M molecules × 1024-dim float16). Returns top-K SMILES per case as
a small JSON, same pattern as generate_channel_candidates.py.

Run on Pod B (which has the 5 GB embeddings index + molecules_canonical parquet
+ Stage-1 200M ckpt locally):
    cd ~/wolverine/rasyn && source .venv/bin/activate
    python scripts/generate_channel1_candidates.py \\
        --backbone rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --embeddings-dir rasyn/data/clean/chembl_embeddings_200m \\
        --molecules rasyn/data/clean/molecules_canonical.parquet \\
        --registry rasyn/data/registry/sealed_case_registry.yaml \\
        --cases ADMET-001,ADMET-002 \\
        --top-k 200 \\
        --out /tmp/ch1_candidates.json
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
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from h200_smiles_lm_pretrain import SMILESEncoder, VOCAB, VOCAB_SIZE, PAD


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tokenize(smi: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    UNK = 1
    ids = [VOCAB.get(c, UNK) for c in smi[:max_len]]
    n = len(ids)
    ids = ids + [PAD] * (max_len - n)
    attn = np.zeros(max_len, dtype=bool)
    attn[:n] = True
    return np.asarray(ids, dtype=np.int64), attn


@torch.no_grad()
def encode_parent(model: SMILESEncoder, smi: str, device: torch.device, max_len: int) -> np.ndarray:
    """Encode one SMILES via the Stage-1 backbone, mean-pool over non-pad tokens."""
    ids, mask = tokenize(smi, max_len)
    ids_t = torch.from_numpy(ids).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        T = ids_t.size(1)
        pos = torch.arange(T, device=device).unsqueeze(0).expand_as(ids_t)
        x = model.tok_emb(ids_t) + model.pos_emb(pos)
        x = model.encoder(x, src_key_padding_mask=~mask_t)
        x = model.norm(x)
        m = mask_t.unsqueeze(-1).float()
        emb = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
    return emb.float().cpu().numpy()[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=Path, required=True,
                   help="Stage-1 200M backbone checkpoint")
    p.add_argument("--embeddings-dir", type=Path, required=True,
                   help="Directory with embeddings.npy + index.parquet")
    p.add_argument("--molecules", type=Path, required=True,
                   help="molecules_canonical.parquet (for chembl_id -> SMILES lookup)")
    p.add_argument("--registry", type=Path,
                   default=Path("rasyn/data/registry/sealed_case_registry.yaml"))
    p.add_argument("--cases", default="ADMET-001,ADMET-002")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    _log(f"Loading Stage-1 backbone from {args.backbone}")
    ckpt = torch.load(args.backbone, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    d_model = cargs.get("d_model", 1024)
    n_heads = cargs.get("n_heads", 16)
    n_layers = cargs.get("n_layers", 16)
    max_len = cargs.get("max_len", 128)
    model = SMILESEncoder(VOCAB_SIZE, d_model, n_heads, n_layers, max_len).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    _log(f"  arch: d={d_model} h={n_heads} L={n_layers} max_len={max_len}")

    _log(f"Loading embeddings index from {args.embeddings_dir}")
    idx_df = pd.read_parquet(args.embeddings_dir / "index.parquet")
    embeddings = np.load(args.embeddings_dir / "embeddings.npy", mmap_mode="r")
    chembl_ids = idx_df["chembl_id"].astype(str).tolist()
    _log(f"  loaded {embeddings.shape} embeddings | {len(chembl_ids):,} chembl_ids")

    _log(f"Loading molecules parquet from {args.molecules}")
    mols = pd.read_parquet(args.molecules)
    smiles_by_id = dict(zip(mols["chembl_id"].astype(str), mols["canonical_smiles"]))
    _log(f"  loaded {len(smiles_by_id):,} SMILES")

    reg = yaml.safe_load(args.registry.read_text())
    cases_by_id = {c["case_id"]: c for c in reg["cases"]}

    # Pre-normalize embeddings for cosine similarity
    # (cosine sim = dot(parent_norm, all_embeddings_norm))
    _log("Pre-normalizing embeddings (~5 GB float16 -> float32)...")
    # To avoid 5GB×2 RAM, compute norms once and normalize in-place batched
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=1, keepdims=True)  # (N, 1)
    norms = norms.astype(np.float32)
    _log(f"  done: norms shape {norms.shape}")

    results: dict = {"channel": "analog_retrieval", "backbone": str(args.backbone), "cases": {}}
    for case_id in args.cases.split(","):
        case_id = case_id.strip()
        case = cases_by_id.get(case_id)
        if not case:
            _log(f"  case {case_id} not in registry; skipping")
            continue
        parent = case.get("parent") or case.get("parent_compound", {})
        parent_smiles = parent.get("canonical_smiles")
        if not parent_smiles:
            _log(f"  {case_id}: parent SMILES null; skipping")
            continue
        _log(f"[{case_id}] parent={parent_smiles[:60]}...")
        t0 = time.time()
        parent_emb = encode_parent(model, parent_smiles, device, max_len)
        parent_norm = parent_emb / (np.linalg.norm(parent_emb) + 1e-8)

        # Cosine similarity: parent_norm @ (embeddings.T / norms.T)
        # Batch in chunks to avoid memory spike.
        N = embeddings.shape[0]
        chunk_size = 200_000
        sims = np.zeros(N, dtype=np.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            emb_chunk = embeddings[start:end].astype(np.float32)
            norm_chunk = norms[start:end]
            sims[start:end] = (emb_chunk @ parent_norm) / (norm_chunk[:, 0] + 1e-8)

        top_idx = np.argpartition(sims, -args.top_k)[-args.top_k:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        candidates_sm: list[str] = []
        for i in top_idx:
            cid = chembl_ids[i]
            sm = smiles_by_id.get(cid)
            if sm and sm != parent_smiles:
                candidates_sm.append(sm)

        _log(f"  top {len(candidates_sm)} similar (top score {sims[top_idx[0]]:.3f}) in {time.time()-t0:.1f}s")
        results["cases"][case_id] = {
            "parent_smiles": parent_smiles,
            "top_k_requested": args.top_k,
            "n_returned": len(candidates_sm),
            "candidates": candidates_sm,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    _log(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
