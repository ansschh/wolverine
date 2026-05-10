"""Precompute aux ADMET predictions for the entire ChEMBL canonical molecule set.

22-dim ADMET prediction vector per molecule, corresponding to TDC's 22 ADMET
endpoints (15 ADME + 7 Tox). Used for:
  - Channel 1 candidate scoring at Stage-5 inference
  - Stage-2 evidence features per pair
  - Candidate prefiltering by predicted ADMET

Output (under --out dir):
  - predictions.npy   (N, 22) float16
  - index.parquet     (chembl_id, row_idx)
  - meta.json         (task names + types + ckpt + sigmoid-policy)

Run on Pod B (8x A100 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/precompute_chembl_aux_predictions.py \\
        --ckpt rasyn/data/clean/aux_finetuned_frozen/checkpoint.pt \\
        --data rasyn/data/clean/molecules_canonical.parquet \\
        --out rasyn/data/clean/chembl_aux_predictions \\
        --bs 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from h200_train_aux_admet import MultiTaskADMET, VOCAB, VOCAB_SIZE, PAD, TDC_DATASETS


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def tokenize_aux(smi: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    UNK = 1
    ids = [VOCAB.get(c, UNK) for c in smi[:max_len]]
    n = len(ids)
    ids = ids + [PAD] * (max_len - n)
    attn = np.zeros(max_len, dtype=bool)
    attn[:n] = True
    return np.asarray(ids, dtype=np.int64), attn


class AuxInferenceDataset(Dataset):
    def __init__(self, smiles: list[str], max_len: int = 128):
        self.smiles = smiles
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ids, attn = tokenize_aux(self.smiles[idx], self.max_len)
        return torch.from_numpy(ids), torch.from_numpy(attn), idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    def log(msg):
        if is_main:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Loading checkpoint {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    d_model = ckpt_args.get("d_model", 768)
    n_heads = ckpt_args.get("n_heads", 12)
    n_layers = ckpt_args.get("n_layers", 8)
    max_len = max(ckpt_args.get("max_len", args.max_len), args.max_len)
    task_names = ckpt.get("task_names") or [n for _, n, _ in TDC_DATASETS]
    task_types = ckpt.get("task_types") or [t for _, _, t in TDC_DATASETS]
    n_tasks = len(task_names)

    log(f"  Architecture: d_model={d_model} n_heads={n_heads} n_layers={n_layers} max_len={max_len}")
    log(f"  Tasks: {n_tasks}")

    model = MultiTaskADMET(
        n_tasks=n_tasks, d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_len=max_len,
    ).to(device)
    state_dict = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    log(f"Loading molecules from {args.data}")
    df = pd.read_parquet(args.data)
    chembl_ids = df["chembl_id"].astype(str).tolist()
    smiles = df["canonical_smiles"].astype(str).tolist()
    log(f"  Loaded {len(smiles):,} molecules")

    args.out.mkdir(parents=True, exist_ok=True)

    ds = AuxInferenceDataset(smiles, max_len=max_len)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
    dl = DataLoader(
        ds, batch_size=args.bs, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    log(f"Predicting {n_tasks} ADMET endpoints (world={world_size}, bs/rank={args.bs})...")

    is_classification = np.array([t == "binary" for t in task_types])
    predictions = np.zeros((len(ds), n_tasks), dtype=np.float32)
    indices_seen = np.zeros(len(ds), dtype=bool)
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for ids, attn, idx in dl:
            ids = ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, attn)
            logits = logits.float().cpu().numpy()
            for t, is_cls in enumerate(is_classification):
                if is_cls:
                    logits[:, t] = 1.0 / (1.0 + np.exp(-logits[:, t]))
            for k, i in enumerate(idx.tolist()):
                predictions[i] = logits[k]
                indices_seen[i] = True
            n_done += len(idx)
            if rank == 0 and (n_done // args.bs) % 50 == 0:
                elapsed = time.time() - t0
                log(f"  predicted {n_done:,} on rank0 | {n_done / max(elapsed, 1):.0f} samp/s")

    seen_idx = np.where(indices_seen)[0]
    np.savez(args.out / f"_shard_rank{rank}.npz",
             indices=seen_idx, predictions=predictions[seen_idx])
    if dist.is_initialized():
        dist.barrier()

    if is_main:
        all_indices = []
        all_pred = []
        for r in range(world_size):
            sh = np.load(args.out / f"_shard_rank{r}.npz")
            all_indices.append(sh["indices"])
            all_pred.append(sh["predictions"])
        idx_arr = np.concatenate(all_indices)
        pred_arr = np.concatenate(all_pred)
        sort = np.argsort(idx_arr)
        idx_arr = idx_arr[sort]
        pred_arr = pred_arr[sort]

        np.save(args.out / "predictions.npy", pred_arr.astype(np.float16))
        index_df = pd.DataFrame({
            "chembl_id": [chembl_ids[i] for i in idx_arr],
            "row_idx": np.arange(len(idx_arr), dtype=np.int64),
        })
        index_df.to_parquet(args.out / "index.parquet", compression="zstd", index=False)
        meta = {
            "ckpt_path": str(args.ckpt),
            "task_names": task_names,
            "task_types": task_types,
            "binary_tasks_use_sigmoid": True,
            "n_molecules": int(len(idx_arr)),
            "dtype": "float16",
        }
        (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

        for r in range(world_size):
            (args.out / f"_shard_rank{r}.npz").unlink()

        log(f"Saved predictions ({pred_arr.shape}, float16) -> {args.out}/")
        log(f"Total wall-clock: {(time.time() - t0) / 60:.1f} min")

    cleanup_distributed()


if __name__ == "__main__":
    main()
