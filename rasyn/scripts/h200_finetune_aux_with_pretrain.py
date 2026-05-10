"""Stage-2 finetune: load Stage-1 SMILES LM backbone + train ADMET heads.

Demonstrates the actual transfer-learning pipeline. Same multi-task ADMET
objective as `h200_train_aux_admet.py`, but the encoder is initialized from
a SMILES masked-LM pretrain checkpoint instead of random init.

Outputs (rasyn/data/clean/aux_finetuned/):
    - checkpoint.pt        (encoder + heads finetuned)
    - per_task_metrics.json
    - training_log.jsonl

Run (8x A100 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/h200_finetune_aux_with_pretrain.py \\
        --pretrain rasyn/data/clean/smiles_lm/checkpoint.pt \\
        --steps 4000 --bs 96
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

# Reuse model + dataset definitions from the aux script.
import sys
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from h200_train_aux_admet import (  # type: ignore[import-not-found]
    MultiTaskADMET,
    TDCMultiTaskDataset,
    TDC_DATASETS,
    VOCAB_SIZE,
    pull_tdc_combined,
)


def load_pretrain_into_model(model: MultiTaskADMET, pretrain_ckpt_path: Path, log) -> int:
    """Load the SMILES LM encoder weights into the aux model. Returns # tensors loaded."""
    ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
    pretrain_sd = ckpt["model"]
    model_sd = model.state_dict()
    n_loaded = 0
    n_skipped = 0
    for k, v in pretrain_sd.items():
        # SMILES LM keys: tok_emb.weight, pos_emb.weight, encoder.layers.X.*, norm.*, lm_head.*
        # Aux model keys: tok_emb.weight, pos_emb.weight, encoder.layers.X.*, norm.*, heads.X.*
        if k.startswith("lm_head"):
            continue  # different head structure
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            n_loaded += 1
        else:
            n_skipped += 1
    model.load_state_dict(model_sd)
    if log:
        log(f"Loaded {n_loaded} tensors from pretrain ({n_skipped} skipped)")
    return n_loaded


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True, help="SMILES LM checkpoint.pt")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-4, help="Lower than from-scratch (1e-4 vs 2e-4) for finetune.")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("rasyn/data/clean/aux_finetuned"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-decontam", action="store_true")
    p.add_argument("--freeze-encoder", action="store_true", help="Freeze backbone, train only heads.")
    args = p.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "training_log.jsonl"

    def log(msg, **extra):
        if is_main:
            payload = {"t": time.time(), "msg": str(msg), **extra}
            print(f"[{time.strftime('%H:%M:%S')}] [r{rank}] {msg}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Pull TDC data (rank 0 caches; others reload).
    smi_idx_path = args.out / "smiles_index.json"
    labels_path = args.out / "labels.npy"
    if is_main:
        smiles, labels, task_names, task_types = pull_tdc_combined(decontam=not args.no_decontam, log=log)
        smi_idx_path.write_text(json.dumps({"smiles": smiles, "task_names": task_names, "task_types": task_types}))
        np.save(labels_path, labels)
    if dist.is_initialized():
        dist.barrier()
    if not is_main:
        meta = json.loads(smi_idx_path.read_text())
        smiles, task_names, task_types = meta["smiles"], meta["task_names"], meta["task_types"]
        labels = np.load(labels_path)

    n_tasks = len(task_names)
    if is_main:
        log(f"World size {world_size} | n_tasks {n_tasks} | n_smiles {len(smiles)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(smiles))
    val_n = max(1024, len(smiles) // 10)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    tr_ds = TDCMultiTaskDataset([smiles[i] for i in tr_idx], labels[tr_idx], max_len=args.max_len)
    vl_ds = TDCMultiTaskDataset([smiles[i] for i in val_idx], labels[val_idx], max_len=args.max_len)

    sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, shuffle=(sampler is None),
                       num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    vl_dl = DataLoader(vl_ds, batch_size=args.bs * 2, num_workers=2, pin_memory=True, persistent_workers=True)

    device = torch.device(f"cuda:{local_rank}")
    model = MultiTaskADMET(
        n_tasks=n_tasks, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, max_len=args.max_len,
    ).to(device)

    # Load pretrained encoder weights
    if is_main:
        log(f"Loading pretrain from: {args.pretrain}")
    n_loaded = load_pretrain_into_model(model, args.pretrain, log if is_main else None)

    if args.freeze_encoder:
        for n, p in model.named_parameters():
            if not n.startswith("heads"):
                p.requires_grad = False
        if is_main:
            log("Froze encoder. Training only heads.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        log(f"Trainable params: {n_params:,} | Total tensors loaded from pretrain: {n_loaded}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95),
    )

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    is_classification = torch.tensor([t == "binary" for t in task_types], device=device)

    def step_fn(batch):
        ids, mask, y = [b.to(device, non_blocking=True) for b in batch]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
        logits = logits.float()
        valid = ~torch.isnan(y)
        if not valid.any():
            return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True), 0
        loss = torch.zeros((), device=device, dtype=torch.float32)
        n_active = 0
        for t in range(n_tasks):
            v = valid[:, t]
            if not v.any():
                continue
            if is_classification[t]:
                tloss = F.binary_cross_entropy_with_logits(logits[v, t], y[v, t].clamp(0, 1))
            else:
                tloss = F.mse_loss(logits[v, t], y[v, t])
            loss = loss + tloss
            n_active += 1
        loss = loss / max(1, n_active)
        return loss, int(valid.sum().item())

    @torch.no_grad()
    def evaluate():
        model.eval()
        per_task_sse = np.zeros(n_tasks)
        per_task_n = np.zeros(n_tasks)
        per_task_correct = np.zeros(n_tasks)
        for ids, mask, y in vl_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
            valid = ~torch.isnan(y)
            for t in range(n_tasks):
                v = valid[:, t]
                if not v.any():
                    continue
                if is_classification[t]:
                    pred = (logits[v, t] > 0).float()
                    per_task_correct[t] += (pred == y[v, t]).sum().item()
                else:
                    per_task_sse[t] += ((logits[v, t] - y[v, t]) ** 2).sum().item()
                per_task_n[t] += int(v.sum().item())
        model.train()
        out = {}
        for t in range(n_tasks):
            if per_task_n[t] == 0:
                continue
            if task_types[t] == "binary":
                out[task_names[t]] = {"acc": per_task_correct[t] / per_task_n[t], "n": int(per_task_n[t])}
            else:
                out[task_names[t]] = {"rmse": math.sqrt(per_task_sse[t] / per_task_n[t]), "n": int(per_task_n[t])}
        return out

    if is_main:
        log(f"Starting Stage-2 finetune: {args.steps} steps, bs={args.bs} per GPU, eff bs={args.bs * world_size}")
    t0 = time.time()
    step = 0
    epoch = 0
    model.train()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in tr_dl:
            for pg in optim.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            optim.zero_grad(set_to_none=True)
            loss, _ = step_fn(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                elapsed = time.time() - t0
                log(f"step {step}/{args.steps} loss={loss.item():.4f} lr={optim.param_groups[0]['lr']:.2e} thru={(step * args.bs * world_size) / elapsed:.0f} samp/s",
                    step=step, loss=float(loss.item()))
            if step % args.val_every == 0 and is_main:
                metrics = evaluate()
                log(f"step {step} val metrics", val=metrics)
                (args.out / "per_task_metrics.json").write_text(json.dumps(metrics, indent=2))
            if step % args.ckpt_every == 0 and is_main:
                torch.save({
                    "step": step,
                    "model": (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict(),
                    "args": vars(args),
                    "task_names": task_names,
                    "task_types": task_types,
                    "pretrain_path": str(args.pretrain),
                }, args.out / "checkpoint.pt")
                log(f"Saved checkpoint at step {step}")
            if step >= args.steps:
                break
        epoch += 1

    if is_main:
        metrics = evaluate()
        (args.out / "per_task_metrics.json").write_text(json.dumps(metrics, indent=2))
        torch.save({
            "step": step,
            "model": (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict(),
            "args": vars(args),
            "task_names": task_names,
            "task_types": task_types,
            "pretrain_path": str(args.pretrain),
        }, args.out / "checkpoint.pt")
        log("FINAL", val=metrics, total_seconds=time.time() - t0)

    cleanup_distributed()


if __name__ == "__main__":
    main()
