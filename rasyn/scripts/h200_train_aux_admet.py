"""H200 multi-task ADMET predictor pretrain.

Pulls TDC's 22 ADMET endpoints, decontaminates against the sealed-case
registry, trains a transformer encoder + 22 multi-task heads via DDP across
all available GPUs, saves checkpoint + per-task metrics.

Run (8x H200 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 scripts/h200_train_aux_admet.py \\
        --steps 5000 --bs 128 --d-model 768 --n-layers 8

Outputs (under rasyn/data/clean/aux_admet/):
    - checkpoint.pt
    - training_log.jsonl
    - per_task_metrics.json
    - smiles_index.json (decontaminated SMILES universe)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from rasyn.data.decontam.quarantine import scrub_rows
from rasyn.data.registry.loader import load_sealed_case_registry

TDC_DATASETS = [
    ("ADME", "Caco2_Wang", "regression"),
    ("ADME", "PAMPA_NCATS", "binary"),
    ("ADME", "HIA_Hou", "binary"),
    ("ADME", "Pgp_Broccatelli", "binary"),
    ("ADME", "Bioavailability_Ma", "binary"),
    ("ADME", "Lipophilicity_AstraZeneca", "regression"),
    ("ADME", "Solubility_AqSolDB", "regression"),
    ("ADME", "BBB_Martins", "binary"),
    ("ADME", "PPBR_AZ", "regression"),
    ("ADME", "VDss_Lombardo", "regression"),
    ("ADME", "CYP2D6_Veith", "binary"),
    ("ADME", "CYP3A4_Veith", "binary"),
    ("ADME", "CYP1A2_Veith", "binary"),
    ("ADME", "CYP2C9_Veith", "binary"),
    ("ADME", "Half_Life_Obach", "regression"),
    ("ADME", "Clearance_Hepatocyte_AZ", "regression"),
    ("ADME", "Clearance_Microsome_AZ", "regression"),
    ("Tox", "hERG", "binary"),
    ("Tox", "AMES", "binary"),
    ("Tox", "DILI", "binary"),
    ("Tox", "ClinTox", "binary"),
    ("Tox", "LD50_Zhu", "regression"),
]

# Simple SMILES char vocabulary - 0=PAD, 1=UNK, 2..N=chars
SMILES_CHARS = "CcONnSPsHFBrClI[]()=+-#@\\/.123456789%*0:"
VOCAB = {c: i + 2 for i, c in enumerate(SMILES_CHARS)}
PAD, UNK = 0, 1
VOCAB_SIZE = len(VOCAB) + 2


def tokenize(smi: str, max_len: int = 128) -> tuple[list[int], int]:
    ids = [VOCAB.get(c, UNK) for c in smi[:max_len]]
    n = len(ids)
    return ids + [PAD] * (max_len - n), n


class TDCMultiTaskDataset(Dataset):
    def __init__(self, smiles: list[str], labels: np.ndarray, max_len: int = 128):
        assert len(smiles) == labels.shape[0]
        self.smiles = smiles
        self.labels = labels.astype(np.float32)  # NaN = missing
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        ids, length = tokenize(self.smiles[idx], self.max_len)
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:length] = True
        labels = torch.from_numpy(self.labels[idx])
        return ids_t, mask, labels


class MultiTaskADMET(nn.Module):
    def __init__(self, n_tasks: int, d_model: int = 768, n_heads: int = 12, n_layers: int = 8, max_len: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_tasks)])

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        # masked mean pool
        m = mask.unsqueeze(-1).float()
        z = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return torch.cat([h(z) for h in self.heads], dim=-1)


def pull_tdc_combined(decontam: bool, log) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    """Pull all TDC datasets, decontaminate, return (smiles, [N, T] labels, task_names, task_types)."""
    from tdc.single_pred import ADME, Tox

    log("Pulling TDC datasets...")
    smiles_to_idx: dict[str, int] = {}
    rows_per_task: list[dict[str, float]] = []
    task_names: list[str] = []
    task_types: list[str] = []

    for module_name, ds_name, ds_type in TDC_DATASETS:
        try:
            ds_cls = ADME if module_name == "ADME" else Tox
            ds = ds_cls(name=ds_name)
            df = ds.get_data()
            log(f"  {ds_name}: {len(df)} rows ({ds_type})")
        except Exception as e:
            log(f"  {ds_name}: FAILED ({e})")
            continue
        task_names.append(ds_name)
        task_types.append(ds_type)
        col_y = df.columns[-1] if "Y" not in df.columns else "Y"
        col_smi = df.columns[1] if "Drug" not in df.columns else "Drug"
        rows_per_task.append({})
        for _, r in df.iterrows():
            smi = str(r[col_smi]).strip()
            try:
                y = float(r[col_y])
            except Exception:
                continue
            rows_per_task[-1][smi] = y

    # Build the union molecule universe
    universe: dict[str, int] = {}
    for d in rows_per_task:
        for smi in d:
            universe.setdefault(smi, len(universe))

    n = len(universe)
    n_tasks = len(task_names)
    labels = np.full((n, n_tasks), np.nan, dtype=np.float32)
    for t, d in enumerate(rows_per_task):
        for smi, y in d.items():
            labels[universe[smi], t] = y

    smiles_list = [s for s, _ in sorted(universe.items(), key=lambda kv: kv[1])]
    log(f"Combined universe: {n} unique SMILES, {n_tasks} tasks")

    if decontam:
        reg = load_sealed_case_registry()
        rows = [{"smiles": s} for s in smiles_list]
        kept, report = scrub_rows(rows, reg, canonicalize=False)
        kept_smiles = {r["smiles"] for r in kept}
        keep_mask = np.array([s in kept_smiles for s in smiles_list], dtype=bool)
        smiles_list = [s for s, k in zip(smiles_list, keep_mask) if k]
        labels = labels[keep_mask]
        log(f"Decontamination report: {report.to_dict()}")
        log(f"Post-decontam: {len(smiles_list)} SMILES")

    return smiles_list, labels, task_names, task_types


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
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=128, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("rasyn/data/clean/aux_admet"))
    p.add_argument("--no-decontam", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "training_log.jsonl"

    def log(msg: str, **extra):
        if is_main:
            payload = {"t": time.time(), "msg": str(msg), **extra}
            print(f"[{time.strftime('%H:%M:%S')}] [r{rank}] {msg}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Pull data on rank 0, broadcast via files (simplest cross-rank sharing)
    smi_idx_path = args.out / "smiles_index.json"
    labels_path = args.out / "labels.npy"
    if is_main:
        smiles, labels, task_names, task_types = pull_tdc_combined(decontam=not args.no_decontam, log=log)
        smi_idx_path.write_text(json.dumps({
            "smiles": smiles,
            "task_names": task_names,
            "task_types": task_types,
        }))
        np.save(labels_path, labels)
    if dist.is_initialized():
        dist.barrier()
    if not is_main:
        meta = json.loads(smi_idx_path.read_text())
        smiles, task_names, task_types = meta["smiles"], meta["task_names"], meta["task_types"]
        labels = np.load(labels_path)

    n_tasks = len(task_names)
    log(f"World size {world_size}, n_tasks {n_tasks}, n_smiles {len(smiles)}")

    # 90/10 split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(smiles))
    val_n = max(1024, len(smiles) // 10)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    tr_ds = TDCMultiTaskDataset([smiles[i] for i in tr_idx], labels[tr_idx], max_len=args.max_len)
    vl_ds = TDCMultiTaskDataset([smiles[i] for i in val_idx], labels[val_idx], max_len=args.max_len)

    sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    tr_dl = DataLoader(
        tr_ds, batch_size=args.bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=4, pin_memory=True, drop_last=True,
    )
    vl_dl = DataLoader(vl_ds, batch_size=args.bs * 2, num_workers=2, pin_memory=True)

    device = torch.device(f"cuda:{local_rank}")
    model = MultiTaskADMET(
        n_tasks=n_tasks, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, max_len=args.max_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scaler = torch.amp.GradScaler("cuda", enabled=False)  # we use bf16 directly
    is_classification = torch.tensor([t == "binary" for t in task_types], device=device)

    def step_fn(batch):
        ids, mask, y = [b.to(device, non_blocking=True) for b in batch]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
        # Per-task loss with NaN masking
        valid = ~torch.isnan(y)
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True), 0
        # Standardise regression targets per-task using running stats? Skip for v1; use raw.
        # MSE for regression, BCE for binary
        loss = 0.0
        for t in range(n_tasks):
            v = valid[:, t]
            if not v.any():
                continue
            if is_classification[t]:
                tloss = F.binary_cross_entropy_with_logits(logits[v, t], y[v, t].clamp(0, 1))
            else:
                tloss = F.mse_loss(logits[v, t], y[v, t])
            loss = loss + tloss
        loss = loss / n_tasks
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

    log(f"Starting training: {args.steps} steps, bs={args.bs} per GPU, effective bs={args.bs * world_size}")
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
            loss, n_valid = step_fn(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                elapsed = time.time() - t0
                log(
                    f"step {step}/{args.steps} loss={loss.item():.4f} lr={optim.param_groups[0]['lr']:.2e} "
                    f"throughput={(step * args.bs * world_size) / elapsed:.0f} samp/s",
                    step=step, loss=float(loss.item()),
                )
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
                    "vocab": VOCAB,
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
            "vocab": VOCAB,
        }, args.out / "checkpoint.pt")
        log("FINAL", val=metrics, total_seconds=time.time() - t0)

    cleanup_distributed()


if __name__ == "__main__":
    main()
