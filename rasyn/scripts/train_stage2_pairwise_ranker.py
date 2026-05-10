"""Stage-2 pairwise rescue ranker training (FULL CAPACITY per L18).

Architecture (per L18 + spec §3-4 of rasyn_admet_rescue_architecture_context.md
+ rasyn_admet_conditioning_architecture_benchmark_spec.md):

  Stage-1 200M backbone (d=1024, n_heads=16, n_layers=16) shared as encoder
  -> two-tower SMILES embedding (parent + candidate)
  -> cross-attention: parent attends to candidate
  -> masked-mean pool both sides
  -> evidence-feature projection (descriptors, deltas, retention/improvement
     one-hots, rationale counts, risk flags) projected to d_model
  -> [parent_pool, candidate_pool, evidence_h] -> MLP pair representation
  -> 5 multi-task heads:
       rescue_label_logits   (7-class: per spec rasyn_admet_conditioning §4)
       failure_mode_logits   (6-class)
       retention_logits      (5-class)
       improvement_logits    (6-class)
       rescue_score          (regression in [0,1])

Training data: rescue_pair_candidates.parquet, filtered to silver tier.
~562K pairs, 8xA100 DDP, full hyperparameters.

Multi-seed: launch with --seed 42 on Pod A, --seed 43 on Pod B for variance
estimate (per L18 multi-seed strategy).

Run (8x A100 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/train_stage2_pairwise_ranker.py \\
        --pretrain rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --pairs    rasyn/data/clean/rescue_pair_candidates.parquet \\
        --steps 6000 --bs 32 --lr 1e-4 --seed 42 \\
        --out rasyn/data/clean/stage2_ranker_seed42

Outputs:
    out/checkpoint.pt          (DDP-stripped state_dict + args + label maps)
    out/training_log.jsonl
    out/per_class_metrics.json (val: rescue_label_acc, retention_acc, ...)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Reuse Stage-1 backbone vocab + encoder.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE, PAD


# ===== Label maps =====
RESCUE_LABELS = [
    "strong_success", "weak_success",
    "failed_activity_loss", "failed_no_liability_improvement",
    "failed_wrong_liability", "failed_new_liability",
    "uncertain",
]
FAILURE_MODES = [
    "activity_loss", "liability_unchanged", "wrong_liability",
    "new_liability", "selectivity_collapse", "synthetic_infeasibility",
]
RETENTION_BUCKETS = ["strong", "acceptable", "weak", "failed", "unknown"]
IMPROVEMENT_CATEGORIES = ["large", "moderate", "minor", "none", "worse", "unknown"]

RESCUE_LABEL2ID = {v: i for i, v in enumerate(RESCUE_LABELS)}
FAILURE_MODE2ID = {v: i for i, v in enumerate(FAILURE_MODES)}
RETENTION2ID = {v: i for i, v in enumerate(RETENTION_BUCKETS)}
IMPROVEMENT2ID = {v: i for i, v in enumerate(IMPROVEMENT_CATEGORIES)}


def derive_rescue_label(retention: str, improvement: str) -> str:
    """Map (retention_bucket, improvement_category) -> 7-class rescue_label.

    Maps spec rasyn_admet_conditioning §4 logic onto our two label axes.
    Cases requiring extra context (wrong_liability, new_liability) default
    to 'uncertain' since we don't have the cross-liability signal here.
    """
    if retention is None or improvement is None:
        return "uncertain"
    if retention == "unknown" or improvement == "unknown":
        return "uncertain"
    if retention in ("strong", "acceptable") and improvement in ("large", "moderate"):
        return "strong_success"
    if retention in ("strong", "acceptable", "weak") and improvement == "minor":
        return "weak_success"
    if improvement in ("large", "moderate", "minor") and retention == "failed":
        return "failed_activity_loss"
    if retention in ("strong", "acceptable") and improvement in ("none", "worse"):
        return "failed_no_liability_improvement"
    return "uncertain"


# ===== Evidence features =====
# Engineered features built directly from the parquet row. 32-dim vector.
EVIDENCE_DIM = 32


def build_evidence_vector(row: dict) -> np.ndarray:
    """Build 32-dim evidence vector from a rescue_pair row."""
    e = np.zeros(EVIDENCE_DIM, dtype=np.float32)
    # 0-3: structural similarity
    e[0] = float(row.get("ecfp_tanimoto") or 0.0)
    e[1] = 1.0 if row.get("murcko_match") else 0.0
    e[2] = float(row.get("heavy_atom_diff") or 0.0)
    # 3-7: activity context (potency)
    e[3] = float(row.get("parent_activity_pchembl") or 0.0)
    e[4] = float(row.get("candidate_activity_pchembl") or 0.0)
    if e[3] != 0 and e[4] != 0:
        e[5] = e[4] - e[3]  # delta pchembl
    # 6-10: liability values (raw, log-scale)
    pl = row.get("parent_liability_value")
    cl = row.get("candidate_liability_value")
    e[6] = float(np.log1p(abs(pl))) if pl is not None and not pd.isna(pl) else 0.0
    e[7] = float(np.log1p(abs(cl))) if cl is not None and not pd.isna(cl) else 0.0
    if pl is not None and cl is not None and not pd.isna(pl) and not pd.isna(cl):
        e[8] = float(np.log1p(abs(cl)) - np.log1p(abs(pl)))
    # 9-13: retention bucket one-hot
    ret = row.get("activity_retention_bucket") or "unknown"
    if ret in RETENTION2ID:
        e[9 + RETENTION2ID[ret]] = 1.0
    # 14-19: improvement category one-hot
    imp = row.get("liability_improvement_category") or "unknown"
    if imp in IMPROVEMENT2ID:
        e[14 + IMPROVEMENT2ID[imp]] = 1.0
    # 20-26: liability_type one-hot (4 v1 families + permeability + unknown + active_metabolite)
    LIABILITY_TYPES = ["hERG", "solubility", "metabolic_stability", "oral_exposure", "permeability", "unknown", "other"]
    lt = row.get("liability_type") or "unknown"
    if lt in LIABILITY_TYPES:
        e[20 + LIABILITY_TYPES.index(lt)] = 1.0
    else:
        e[20 + 6] = 1.0  # other
    # 27: hard_negative_type indicator
    hn = row.get("hard_negative_type")
    if hn == "improved_but_activity_lost":
        e[27] = 1.0
    elif hn == "retained_but_liability_unfixed":
        e[28] = 1.0
    # 29-31 reserved for future
    return e


def tokenize(smi: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    UNK = 1
    ids = [VOCAB.get(c, UNK) for c in smi[:max_len]]
    n = len(ids)
    ids = ids + [PAD] * (max_len - n)
    attn = np.zeros(max_len, dtype=bool)
    attn[:n] = True
    return np.asarray(ids, dtype=np.int64), attn


class PairDataset(Dataset):
    def __init__(self, rows: list[dict], max_len: int = 128):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p_ids, p_mask = tokenize(r["parent_smiles"] or "C", self.max_len)
        c_ids, c_mask = tokenize(r["candidate_smiles"] or "C", self.max_len)
        evidence = build_evidence_vector(r)
        rescue_id = RESCUE_LABEL2ID[derive_rescue_label(
            r.get("activity_retention_bucket"), r.get("liability_improvement_category")
        )]
        retention_id = RETENTION2ID.get(r.get("activity_retention_bucket") or "unknown", RETENTION2ID["unknown"])
        improvement_id = IMPROVEMENT2ID.get(r.get("liability_improvement_category") or "unknown", IMPROVEMENT2ID["unknown"])
        # Failure mode: derive simple rule (ignore for unknown)
        failure_id = -100  # ignore_index
        if r.get("hard_negative_type") == "improved_but_activity_lost":
            failure_id = FAILURE_MODE2ID["activity_loss"]
        elif r.get("hard_negative_type") == "retained_but_liability_unfixed":
            failure_id = FAILURE_MODE2ID["liability_unchanged"]
        # rescue_score: continuous in [0,1] derived from rescue_label
        rs_id_to_score = {0: 1.0, 1: 0.7, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.5}
        rescue_score = rs_id_to_score[rescue_id]
        return (
            torch.from_numpy(p_ids), torch.from_numpy(p_mask),
            torch.from_numpy(c_ids), torch.from_numpy(c_mask),
            torch.from_numpy(evidence),
            torch.tensor(rescue_id, dtype=torch.long),
            torch.tensor(failure_id, dtype=torch.long),
            torch.tensor(retention_id, dtype=torch.long),
            torch.tensor(improvement_id, dtype=torch.long),
            torch.tensor(rescue_score, dtype=torch.float32),
        )


class PairwiseRescueRanker(nn.Module):
    """Two-tower transformer + cross-attention + multi-task heads."""

    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 16,
                 n_layers: int = 16, max_len: int = 128, evidence_dim: int = EVIDENCE_DIM):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.evidence_proj = nn.Sequential(
            nn.Linear(evidence_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Dropout(0.1),
        )

        self.head_rescue_label = nn.Linear(d_model, len(RESCUE_LABELS))
        self.head_failure_mode = nn.Linear(d_model, len(FAILURE_MODES))
        self.head_retention = nn.Linear(d_model, len(RETENTION_BUCKETS))
        self.head_improvement = nn.Linear(d_model, len(IMPROVEMENT_CATEGORIES))
        self.head_rescue_score = nn.Linear(d_model, 1)

        self.max_len = max_len

    def encode(self, ids, mask):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        return x

    def forward(self, p_ids, p_mask, c_ids, c_mask, evidence):
        p_seq = self.encode(p_ids, p_mask)
        c_seq = self.encode(c_ids, c_mask)
        # Parent attends to candidate
        p_attended, _ = self.cross_attn(
            p_seq, c_seq, c_seq, key_padding_mask=~c_mask, need_weights=False
        )
        # Masked mean pool
        def _mp(x, mask):
            m = mask.unsqueeze(-1).float()
            return (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        p_pool = _mp(p_attended, p_mask)
        c_pool = _mp(c_seq, c_mask)
        e_h = self.evidence_proj(evidence)
        pair = self.pair_proj(torch.cat([p_pool, c_pool, e_h], dim=-1))
        return {
            "rescue_label_logits": self.head_rescue_label(pair),
            "failure_mode_logits": self.head_failure_mode(pair),
            "retention_logits": self.head_retention(pair),
            "improvement_logits": self.head_improvement(pair),
            "rescue_score": torch.sigmoid(self.head_rescue_score(pair)).squeeze(-1),
        }


def load_pretrain_into_ranker(model: PairwiseRescueRanker, ckpt_path: Path, log) -> int:
    """Copy Stage-1 backbone weights into the ranker's encoder + embeddings."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrain_sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model_sd = model.state_dict()
    n_loaded = 0
    n_skipped = 0
    for k, v in pretrain_sd.items():
        # SMILES LM keys: tok_emb.weight, pos_emb.weight, encoder.layers.X.*, norm.*, lm_head.*
        if k.startswith("lm_head"):
            continue  # not present in ranker
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            n_loaded += 1
        else:
            n_skipped += 1
    model.load_state_dict(model_sd)
    if log:
        log(f"Loaded {n_loaded} backbone tensors from {ckpt_path} ({n_skipped} skipped/new)")
    return n_loaded


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=2))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--tier", default="silver")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=32, help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze Stage-1 backbone (faster, lower quality).")
    p.add_argument("--loss-rescue-label", type=float, default=1.0)
    p.add_argument("--loss-failure-mode", type=float, default=0.3)
    p.add_argument("--loss-retention", type=float, default=0.4)
    p.add_argument("--loss-improvement", type=float, default=0.4)
    p.add_argument("--loss-rescue-score", type=float, default=0.5)
    args = p.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "training_log.jsonl"

    def log(msg, **extra):
        if is_main:
            print(f"[{time.strftime('%H:%M:%S')}] [r{rank}] {msg}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "msg": str(msg), **extra}) + "\n")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}")

    if is_main:
        log(f"Loading pairs from {args.pairs}")
    df = pd.read_parquet(args.pairs)
    if "quality_tier" in df.columns and args.tier:
        before = len(df)
        df = df[df["quality_tier"] == args.tier]
        log(f"Filtered tier={args.tier}: {before:,} -> {len(df):,} pairs")

    df = df[df["parent_smiles"].notna() & df["candidate_smiles"].notna()]
    log(f"After SMILES non-null filter: {len(df):,}")
    if len(df) < 1000:
        raise SystemExit(f"Too few rows ({len(df)}) for training; aborting.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    val_n = max(1024, int(len(df) * args.val_frac))
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    tr_rows = df.iloc[tr_idx].to_dict(orient="records")
    val_rows = df.iloc[val_idx].to_dict(orient="records")

    tr_ds = PairDataset(tr_rows, max_len=args.max_len)
    val_ds = PairDataset(val_rows, max_len=args.max_len)
    sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, shuffle=(sampler is None),
                       num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs * 2, num_workers=2, pin_memory=True)

    if is_main:
        log(f"World={world_size} | per-GPU bs={args.bs} | eff bs={args.bs * world_size} | "
            f"train={len(tr_ds):,} val={len(val_ds):,}")

    # Auto-detect Stage-1 backbone arch from ckpt args
    ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    d_model = cargs.get("d_model", 1024)
    n_heads = cargs.get("n_heads", 16)
    n_layers = cargs.get("n_layers", 16)
    max_len = max(cargs.get("max_len", args.max_len), args.max_len)

    if is_main:
        log(f"Backbone arch: d={d_model} h={n_heads} L={n_layers} max_len={max_len}")

    model = PairwiseRescueRanker(
        VOCAB_SIZE, d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_len=max_len,
    ).to(device)

    if is_main:
        load_pretrain_into_ranker(model, args.pretrain, log)
    if dist.is_initialized():
        dist.barrier()
    if not is_main:
        load_pretrain_into_ranker(model, args.pretrain, None)

    if args.freeze_encoder:
        for n, p in model.named_parameters():
            if any(n.startswith(prefix) for prefix in ("tok_emb", "pos_emb", "encoder", "norm.")):
                p.requires_grad = False
        if is_main:
            log("Froze backbone tensors. Training cross_attn + heads only.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        log(f"Trainable params: {n_params:,}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95),
    )

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        n = 0
        rescue_correct = 0
        retention_correct = 0
        improvement_correct = 0
        for batch in val_dl:
            batch = [b.to(device, non_blocking=True) for b in batch]
            p_ids, p_mask, c_ids, c_mask, evidence, rescue_id, fail_id, ret_id, imp_id, score = batch
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(p_ids, p_mask, c_ids, c_mask, evidence)
            rescue_correct += (out["rescue_label_logits"].float().argmax(-1) == rescue_id).sum().item()
            retention_correct += (out["retention_logits"].float().argmax(-1) == ret_id).sum().item()
            improvement_correct += (out["improvement_logits"].float().argmax(-1) == imp_id).sum().item()
            n += rescue_id.size(0)
        model.train()
        return {
            "rescue_label_acc": rescue_correct / max(1, n),
            "retention_acc": retention_correct / max(1, n),
            "improvement_acc": improvement_correct / max(1, n),
            "n_val": n,
        }

    if is_main:
        log(f"Starting Stage-2 ranker training: {args.steps} steps, eff bs {args.bs * world_size}")

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
            batch = [b.to(device, non_blocking=True) for b in batch]
            p_ids, p_mask, c_ids, c_mask, evidence, rescue_id, fail_id, ret_id, imp_id, score = batch
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(p_ids, p_mask, c_ids, c_mask, evidence)
            rl = F.cross_entropy(out["rescue_label_logits"].float(), rescue_id)
            fl = F.cross_entropy(out["failure_mode_logits"].float(), fail_id, ignore_index=-100)
            rt = F.cross_entropy(out["retention_logits"].float(), ret_id)
            ip = F.cross_entropy(out["improvement_logits"].float(), imp_id)
            ss = F.mse_loss(out["rescue_score"].float(), score)
            loss = (
                args.loss_rescue_label * rl
                + args.loss_failure_mode * fl
                + args.loss_retention * rt
                + args.loss_improvement * ip
                + args.loss_rescue_score * ss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                log(
                    f"step {step}/{args.steps} loss={loss.item():.4f} "
                    f"rl={rl.item():.3f} ret={rt.item():.3f} imp={ip.item():.3f} ss={ss.item():.3f}",
                    step=step, loss=float(loss.item()), rl=float(rl.item()), ret=float(rt.item()),
                    imp=float(ip.item()), ss=float(ss.item()),
                )
            if step % args.val_every == 0 and is_main:
                metrics = evaluate()
                log(f"step {step} val: {metrics}", val=metrics)
                (args.out / "per_class_metrics.json").write_text(json.dumps(metrics, indent=2))
            if step % args.ckpt_every == 0 and is_main:
                _save(model, args, step)
                log(f"Saved checkpoint at step {step}")
            if step >= args.steps:
                break
        epoch += 1

    if is_main:
        metrics = evaluate()
        (args.out / "per_class_metrics.json").write_text(json.dumps(metrics, indent=2))
        _save(model, args, step)
        log("FINAL", val=metrics, total_seconds=time.time() - t0)

    cleanup_distributed()
    return 0


def _save(model, args, step):
    state_dict = (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict()
    torch.save({
        "step": step,
        "model": state_dict,
        "args": vars(args),
        "rescue_labels": RESCUE_LABELS,
        "failure_modes": FAILURE_MODES,
        "retention_buckets": RETENTION_BUCKETS,
        "improvement_categories": IMPROVEMENT_CATEGORIES,
        "evidence_dim": EVIDENCE_DIM,
        "vocab_size": VOCAB_SIZE,
    }, args.out / "checkpoint.pt")


if __name__ == "__main__":
    sys.exit(main())
