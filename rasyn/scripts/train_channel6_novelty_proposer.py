"""Channel 6: pure learned novelty proposer (B-6 from PLAN.md).

Per `proposer_system_test_cases.md` channel 6: unconstrained generative
model trained on ChEMBL alone (no conditioning labels needed).

Architecture: causal-masked transformer decoder over SMILES tokens.
Trained autoregressively on the 2.47M decontaminated ChEMBL canonical SMILES.

Output: a generative model that produces novel valid SMILES at inference.

This is the ONLY ML proposer trainable without rescue_pairs.parquet.
Channels 4 (learned inverse-delta) and 5 (forward-reward) are gated on
Phase A-4 completion.

Run (8x A100 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/train_channel6_novelty_proposer.py \\
        --data rasyn/data/clean/molecules_canonical.parquet \\
        --steps 8000 --bs 96 --d-model 768 --n-layers 8

Outputs (rasyn/data/clean/channel6_novelty/):
    checkpoint.pt
    training_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Match Stage-1 vocab so encoder weights can be reused if desired.
SMILES_CHARS = "CcONnSPsHFBrClI[]()=+-#@\\/.123456789%*0:"
PAD, UNK, BOS, EOS = 0, 1, 2, 3
VOCAB = {c: i + 4 for i, c in enumerate(SMILES_CHARS)}
VOCAB_SIZE = len(VOCAB) + 4


class SMILESLMDataset(Dataset):
    """Autoregressive LM dataset. Each item is (input_ids, target_ids)
    with input shifted by 1.

    Sequences: [BOS] tok1 tok2 ... tokN [EOS] [PAD] ...
    inputs:    [BOS] tok1 tok2 ... tokN [EOS]
    targets:   tok1 tok2 ...      tokN  [EOS] [IGNORE]   (shift left)
    """

    def __init__(self, smiles_list: list[str], max_len: int = 130):
        self.smiles = smiles_list
        self.max_len = max_len  # incl BOS+EOS

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        toks = [VOCAB.get(c, UNK) for c in smi[: self.max_len - 2]]
        seq = [BOS] + toks + [EOS]
        n = len(seq)
        seq = seq + [PAD] * (self.max_len - n)
        ids = np.asarray(seq, dtype=np.int64)
        # input is everything except the last token; target is shifted-left.
        # We use the same buffer of length max_len for simplicity, and ignore
        # PAD positions in loss.
        attn_mask = np.zeros(self.max_len, dtype=bool)
        attn_mask[:n] = True
        return torch.from_numpy(ids), torch.from_numpy(attn_mask)


class CausalSMILESLM(nn.Module):
    """Causal-masked transformer for autoregressive SMILES generation."""

    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12, n_layers: int = 8, max_len: int = 130):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        # Use TransformerEncoder with a causal mask = behaves as decoder-only LM.
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie embeddings + LM head (saves params)
        self.lm_head.weight = self.tok_emb.weight
        self.max_len = max_len

    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        # Causal mask (upper triangular = -inf so token t can't attend to t+1...)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        # Padding mask: True where padding
        key_padding_mask = ~attn_mask  # padding positions = True
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return self.lm_head(x)


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
    p.add_argument("--data", type=Path, required=True, help="Parquet with 'canonical_smiles' column.")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--bs", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=130)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("rasyn/data/clean/channel6_novelty"))
    p.add_argument("--seed", type=int, default=42)
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

    if is_main:
        log(f"Loading parquet: {args.data}")
    df = pd.read_parquet(args.data)
    smiles_list = df["canonical_smiles"].astype(str).tolist()
    if is_main:
        log(f"Loaded {len(smiles_list):,} SMILES (max_len={args.max_len})")

    ds = SMILESLMDataset(smiles_list, max_len=args.max_len)
    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    dl = DataLoader(
        ds, batch_size=args.bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
    )

    device = torch.device(f"cuda:{local_rank}")
    model = CausalSMILESLM(
        VOCAB_SIZE, args.d_model, args.n_heads, args.n_layers, args.max_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        log(f"Model params: {n_params:,} | World size: {world_size} | Effective bs: {args.bs * world_size}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    if is_main:
        log(f"Starting Channel-6 generative pretrain: {args.steps} steps")

    t0 = time.time()
    step = 0
    epoch = 0
    model.train()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            for pg in optim.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            optim.zero_grad(set_to_none=True)
            ids, attn_mask = [b.to(device, non_blocking=True) for b in batch]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, attn_mask)
            logits = logits.float()
            # Shift targets left by 1; ignore PAD in loss.
            inputs = ids[:, :-1]
            targets = ids[:, 1:].contiguous()
            mask_targets = attn_mask[:, 1:]
            logits = logits[:, :-1].contiguous()
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=PAD,
                reduction="none",
            )
            loss = (loss * mask_targets.reshape(-1).float()).sum() / mask_targets.sum().clamp(min=1.0).float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                elapsed = time.time() - t0
                throughput = (step * args.bs * world_size) / max(1.0, elapsed)
                log(
                    f"step {step}/{args.steps} loss={loss.item():.4f} lr={optim.param_groups[0]['lr']:.2e} thru={throughput:.0f} samp/s",
                    step=step, loss=float(loss.item()),
                )
            if step % args.ckpt_every == 0 and is_main:
                torch.save(
                    {
                        "step": step,
                        "model": (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict(),
                        "args": vars(args),
                        "vocab": VOCAB,
                        "vocab_size": VOCAB_SIZE,
                        "channel": "learned_novelty",
                    },
                    args.out / "checkpoint.pt",
                )
                log(f"Saved checkpoint at step {step}")
            if step >= args.steps:
                break
        epoch += 1

    if is_main:
        torch.save(
            {
                "step": step,
                "model": (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict(),
                "args": vars(args),
                "vocab": VOCAB,
                "vocab_size": VOCAB_SIZE,
                "channel": "learned_novelty",
            },
            args.out / "checkpoint.pt",
        )
        log("FINAL", total_seconds=time.time() - t0, total_params=n_params)

    cleanup_distributed()


if __name__ == "__main__":
    main()
