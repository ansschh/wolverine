"""Autoregressive SMILES language model — generative replacement for the MLM.

The MLM trained by `h200_smiles_lm_pretrain.py` is a BERT-style masked LM
and cannot autoregressively sample new molecules. This trains a causal LM
with the same vocabulary so we can:
  1. Sample SMILES via top-p / temperature
  2. RL-fine-tune toward a ranker reward (Stokes-style)

Architecture parallels the MLM: same VOCAB, same tokenization, same
(1024d / 16h / 16L) transformer. Only differences:
  - causal upper-triangular attention mask
  - next-token cross-entropy loss (no masking)
  - generation hooks (`sample(...)`)

Run on 5x A100:
    torchrun --nproc_per_node=5 --standalone scripts/train_smiles_ar_lm.py \\
        --data rasyn/data/clean/chembl_all_smiles.parquet \\
        --steps 12000 --bs 32 --d-model 1024 --n-heads 16 --n-layers 16 \\
        --warmup 500 --ckpt-every 2000 \\
        --out rasyn/data/clean/smiles_ar_lm_200m

Outputs:
    out/checkpoint.pt
    out/training_log.jsonl
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

# Same vocab as MLM — kept identical so checkpoints can be loaded/swapped.
SMILES_CHARS = (
    "abcdefghiklmnoprstuyACBEFGHIKLMNOPSRTUVXY"  # atom-and-aromatic chars
    "0123456789"
    "()[]{}-=#+@/\\.%"
    "*"  # wildcard
)
PAD = 0
BOS = 1
EOS = 2
UNK = 3
VOCAB = {c: i + 4 for i, c in enumerate(SMILES_CHARS)}
VOCAB_SIZE = len(VOCAB) + 4


def tokenize_with_bos_eos(smi: str, max_len: int) -> tuple[list[int], int]:
    """[BOS] + chars + [EOS], truncated/padded to max_len. Returns (ids, n_real_tokens)."""
    toks = [BOS] + [VOCAB.get(c, UNK) for c in smi[: max_len - 2]] + [EOS]
    n = len(toks)
    toks = toks + [PAD] * (max_len - n)
    return toks, n


class ARSMILESDataset(Dataset):
    def __init__(self, smiles_list: list[str], max_len: int = 128):
        self.smiles = smiles_list
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        ids, n = tokenize_with_bos_eos(self.smiles[idx], self.max_len)
        ids = torch.tensor(ids, dtype=torch.long)
        # Labels: same as ids but shifted by 1 (predict next token).
        # At positions where target is PAD we use ignore_index = -100.
        labels = ids.clone()
        labels[labels == PAD] = -100
        # For non-PAD positions, the LABEL at position i is the TOKEN at position i+1.
        # We construct that by rolling and masking. But easiest is: model predicts logit_i for token_i+1.
        # We'll let the training loop do the shift.
        return ids


class ARSMILESLM(nn.Module):
    """Decoder-only transformer, causal mask, tied input/output embeddings."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 1024,
                 n_heads: int = 16, n_layers: int = 16, max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        # Build causal mask (T,T) so position i attends only to 0..i
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        # Pad-key mask
        pad_mask = ids == PAD  # (B, T)
        x = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        return self.lm_head(x)  # (B, T, V)

    @torch.no_grad()
    def sample(self, n: int, max_len: int = 128, temperature: float = 1.0,
               top_k: int | None = None, top_p: float | None = None,
               device: torch.device = torch.device("cpu")) -> list[str]:
        """Sample n SMILES by autoregressive decoding."""
        self.eval()
        seqs = torch.full((n, max_len), PAD, dtype=torch.long, device=device)
        seqs[:, 0] = BOS
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        for t in range(1, max_len):
            ids_in = seqs[:, :t]
            logits = self(ids_in)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                vals, _ = logits.topk(top_k, dim=-1)
                cutoff = vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < cutoff,
                                      torch.full_like(logits, float("-inf")), logits)
            if top_p is not None:
                sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                mask = cum > top_p
                mask[:, 0] = False  # always keep top-1
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            seqs[:, t] = nxt
            finished = finished | (nxt == EOS)
            if finished.all():
                break
        # Detokenize
        inv = {v: k for k, v in VOCAB.items()}
        out: list[str] = []
        for i in range(n):
            chars = []
            for t in range(1, max_len):  # skip BOS
                tok = int(seqs[i, t].item())
                if tok == EOS or tok == PAD:
                    break
                if tok in inv:
                    chars.append(inv[tok])
            out.append("".join(chars))
        return out


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=2))
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=16)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rank, world_size, local_rank = setup_ddp()
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

    log(f"Loading {args.data}")
    df = pd.read_parquet(args.data)
    smiles_list = df["canonical_smiles"].astype(str).tolist()
    log(f"Loaded {len(smiles_list):,} SMILES")

    ds = ARSMILESDataset(smiles_list, max_len=args.max_len)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, shuffle=(sampler is None),
                    num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)

    model = ARSMILESLM(d_model=args.d_model, n_heads=args.n_heads,
                        n_layers=args.n_layers, max_len=args.max_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,} | World size: {world_size} | Effective bs: {args.bs * world_size}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    log(f"Starting AR-LM pretrain: {args.steps} steps")
    t0 = time.time()
    step = epoch = 0
    model.train()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for ids in dl:
            ids = ids.to(device, non_blocking=True)
            for pg in optim.param_groups:
                pg["lr"] = args.lr * lr_at(step)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)  # (B, T, V)
            # Shift: position i predicts token at position i+1
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = ids[:, 1:].contiguous()
            shift_labels = shift_labels.masked_fill(shift_labels == PAD, -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, VOCAB_SIZE),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                el = time.time() - t0
                log(f"step {step}/{args.steps} loss={loss.item():.4f} "
                    f"lr={optim.param_groups[0]['lr']:.2e} thru={(step*args.bs*world_size)/el:.0f} samp/s",
                    step=step, loss=float(loss.item()))
            if step % args.ckpt_every == 0 and is_main:
                _save(model, args, step)
                log(f"Saved ckpt at step {step}")
            if step >= args.steps:
                break
        epoch += 1
    if is_main:
        _save(model, args, step)
        log("FINAL", total_seconds=time.time() - t0)
    cleanup_ddp()
    return 0


def _save(model, args, step):
    sd = (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict()
    torch.save({
        "step": step, "model": sd, "args": vars(args),
        "vocab": VOCAB, "vocab_size": VOCAB_SIZE,
        "BOS": BOS, "EOS": EOS, "PAD": PAD, "UNK": UNK,
        "framework": "ar_smiles_lm_decoder_only",
    }, args.out / "checkpoint.pt")


if __name__ == "__main__":
    sys.exit(main())
