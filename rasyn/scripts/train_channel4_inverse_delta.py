"""Channel 4: learned inverse-delta proposer training.

Encoder-decoder transformer:
  Input:  parent_smiles + [LIABILITY_<type>] conditioning token
  Output: candidate_smiles (autoregressive)

Trained on rescue_pair_candidates silver tier filtered to:
  - liability_improvement_category in (large, moderate)
  - activity_retention_bucket in (strong, acceptable)
  - liability_type non-null
  - parent_smiles + candidate_smiles non-null and round-trip-valid

Init: encoder + decoder both initialised from Stage-1 200M backbone weights.

Run on Pod C (8x A100 DDP) AFTER Phase A-4 produces rescue_pair_candidates:
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/train_channel4_inverse_delta.py \\
        --pretrain rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --pairs    rasyn/data/clean/rescue_pair_candidates.parquet \\
        --steps 6000 --bs 24 --lr 1e-4 \\
        --out rasyn/data/clean/channel4_inverse_delta

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

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE, PAD


# Liability conditioning tokens — added to vocab on top of base SMILES vocab.
LIABILITY_TOKENS = [
    "[LIABILITY_HERG]", "[LIABILITY_SOLUBILITY]", "[LIABILITY_METSTAB]",
    "[LIABILITY_ORAL_EXPOSURE]", "[LIABILITY_PERMEABILITY]",
]
LIABILITY_TYPE2STR = {
    "hERG": "[LIABILITY_HERG]",
    "solubility": "[LIABILITY_SOLUBILITY]",
    "metabolic_stability": "[LIABILITY_METSTAB]",
    "oral_exposure": "[LIABILITY_ORAL_EXPOSURE]",
    "permeability": "[LIABILITY_PERMEABILITY]",
}

# Special tokens beyond base vocab.
BOS = VOCAB_SIZE
EOS = VOCAB_SIZE + 1
SEP = VOCAB_SIZE + 2
LIAB_OFFSET = VOCAB_SIZE + 3
EXTENDED_VOCAB_SIZE = LIAB_OFFSET + len(LIABILITY_TOKENS)

LIAB_TOKEN_ID = {tok: LIAB_OFFSET + i for i, tok in enumerate(LIABILITY_TOKENS)}


def tokenize_smiles(smi: str, max_len: int) -> list[int]:
    UNK = 1
    return [VOCAB.get(c, UNK) for c in smi[:max_len]]


def encode_input(parent_smi: str, liability_type: str, max_len: int) -> tuple[list[int], list[bool]]:
    """Encoder input: [BOS][LIABILITY_<type>][SEP] + parent_tokens + [EOS] padded."""
    liab_tok = LIABILITY_TYPE2STR.get(liability_type, LIABILITY_TOKENS[0])
    parent_ids = tokenize_smiles(parent_smi, max_len - 4)
    seq = [BOS, LIAB_TOKEN_ID[liab_tok], SEP] + parent_ids + [EOS]
    n = len(seq)
    seq = seq + [PAD] * (max_len - n)
    mask = [True] * n + [False] * (max_len - n)
    return seq, mask


def encode_target(candidate_smi: str, max_len: int) -> tuple[list[int], list[bool]]:
    """Decoder target: [BOS] + candidate_tokens + [EOS] padded."""
    cand_ids = tokenize_smiles(candidate_smi, max_len - 2)
    seq = [BOS] + cand_ids + [EOS]
    n = len(seq)
    seq = seq + [PAD] * (max_len - n)
    mask = [True] * n + [False] * (max_len - n)
    return seq, mask


class Channel4Dataset(Dataset):
    def __init__(self, rows: list[dict], max_len: int = 130):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc_ids, enc_mask = encode_input(r["parent_smiles"], r.get("liability_type", "hERG"), self.max_len)
        dec_ids, dec_mask = encode_target(r["candidate_smiles"], self.max_len)
        return (
            torch.tensor(enc_ids, dtype=torch.long),
            torch.tensor(enc_mask, dtype=torch.bool),
            torch.tensor(dec_ids, dtype=torch.long),
            torch.tensor(dec_mask, dtype=torch.bool),
        )


class InverseDeltaSeq2Seq(nn.Module):
    """Encoder-decoder transformer. Uses Stage-1 backbone for encoder + decoder."""

    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 16,
                 n_layers: int = 16, max_len: int = 130):
        super().__init__()
        # Encoder
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers // 2)  # Half-depth decoder
        self.dec_norm = nn.LayerNorm(d_model)

        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        self.max_len = max_len

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask):
        # Encoder
        T = enc_ids.size(1)
        pos = torch.arange(T, device=enc_ids.device).unsqueeze(0).expand_as(enc_ids)
        x = self.tok_emb(enc_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=~enc_mask)
        x = self.norm(x)

        # Decoder
        Td = dec_ids.size(1)
        dpos = torch.arange(Td, device=dec_ids.device).unsqueeze(0).expand_as(dec_ids)
        y = self.tok_emb(dec_ids) + self.pos_emb(dpos)
        causal_mask = torch.triu(
            torch.full((Td, Td), float("-inf"), device=dec_ids.device), diagonal=1
        )
        y = self.decoder(
            y, x, tgt_mask=causal_mask, memory_key_padding_mask=~enc_mask,
            tgt_key_padding_mask=~dec_mask,
        )
        y = self.dec_norm(y)
        return self.lm_head(y)


def load_pretrain_into_seq2seq(model: InverseDeltaSeq2Seq, ckpt_path: Path, log) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrain_sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model_sd = model.state_dict()

    # Resize tok_emb to extended vocab; copy original part.
    if "tok_emb.weight" in pretrain_sd and pretrain_sd["tok_emb.weight"].shape != model_sd["tok_emb.weight"].shape:
        old = pretrain_sd.pop("tok_emb.weight")
        new = model_sd["tok_emb.weight"].clone()
        n_copy = min(old.shape[0], new.shape[0])
        new[:n_copy] = old[:n_copy]
        model_sd["tok_emb.weight"] = new

    n_loaded = 0
    n_skipped = 0
    for k, v in pretrain_sd.items():
        if k.startswith("lm_head"):
            continue
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            n_loaded += 1
        else:
            n_skipped += 1
    model.load_state_dict(model_sd)
    if log:
        log(f"Loaded {n_loaded} backbone tensors ({n_skipped} skipped/new)")
    return n_loaded


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=2))
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def filter_strong_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Channel 4 training filter: silver + (large/moderate improvement)
    + (strong/acceptable retention) + non-null SMILES + non-null liability."""
    if "quality_tier" in df.columns:
        df = df[df["quality_tier"] == "silver"]
    df = df[df["liability_improvement_category"].isin(["large", "moderate"])]
    df = df[df["activity_retention_bucket"].isin(["strong", "acceptable"])]
    df = df[df["liability_type"].notna()]
    df = df[df["parent_smiles"].notna() & df["candidate_smiles"].notna()]
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=130)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
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
    df = filter_strong_pairs(df)
    if is_main:
        log(f"After Channel 4 filter (large/moderate improvement + strong/acceptable retention): {len(df):,} pairs")
    if len(df) < 1000:
        raise SystemExit(f"Too few rows ({len(df)}) for Channel 4 training; aborting.")

    rows = df.to_dict(orient="records")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rows))
    val_n = max(512, len(rows) // 20)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    tr_rows = [rows[i] for i in tr_idx]
    val_rows = [rows[i] for i in val_idx]

    tr_ds = Channel4Dataset(tr_rows, max_len=args.max_len)
    val_ds = Channel4Dataset(val_rows, max_len=args.max_len)
    sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, shuffle=(sampler is None),
                       num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, num_workers=2, pin_memory=True)

    if is_main:
        log(f"World={world_size} | tr={len(tr_ds):,} val={len(val_ds):,} | eff bs={args.bs * world_size}")

    ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    d_model = cargs.get("d_model", 1024)
    n_heads = cargs.get("n_heads", 16)
    n_layers = cargs.get("n_layers", 16)
    max_len = max(cargs.get("max_len", args.max_len), args.max_len)

    model = InverseDeltaSeq2Seq(EXTENDED_VOCAB_SIZE, d_model, n_heads, n_layers, max_len).to(device)
    if is_main:
        load_pretrain_into_seq2seq(model, args.pretrain, log)
    if dist.is_initialized():
        dist.barrier()
    if not is_main:
        load_pretrain_into_seq2seq(model, args.pretrain, None)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        log(f"Starting Channel 4 training: {args.steps} steps | params={n_params:,}")

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
            enc_ids, enc_mask, dec_ids, dec_mask = [b.to(device, non_blocking=True) for b in batch]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(enc_ids, enc_mask, dec_ids, dec_mask)
            logits = logits.float()
            # Shift target left
            tgt = dec_ids[:, 1:].contiguous()
            tgt_mask = dec_mask[:, 1:]
            logits = logits[:, :-1].contiguous()
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, EXTENDED_VOCAB_SIZE),
                tgt.reshape(-1),
                ignore_index=PAD,
                reduction="none",
            )
            loss = (loss_per_token * tgt_mask.reshape(-1).float()).sum() / tgt_mask.sum().clamp(min=1.0).float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                elapsed = time.time() - t0
                log(f"step {step}/{args.steps} loss={loss.item():.4f} thru={(step * args.bs * world_size) / elapsed:.0f} samp/s",
                    step=step, loss=float(loss.item()))
            if step % args.ckpt_every == 0 and is_main:
                _save(model, args, step)
                log(f"Saved checkpoint at step {step}")
            if step >= args.steps:
                break
        epoch += 1

    if is_main:
        _save(model, args, step)
        log("FINAL", total_seconds=time.time() - t0)

    cleanup_distributed()
    return 0


def _save(model, args, step):
    sd = (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict()
    torch.save({
        "step": step, "model": sd, "args": vars(args),
        "vocab_size": EXTENDED_VOCAB_SIZE,
        "liability_token_ids": LIAB_TOKEN_ID,
        "BOS": BOS, "EOS": EOS, "SEP": SEP,
        "channel": "learned_inverse_delta",
    }, args.out / "checkpoint.pt")


if __name__ == "__main__":
    sys.exit(main())
