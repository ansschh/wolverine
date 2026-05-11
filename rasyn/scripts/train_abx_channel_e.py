"""ABX Channel E: organism-conditioned antibacterial generator.

Encoder-decoder transformer (clone of Channel 4 inverse-delta seq2seq, adapted):
  Encoder input:  [BOS][ORG_<organism>][SEP] (minimal organism conditioning;
                  no parent molecule — antibiotic generation is organism-conditional,
                  not parent-rescue).
  Decoder target: [BOS] + full antibacterial SMILES + [EOS]

Trained on `generative_training_examples.parquet` from `build_abx_dataset.py`
(24,726 active (molecule, organism) tuples across 8 organisms). Filter for
Channel E: ALL active rows.

Init: backbone weights from `smiles_lm_200m/checkpoint.pt` (Stage-1 200M MLM).

Run on Pod B (8x A100 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/train_abx_channel_e.py \\
        --pretrain rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --gen-examples rasyn/data/clean/antibiotic/generative_training_examples.parquet \\
        --steps 4000 --bs 32 --lr 1e-4 --seed 42 \\
        --out rasyn/data/clean/abx_channel_e

Outputs:
    out/checkpoint.pt
    out/training_log.jsonl

Per L41: NOT a graph-diffusion placeholder — this is a real trained seq2seq
producing organism-conditional candidates. Same architecture family as
ADMET Ch4/Ch5; different teacher data (antibiotic active molecules) and
different conditioning vocabulary (organism instead of liability).
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


# 11 organism conditioning tokens — must match rasyn/antibiotic/schemas.py Organism literal.
ORGANISM_TOKENS = [
    "[ORG_ECOLI]",
    "[ORG_SAUREUS]",
    "[ORG_MRSA]",
    "[ORG_KPNEUMONIAE]",
    "[ORG_ABAUMANNII]",
    "[ORG_PAERUGINOSA]",
    "[ORG_NGONORRHOEAE]",
    "[ORG_MTB]",
    "[ORG_CDIFFICILE]",
    "[ORG_HPYLORI]",
    "[ORG_UNKNOWN]",
]
ORGANISM_TYPE2STR = {
    "E.coli":            "[ORG_ECOLI]",
    "S.aureus":          "[ORG_SAUREUS]",
    "MRSA":              "[ORG_MRSA]",
    "K.pneumoniae":      "[ORG_KPNEUMONIAE]",
    "A.baumannii":       "[ORG_ABAUMANNII]",
    "P.aeruginosa":      "[ORG_PAERUGINOSA]",
    "N.gonorrhoeae":     "[ORG_NGONORRHOEAE]",
    "MTB":               "[ORG_MTB]",
    "C.difficile":       "[ORG_CDIFFICILE]",
    "H.pylori":          "[ORG_HPYLORI]",
    "unknown":           "[ORG_UNKNOWN]",
}

# Special tokens beyond base SMILES vocab.
BOS = VOCAB_SIZE
EOS = VOCAB_SIZE + 1
SEP = VOCAB_SIZE + 2
ORG_OFFSET = VOCAB_SIZE + 3
EXTENDED_VOCAB_SIZE = ORG_OFFSET + len(ORGANISM_TOKENS)
ORG_TOKEN_ID = {tok: ORG_OFFSET + i for i, tok in enumerate(ORGANISM_TOKENS)}


def tokenize_smiles(smi: str, max_len: int) -> list[int]:
    UNK = 1
    return [VOCAB.get(c, UNK) for c in smi[:max_len]]


def encode_input(organism: str, max_len: int) -> tuple[list[int], list[bool]]:
    """Encoder input: [BOS][ORG_<type>][SEP] padded. No parent molecule."""
    org_tok = ORGANISM_TYPE2STR.get(organism, "[ORG_UNKNOWN]")
    seq = [BOS, ORG_TOKEN_ID[org_tok], SEP]
    n = len(seq)
    seq = seq + [PAD] * (max_len - n)
    mask = [True] * n + [False] * (max_len - n)
    return seq, mask


def encode_target(smi: str, max_len: int) -> tuple[list[int], list[bool]]:
    """Decoder target: [BOS] + SMILES_tokens + [EOS] padded."""
    ids = tokenize_smiles(smi, max_len - 2)
    seq = [BOS] + ids + [EOS]
    n = len(seq)
    seq = seq + [PAD] * (max_len - n)
    mask = [True] * n + [False] * (max_len - n)
    return seq, mask


class ABXChannelEDataset(Dataset):
    def __init__(self, rows: list[dict], enc_len: int = 16, dec_len: int = 130):
        self.rows = rows
        self.enc_len = enc_len
        self.dec_len = dec_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc_ids, enc_mask = encode_input(r.get("organism_context", "unknown"), self.enc_len)
        dec_ids, dec_mask = encode_target(r["full_molecule_smiles"], self.dec_len)
        return (
            torch.tensor(enc_ids, dtype=torch.long),
            torch.tensor(enc_mask, dtype=torch.bool),
            torch.tensor(dec_ids, dtype=torch.long),
            torch.tensor(dec_mask, dtype=torch.bool),
        )


class ABXChannelESeq2Seq(nn.Module):
    """Encoder-decoder transformer (matches Channel 4 architecture, ABX vocab)."""

    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 16,
                 n_layers: int = 16, max_len: int = 130):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers // 2)
        self.dec_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self.max_len = max_len

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask):
        T = enc_ids.size(1)
        pos = torch.arange(T, device=enc_ids.device).unsqueeze(0).expand_as(enc_ids)
        x = self.tok_emb(enc_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=~enc_mask)
        x = self.norm(x)
        Td = dec_ids.size(1)
        dpos = torch.arange(Td, device=dec_ids.device).unsqueeze(0).expand_as(dec_ids)
        y = self.tok_emb(dec_ids) + self.pos_emb(dpos)
        causal = torch.triu(torch.full((Td, Td), float("-inf"), device=dec_ids.device), diagonal=1)
        y = self.decoder(
            y, x, tgt_mask=causal, memory_key_padding_mask=~enc_mask,
            tgt_key_padding_mask=~dec_mask,
        )
        y = self.dec_norm(y)
        return self.lm_head(y)


def load_pretrain_into_seq2seq(model: ABXChannelESeq2Seq, ckpt_path: Path, log) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pretrain_sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model_sd = model.state_dict()
    if "tok_emb.weight" in pretrain_sd and pretrain_sd["tok_emb.weight"].shape != model_sd["tok_emb.weight"].shape:
        old = pretrain_sd.pop("tok_emb.weight")
        new = model_sd["tok_emb.weight"].clone()
        n_copy = min(old.shape[0], new.shape[0])
        new[:n_copy] = old[:n_copy]
        model_sd["tok_emb.weight"] = new
    n_loaded = n_skipped = 0
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


def filter_examples_e(df: pd.DataFrame) -> pd.DataFrame:
    """Channel E filter: all active rows with valid SMILES + known organism."""
    df = df[df["full_molecule_smiles"].notna()]
    df = df[df["organism_context"].notna()]
    df = df[df["activity_label"] == "active"]
    return df


def main(filter_fn=None, channel_marker: str = "abx_channel_e") -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--gen-examples", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--enc-len", type=int, default=16)
    p.add_argument("--dec-len", type=int, default=130)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    filter_fn = filter_fn or filter_examples_e

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
        log(f"Loading gen examples from {args.gen_examples}")
    df = pd.read_parquet(args.gen_examples)
    df = filter_fn(df)
    if is_main:
        log(f"After filter ({channel_marker}): {len(df):,} rows")
    if len(df) < 500:
        raise SystemExit(f"Too few rows ({len(df)}); aborting.")

    rows = df.to_dict(orient="records")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rows))
    val_n = max(256, len(rows) // 20)
    val_idx = perm[:val_n]
    tr_idx = perm[val_n:]
    tr_rows = [rows[i] for i in tr_idx]
    val_rows = [rows[i] for i in val_idx]

    tr_ds = ABXChannelEDataset(tr_rows, enc_len=args.enc_len, dec_len=args.dec_len)
    val_ds = ABXChannelEDataset(val_rows, enc_len=args.enc_len, dec_len=args.dec_len)
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
    max_len = max(cargs.get("max_len", args.dec_len), args.dec_len, args.enc_len)

    model = ABXChannelESeq2Seq(EXTENDED_VOCAB_SIZE, d_model, n_heads, n_layers, max_len).to(device)
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
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        log(f"Starting {channel_marker} training: {args.steps} steps | params={n_params:,}")

    t0 = time.time()
    step = epoch = 0
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
            tgt = dec_ids[:, 1:].contiguous()
            tgt_mask = dec_mask[:, 1:]
            logits = logits[:, :-1].contiguous()
            lpt = F.cross_entropy(
                logits.reshape(-1, EXTENDED_VOCAB_SIZE), tgt.reshape(-1),
                ignore_index=PAD, reduction="none",
            )
            loss = (lpt * tgt_mask.reshape(-1).float()).sum() / tgt_mask.sum().clamp(min=1.0).float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                el = time.time() - t0
                log(f"step {step}/{args.steps} loss={loss.item():.4f} thru={(step*args.bs*world_size)/el:.0f} samp/s",
                    step=step, loss=float(loss.item()))
            if step % args.ckpt_every == 0 and is_main:
                _save(model, args, step, channel_marker)
                log(f"Saved ckpt at step {step}")
            if step >= args.steps:
                break
        epoch += 1

    if is_main:
        _save(model, args, step, channel_marker)
        log("FINAL", total_seconds=time.time() - t0)

    cleanup_distributed()
    return 0


def _save(model, args, step, channel_marker: str):
    sd = (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict()
    torch.save({
        "step": step, "model": sd, "args": vars(args),
        "vocab_size": EXTENDED_VOCAB_SIZE,
        "organism_token_ids": ORG_TOKEN_ID,
        "BOS": BOS, "EOS": EOS, "SEP": SEP,
        "channel": channel_marker,
    }, args.out / "checkpoint.pt")


if __name__ == "__main__":
    sys.exit(main())
