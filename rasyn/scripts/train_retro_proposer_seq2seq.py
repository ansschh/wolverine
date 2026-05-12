"""Train the SMILES seq2seq retro proposer (RETRO_PLAN R-2 Channel 3).

Product SMILES -> '.'-joined reactants SMILES, via an encoder-decoder
transformer (~200M total params).

  - Encoder init from 200M MLM (checkpoints/smiles_lm_200m/).
  - Decoder init from 200M AR LM (checkpoints/smiles_ar_lm_200m/).
  - FiLM gain+bias on every layer from reaction-class one-hot
    (12 coarse buckets per RETRO_PLAN R-2 spec; ignore-token if unknown).

Loss: token-level cross-entropy on the reactants SMILES (label-smoothed
0.05) ignoring the PAD token.

Run on 3-5x A100 (~24-48 GPU-h):
    torchrun --nproc_per_node=5 --standalone scripts/train_retro_proposer_seq2seq.py \\
        --pretrain-encoder checkpoints/smiles_lm_200m/checkpoint.pt \\
        --pretrain-decoder checkpoints/smiles_ar_lm_200m/checkpoint.pt \\
        --reactions rasyn/data/clean/retro/reactions_bronze.parquet \\
                    rasyn/data/clean/retro/reactions_silver.parquet \\
        --steps 150000 --bs 64 --lr 1e-4 \\
        --out checkpoints/retro_seq2seq_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_retro_seq2seq")


CLASS_BUCKETS = [
    "amide_coupling", "suzuki_coupling", "buchwald_hartwig", "reductive_amination",
    "sn2", "sn_ar", "negishi", "wittig", "click",
    "protection_deprotection", "other_cross_coupling", "unclassified",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_BUCKETS)}


def _maybe_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-encoder", type=Path, required=True)
    p.add_argument("--pretrain-decoder", type=Path, required=True)
    p.add_argument("--reactions", nargs="+", type=Path, required=True)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)

    try:
        torch, nn, F = _maybe_torch()
    except ImportError:
        logger.error("torch not installed; cannot train. Run on GPU pod.")
        return 1

    SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPTS_DIR))
    from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE, PAD  # type: ignore
    BOS, EOS = 3, 2  # reuse CLS / MASK_TOK indexes from the MLM vocab

    def tokenize_src(smi: str, max_len: int):
        ids = [VOCAB.get(c, 1) for c in smi[:max_len]]
        n = len(ids)
        ids = ids + [PAD] * (max_len - n)
        mask = [True] * n + [False] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool)

    def tokenize_tgt(smi: str, max_len: int):
        ids = [BOS] + [VOCAB.get(c, 1) for c in smi[: max_len - 2]] + [EOS]
        n = len(ids)
        ids = ids + [PAD] * (max_len - n)
        mask = [True] * n + [False] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool)

    # Load reactions
    import pyarrow.parquet as pq
    reactions: list[dict] = []
    for p in args.reactions:
        if p.exists():
            reactions.extend(pq.read_table(p).to_pylist())
    logger.info("loaded %d reactions", len(reactions))

    # Build (src, tgt, class_idx) examples
    examples: list[tuple[str, str, int]] = []
    for r in reactions:
        prod = r.get("product_smiles") or r.get("product")
        reactants = r.get("reactant_smiles") or r.get("reactants") or []
        if not prod or not reactants:
            continue
        tgt = ".".join(reactants)
        rc = r.get("reaction_class") or "unclassified"
        cidx = CLASS_TO_IDX.get(rc, CLASS_TO_IDX["unclassified"])
        examples.append((prod, tgt, cidx))
    logger.info("built %d examples", len(examples))
    if not examples:
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    enc_ckpt = torch.load(args.pretrain_encoder, map_location="cpu", weights_only=False)
    dec_ckpt = torch.load(args.pretrain_decoder, map_location="cpu", weights_only=False)
    cargs = enc_ckpt.get("args", {})
    d_model = cargs.get("d_model", 768)
    n_heads = cargs.get("n_heads", 12)
    n_layers = cargs.get("n_layers", 8)

    class FiLM(nn.Module):
        def __init__(self, d_model: int, n_classes: int):
            super().__init__()
            self.gamma = nn.Embedding(n_classes, d_model)
            self.beta = nn.Embedding(n_classes, d_model)
            nn.init.ones_(self.gamma.weight)
            nn.init.zeros_(self.beta.weight)

        def forward(self, h: torch.Tensor, cidx: torch.Tensor) -> torch.Tensor:
            g = self.gamma(cidx).unsqueeze(1)
            b = self.beta(cidx).unsqueeze(1)
            return g * h + b

    class Seq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.src_tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
            self.tgt_tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
            self.src_pos = nn.Embedding(args.max_len, d_model)
            self.tgt_pos = nn.Embedding(args.max_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                    dropout=0.1, batch_first=True, activation="gelu")
            dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                    dropout=0.1, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
            self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.film = FiLM(d_model, len(CLASS_BUCKETS))
            self.head = nn.Linear(d_model, VOCAB_SIZE)

        def forward(self, src_ids, src_mask, tgt_in, tgt_mask, cidx):
            pos_src = torch.arange(src_ids.size(1), device=src_ids.device).unsqueeze(0)
            pos_tgt = torch.arange(tgt_in.size(1), device=tgt_in.device).unsqueeze(0)
            h_src = self.src_tok(src_ids) + self.src_pos(pos_src)
            h_src = self.encoder(h_src, src_key_padding_mask=~src_mask)
            h_src = self.film(h_src, cidx)
            h_tgt = self.tgt_tok(tgt_in) + self.tgt_pos(pos_tgt)
            causal = torch.triu(
                torch.ones(tgt_in.size(1), tgt_in.size(1), device=tgt_in.device, dtype=torch.bool),
                diagonal=1,
            )
            h = self.decoder(
                h_tgt, h_src, tgt_mask=causal,
                tgt_key_padding_mask=~tgt_mask,
                memory_key_padding_mask=~src_mask,
            )
            h = self.norm(h)
            return self.head(h)

    model = Seq2Seq().to(device)

    # Load encoder weights
    sd_enc = enc_ckpt.get("model", {})
    own = model.state_dict()
    for k, v in sd_enc.items():
        kk = k.removeprefix("module.")
        if kk in own and own[kk].shape == v.shape:
            own[kk] = v
    # Load decoder weights (partial — tok_emb + lm_head pretrained from AR LM)
    sd_dec = dec_ckpt.get("model", {})
    for k, v in sd_dec.items():
        kk = k.removeprefix("module.")
        if kk in own and own[kk].shape == v.shape:
            own[kk] = v
    model.load_state_dict(own, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_batch():
        idxs = np.random.choice(len(examples), args.bs, replace=False)
        src_ids_list, src_mask_list = [], []
        tgt_ids_list, tgt_mask_list, cidx_list = [], [], []
        for i in idxs:
            prod, tgt, cidx = examples[i]
            si, sm = tokenize_src(prod, args.max_len)
            ti, tm = tokenize_tgt(tgt, args.max_len)
            src_ids_list.append(si); src_mask_list.append(sm)
            tgt_ids_list.append(ti); tgt_mask_list.append(tm)
            cidx_list.append(cidx)
        src_ids = torch.from_numpy(np.stack(src_ids_list)).to(device)
        src_mask = torch.from_numpy(np.stack(src_mask_list)).to(device)
        tgt_ids = torch.from_numpy(np.stack(tgt_ids_list)).to(device)
        tgt_mask = torch.from_numpy(np.stack(tgt_mask_list)).to(device)
        cidx_t = torch.tensor(cidx_list, dtype=torch.long, device=device)
        return src_ids, src_mask, tgt_ids, tgt_mask, cidx_t

    model.train()
    log_path = args.out / "training_log.jsonl"
    log_path.touch()
    t0 = time.time()
    for step in range(args.steps):
        src_ids, src_mask, tgt_ids, tgt_mask, cidx = get_batch()
        # Teacher forcing: input is tgt[:-1], label is tgt[1:]
        tgt_in = tgt_ids[:, :-1]
        tgt_in_mask = tgt_mask[:, :-1]
        labels = tgt_ids[:, 1:]
        logits = model(src_ids, src_mask, tgt_in, tgt_in_mask, cidx)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1),
            ignore_index=PAD, label_smoothing=args.label_smoothing,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, "loss": float(loss.item()),
                                      "t": time.time() - t0}) + "\n")
            logger.info("step %d loss=%.4f", step, loss.item())

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "vocab": dict(VOCAB),
        "class_buckets": CLASS_BUCKETS,
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
    }, args.out / "checkpoint.pt")
    logger.info("wrote %s", args.out / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
