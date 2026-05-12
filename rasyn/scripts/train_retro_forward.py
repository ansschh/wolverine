"""Train the forward-reaction validator (RETRO_PLAN R-3, Lock 5).

Reactants SMILES -> product SMILES via encoder-decoder transformer.
Same architecture as the seq2seq retro proposer, but reversed direction.
Encoder init from 200M MLM; decoder init from 200M AR LM.

Run on 3-5x A100 (~24-48 GPU-h):
    torchrun --nproc_per_node=5 --standalone scripts/train_retro_forward.py \\
        --pretrain-encoder checkpoints/smiles_lm_200m/checkpoint.pt \\
        --pretrain-decoder checkpoints/smiles_ar_lm_200m/checkpoint.pt \\
        --reactions rasyn/data/clean/retro/reactions_bronze.parquet \\
                    rasyn/data/clean/retro/reactions_silver.parquet \\
        --steps 150000 --bs 64 --lr 1e-4 \\
        --out checkpoints/retro_forward_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_retro_forward")


CLASS_BUCKETS = [
    "amide_coupling", "suzuki_coupling", "buchwald_hartwig", "reductive_amination",
    "sn2", "sn_ar", "negishi", "wittig", "click",
    "protection_deprotection", "other_cross_coupling", "unclassified",
]


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
    BOS, EOS = 3, 2

    def tok_src(smi, max_len):
        ids = [VOCAB.get(c, 1) for c in smi[:max_len]]
        n = len(ids)
        ids = ids + [PAD] * (max_len - n)
        mask = [True] * n + [False] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool)

    def tok_tgt(smi, max_len):
        ids = [BOS] + [VOCAB.get(c, 1) for c in smi[: max_len - 2]] + [EOS]
        n = len(ids)
        ids = ids + [PAD] * (max_len - n)
        mask = [True] * n + [False] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool)

    import pyarrow.parquet as pq
    reactions = []
    for path in args.reactions:
        if path.exists():
            reactions.extend(pq.read_table(path).to_pylist())
    logger.info("loaded %d reactions", len(reactions))

    # Build (reactants_str, product_str, class_idx) examples
    examples = []
    for r in reactions:
        prod = r.get("product_smiles") or r.get("product")
        reactants = r.get("reactant_smiles") or r.get("reactants") or []
        if not prod or not reactants:
            continue
        src = ".".join(reactants)
        rc = r.get("reaction_class") or "unclassified"
        cidx = CLASS_BUCKETS.index(rc) if rc in CLASS_BUCKETS else len(CLASS_BUCKETS) - 1
        examples.append((src, prod, cidx))
    logger.info("built %d examples", len(examples))
    if not examples:
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_ckpt = torch.load(args.pretrain_encoder, map_location="cpu", weights_only=False)
    dec_ckpt = torch.load(args.pretrain_decoder, map_location="cpu", weights_only=False)
    cargs = enc_ckpt.get("args", {})
    d_model = cargs.get("d_model", 768)
    n_heads = cargs.get("n_heads", 12)
    n_layers = cargs.get("n_layers", 8)

    class FiLM(nn.Module):
        def __init__(self, d, k):
            super().__init__()
            self.g = nn.Embedding(k, d); self.b = nn.Embedding(k, d)
            nn.init.ones_(self.g.weight); nn.init.zeros_(self.b.weight)
        def forward(self, h, cidx):
            return self.g(cidx).unsqueeze(1) * h + self.b(cidx).unsqueeze(1)

    class ForwardSeq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.src_tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
            self.tgt_tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
            self.src_pos = nn.Embedding(args.max_len, d_model)
            self.tgt_pos = nn.Embedding(args.max_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1,
                                                    batch_first=True, activation="gelu")
            dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, 0.1,
                                                    batch_first=True, activation="gelu")
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
            h = self.decoder(h_tgt, h_src, tgt_mask=causal,
                              tgt_key_padding_mask=~tgt_mask,
                              memory_key_padding_mask=~src_mask)
            h = self.norm(h)
            return self.head(h)

    model = ForwardSeq2Seq().to(device)
    own = model.state_dict()
    for k, v in (enc_ckpt.get("model") or {}).items():
        kk = k.removeprefix("module.")
        if kk in own and own[kk].shape == v.shape:
            own[kk] = v
    for k, v in (dec_ckpt.get("model") or {}).items():
        kk = k.removeprefix("module.")
        if kk in own and own[kk].shape == v.shape:
            own[kk] = v
    model.load_state_dict(own, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_batch():
        idxs = np.random.choice(len(examples), args.bs, replace=False)
        src_ids, src_masks = [], []
        tgt_ids, tgt_masks, cidx_list = [], [], []
        for i in idxs:
            src, tgt, cidx = examples[i]
            si, sm = tok_src(src, args.max_len)
            ti, tm = tok_tgt(tgt, args.max_len)
            src_ids.append(si); src_masks.append(sm)
            tgt_ids.append(ti); tgt_masks.append(tm)
            cidx_list.append(cidx)
        return (
            torch.from_numpy(np.stack(src_ids)).to(device),
            torch.from_numpy(np.stack(src_masks)).to(device),
            torch.from_numpy(np.stack(tgt_ids)).to(device),
            torch.from_numpy(np.stack(tgt_masks)).to(device),
            torch.tensor(cidx_list, dtype=torch.long, device=device),
        )

    model.train()
    log_path = args.out / "training_log.jsonl"
    log_path.touch()
    t0 = time.time()
    for step in range(args.steps):
        src_ids, src_mask, tgt_ids, tgt_mask, cidx = get_batch()
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
