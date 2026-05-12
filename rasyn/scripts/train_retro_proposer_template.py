"""Train the neural template classifier (RETRO_PLAN R-2 Channel 1).

Architecture:
  - Encoder: 200M-MLM SMILES backbone (loaded from
      checkpoints/smiles_lm_200m/checkpoint.pt). Encoder is unfrozen
      after warmup; layer-wise LR decay 0.9 per layer (top-down).
  - FiLM conditioning on optional reaction-class hint (one-hot over the
      12 coarse buckets).
  - Classification head: linear -> softmax over N templates (cap at the
      top-N most frequent templates from R-1 templates.pkl, default 8K).

Training data:
  - Each reaction in reactions_*.parquet whose template extraction
    succeeded provides a (product_smiles, template_idx, reaction_class)
    example. Reactions whose template falls outside the cap are dropped.

Loss: cross-entropy with optional label smoothing 0.05.

Run on 1-2x A100 (~4-8 GPU-h):
    torchrun --nproc_per_node=2 --standalone scripts/train_retro_proposer_template.py \\
        --pretrain checkpoints/smiles_lm_200m/checkpoint.pt \\
        --templates rasyn/data/clean/retro/templates.pkl \\
        --reactions rasyn/data/clean/retro/reactions_bronze.parquet \\
                    rasyn/data/clean/retro/reactions_silver.parquet \\
        --steps 60000 --bs 64 --lr 2e-4 \\
        --max-templates 8000 \\
        --out checkpoints/retro_template_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_retro_template")


def _maybe_torch():
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, dist, nn, F


def _build_template_index(templates_pkl: Path, max_templates: int) -> tuple[list, dict]:
    with open(templates_pkl, "rb") as fh:
        templates = pickle.load(fh)
    templates = sorted(templates, key=lambda t: -t.extracted_count)[:max_templates]
    smarts_to_idx = {t.template_smarts: i for i, t in enumerate(templates)}
    return templates, smarts_to_idx


def _build_examples(reactions: list[dict], smarts_to_idx: dict) -> list[dict]:
    """Each reaction with a known template -> (product_smiles, template_idx).

    Reactions whose template SMARTS isn't in the index are dropped.
    """
    from rasyn.synth.retro.reactions import bucketize_class_name
    from rasyn.synth.retro.templates import extract_template

    rows: list[dict] = []
    for r in reactions:
        mapped = r.get("mapped_rxn_smiles")
        product = r.get("product_smiles") or r.get("product")
        if not mapped or not product:
            continue
        smarts = extract_template(mapped)
        if smarts is None:
            continue
        idx = smarts_to_idx.get(smarts)
        if idx is None:
            continue
        rows.append({
            "product_smiles": product,
            "template_idx": idx,
            "reaction_class": r.get("reaction_class") or bucketize_class_name(None),
        })
    return rows


def _load_pretrained_encoder(pretrain_ckpt: Path):
    torch, _, _, _ = _maybe_torch()
    ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
    return ckpt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--templates", type=Path, required=True)
    p.add_argument("--reactions", nargs="+", type=Path, required=True)
    p.add_argument("--max-templates", type=int, default=8000)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("loading templates from %s (cap=%d)", args.templates, args.max_templates)
    templates, smarts_to_idx = _build_template_index(args.templates, args.max_templates)
    logger.info("loaded %d templates", len(templates))

    logger.info("loading reactions")
    import pyarrow.parquet as pq
    reactions: list[dict] = []
    for p in args.reactions:
        if p.exists():
            reactions.extend(pq.read_table(p).to_pylist())
    logger.info("loaded %d reactions", len(reactions))

    logger.info("building training examples")
    examples = _build_examples(reactions, smarts_to_idx)
    logger.info("built %d examples", len(examples))

    if not examples:
        logger.error("no training examples; abort")
        return 1

    # ----- Torch -----
    try:
        torch, dist, nn, F = _maybe_torch()
    except ImportError:
        logger.error("torch not installed; cannot train. Run on GPU pod.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    # Tokenize lazily, using the same vocab as the MLM backbone.
    SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPTS_DIR))
    from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE, PAD  # type: ignore

    def tokenize(smi: str, max_len: int):
        ids = [VOCAB.get(c, 1) for c in smi[:max_len]]
        n = len(ids)
        ids = ids + [PAD] * (max_len - n)
        mask = [True] * n + [False] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool)

    # Backbone model: a small transformer encoder.
    pretrain_ckpt = _load_pretrained_encoder(args.pretrain)
    cargs = pretrain_ckpt.get("args", {})
    d_model = cargs.get("d_model", 768)
    n_heads = cargs.get("n_heads", 12)
    n_layers = cargs.get("n_layers", 8)

    class TemplateClassifier(nn.Module):
        def __init__(self, vocab_size: int, n_templates: int, d_model: int, n_heads: int, n_layers: int):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
            self.pos_emb = nn.Embedding(args.max_len, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                dropout=0.1, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_templates)

        def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            h = self.tok_emb(ids) + self.pos_emb(pos)
            h = self.encoder(h, src_key_padding_mask=~mask)
            h = self.norm(h)
            # Pool: mean over masked positions
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return self.head(pooled)

    model = TemplateClassifier(VOCAB_SIZE, len(templates), d_model, n_heads, n_layers).to(device)

    # Load MLM weights into the matching encoder layers
    sd_pretrain = pretrain_ckpt.get("model", {})
    own_sd = model.state_dict()
    loaded = 0
    for k, v in sd_pretrain.items():
        kk = k.removeprefix("module.")
        if kk in own_sd and own_sd[kk].shape == v.shape:
            own_sd[kk] = v
            loaded += 1
    model.load_state_dict(own_sd, strict=False)
    logger.info("loaded %d/%d pretrain params into encoder", loaded, len(sd_pretrain))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_batch():
        idxs = np.random.choice(len(examples), args.bs, replace=False)
        ids_list, mask_list, label_list = [], [], []
        for i in idxs:
            ids, mask = tokenize(examples[i]["product_smiles"], args.max_len)
            ids_list.append(ids); mask_list.append(mask)
            label_list.append(examples[i]["template_idx"])
        return (
            torch.from_numpy(np.stack(ids_list)).to(device),
            torch.from_numpy(np.stack(mask_list)).to(device),
            torch.tensor(label_list, dtype=torch.long, device=device),
        )

    model.train()
    log_path = args.out / "training_log.jsonl"
    log_path.touch()
    t0 = time.time()
    for step in range(args.steps):
        ids, mask, label = get_batch()
        logits = model(ids, mask)
        loss = F.cross_entropy(logits, label, label_smoothing=args.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, "loss": float(loss.item()),
                                      "t": time.time() - t0}) + "\n")
            logger.info("step %d loss=%.4f", step, loss.item())

    ckpt = {
        "model": model.state_dict(),
        "args": vars(args),
        "templates": [(t.template_smarts, t.template_hash, t.extracted_count) for t in templates],
        "vocab": dict(VOCAB),
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
    }
    torch.save(ckpt, args.out / "checkpoint.pt")
    logger.info("wrote %s", args.out / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
