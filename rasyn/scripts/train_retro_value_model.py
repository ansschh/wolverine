"""Train the Retro*-style value model V(node) (RETRO_PLAN R-4).

Generates offline supervision by:
  1. Sampling N molecules from the curated reactions parquet (product
     molecules + intermediates from named-intermediate registry tables).
  2. For each sampled molecule, expanding an AND-OR tree to depth D with
     the trained template proposer. Label each OR_molecule node with the
     observed cost-to-go (min over expansions): depth-to-buyable, or +inf
     if the subtree never reaches a buyable within depth D.
  3. Train a small SMILES encoder + MLP regressor on (SMILES, depth) ->
     cost-to-go.

Run on 2-4x A100 (~16-32 GPU-h, dominated by tree expansion):
    torchrun --nproc_per_node=2 --standalone scripts/train_retro_value_model.py \\
        --pretrain checkpoints/smiles_lm_200m/checkpoint.pt \\
        --template-ckpt checkpoints/retro_template_v1/checkpoint.pt \\
        --buyables rasyn/data/clean/retro/buyables.parquet \\
        --reactions rasyn/data/clean/retro/reactions_silver.parquet \\
        --steps 30000 --bs 64 --lr 2e-4 \\
        --offline-samples 100000 --expansion-depth 5 \\
        --out checkpoints/retro_value_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_retro_value")


def _maybe_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _expand_tree_offline(
    target_smiles: str,
    template_proposer,
    buyability_index,
    max_depth: int,
) -> dict[str, float]:
    """Best-first depth-limited expansion. Returns {smiles -> cost_to_go}.

    cost_to_go is the minimum depth at which this molecule reaches a
    buyable (infinity = unreached -> capped at max_depth + 1 for training).
    """
    discovered: dict[str, float] = {}
    queue = [(target_smiles, 0)]
    while queue:
        smi, d = queue.pop(0)
        if smi in discovered:
            continue
        # Compute InChIKey for buyability lookup
        from rasyn.synth.retro.reactions import inchi_key_from_smiles
        ik = inchi_key_from_smiles(smi)
        if ik is None:
            discovered[smi] = max_depth + 1
            continue
        if buyability_index.is_buyable(ik):
            discovered[smi] = 0.0
            continue
        if d >= max_depth:
            discovered[smi] = max_depth + 1
            continue
        out = template_proposer.propose(
            target_smiles=smi,
            target_inchi_key=ik,
            top_k=3,
        )
        if not out.candidates:
            discovered[smi] = max_depth + 1
            continue
        # Best subtree estimate: the precursor set whose every member is
        # buyable gives cost = 1; otherwise the recursion via children.
        best_subtree_cost = max_depth + 1
        for precursor_smiles in out.candidate_smiles:
            sub_costs = []
            for p in precursor_smiles:
                queue.append((p, d + 1))
                sub_costs.append(discovered.get(p, max_depth + 1))
            best_subtree_cost = min(best_subtree_cost, 1.0 + max(sub_costs))
        discovered[smi] = best_subtree_cost
    return discovered


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--template-ckpt", type=Path, required=True)
    p.add_argument("--templates", type=Path, required=True)
    p.add_argument("--buyables", type=Path, required=True)
    p.add_argument("--reactions", nargs="+", type=Path, required=True)
    p.add_argument("--offline-samples", type=int, default=100000)
    p.add_argument("--expansion-depth", type=int, default=5)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
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
        logger.error("torch not installed; cannot train.")
        return 1

    # Load buyability index + template proposer
    from rasyn.synth.retro.buyability import BuyabilityIndex, BuyabilityIndexConfig
    from rasyn.synth.retro.proposers import TemplateProposer, TemplateProposerConfig

    buyability = BuyabilityIndex(BuyabilityIndexConfig(parquet_path=args.buyables))
    logger.info("buyability index size: %d", len(buyability))

    template_prop = TemplateProposer(TemplateProposerConfig(
        checkpoint_path=args.template_ckpt,
        templates_path=args.templates,
    ))

    # Sample target molecules from reactions (use products as targets)
    import pyarrow.parquet as pq
    samples: list[str] = []
    for path in args.reactions:
        if path.exists():
            for row in pq.read_table(path).to_pylist():
                p = row.get("product_smiles") or row.get("product")
                if p:
                    samples.append(p)
    samples = samples[: args.offline_samples]
    logger.info("offline target sample count: %d", len(samples))

    # Generate (smiles, cost_to_go) supervision via tree expansion
    examples: list[tuple[str, float]] = []
    t0 = time.time()
    for i, smi in enumerate(samples):
        if i % 1000 == 0:
            logger.info("  expand %d/%d in %.1fs", i, len(samples), time.time() - t0)
        sub = _expand_tree_offline(smi, template_prop, buyability, args.expansion_depth)
        for s, c in sub.items():
            examples.append((s, c))
    logger.info("collected %d (smiles, cost) training pairs", len(examples))
    if not examples:
        return 1

    # Train value MLP on the (SMILES encoding, depth) -> cost
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPTS_DIR))
    from h200_smiles_lm_pretrain import VOCAB, VOCAB_SIZE, PAD  # type: ignore

    def tokenize(smi):
        ids = [VOCAB.get(c, 1) for c in smi[:args.max_len]]
        n = len(ids)
        ids = ids + [PAD] * (args.max_len - n)
        mask = [True] * n + [False] * (args.max_len - n)
        return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=bool)

    pretrain = torch.load(args.pretrain, map_location="cpu", weights_only=False)
    cargs = pretrain.get("args", {})
    d_model = cargs.get("d_model", 768)
    n_heads = cargs.get("n_heads", 12)
    n_layers = cargs.get("n_layers", 8)

    class ValueModelTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
            self.pos_emb = nn.Embedding(args.max_len, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1,
                                                 batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, 1),
            )

        def forward(self, ids, mask):
            pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            h = self.tok_emb(ids) + self.pos_emb(pos)
            h = self.encoder(h, src_key_padding_mask=~mask)
            h = self.norm(h)
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return self.head(pooled).squeeze(-1)

    model = ValueModelTorch().to(device)
    own = model.state_dict()
    for k, v in (pretrain.get("model") or {}).items():
        kk = k.removeprefix("module.")
        if kk in own and own[kk].shape == v.shape:
            own[kk] = v
    model.load_state_dict(own, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_batch():
        idxs = np.random.choice(len(examples), args.bs, replace=False)
        ids_list, mask_list, label_list = [], [], []
        for i in idxs:
            smi, cost = examples[i]
            ids, mask = tokenize(smi)
            ids_list.append(ids); mask_list.append(mask)
            label_list.append(cost)
        return (
            torch.from_numpy(np.stack(ids_list)).to(device),
            torch.from_numpy(np.stack(mask_list)).to(device),
            torch.tensor(label_list, dtype=torch.float32, device=device),
        )

    model.train()
    log_path = args.out / "training_log.jsonl"
    log_path.touch()
    t1 = time.time()
    for step in range(args.steps):
        ids, mask, lbl = get_batch()
        pred = model(ids, mask)
        loss = F.mse_loss(pred, lbl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, "loss": float(loss.item()),
                                      "t": time.time() - t1}) + "\n")
            logger.info("step %d mse=%.4f", step, loss.item())

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "vocab": dict(VOCAB),
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
    }, args.out / "checkpoint.pt")
    logger.info("wrote %s", args.out / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
