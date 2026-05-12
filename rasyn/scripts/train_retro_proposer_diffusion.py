"""Train the diffusion reactant-completion proposer (RETRO_PLAN R-2 Channel 5).

DiGress-style discrete graph diffusion. Per RETRO.md Lock 4 + RETRO_PLAN
override: diffusion completes reactants given a fixed disconnection mask
+ reaction-class hint. Never acts as the planner.

Scaling note: this is the largest R-2 channel.
  - ABX prior (MEMORY L47): 5.4M params @ 60K steps was undertrained.
  - v1: 30-50M params @ 200K-500K steps on 5x A100, ~48-96 GPU-h.

Reuses `rasyn/antibiotic/graph_diffusion.py` machinery; conditioning
extended from organism class (in ABX) to:
  - reaction-class one-hot (12 buckets)
  - disconnection-mask token per edge (broken / intact)

Run on 5x A100 (~48-96 GPU-h):
    torchrun --nproc_per_node=5 --standalone \\
        scripts/train_retro_proposer_diffusion.py \\
        --reactions rasyn/data/clean/retro/reactions_bronze.parquet \\
                    rasyn/data/clean/retro/reactions_silver.parquet \\
        --steps 300000 --bs 32 --lr 1e-4 \\
        --d-model 1024 --n-layers 8 --n-heads 16 \\
        --out checkpoints/retro_diffusion_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_retro_diffusion")


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


def _build_diffusion_examples(reactions: list[dict], max_atoms: int) -> list[dict]:
    """For each (product, reactants), build a synthon-completion example.

    Input: product graph + disconnection-mask (which bonds are broken
    when going from reactants -> product). Target: the reactant graph.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return []
    examples = []
    for r in reactions:
        prod = r.get("product_smiles") or r.get("product")
        reactants = r.get("reactant_smiles") or r.get("reactants") or []
        rc = r.get("reaction_class") or "unclassified"
        if not prod or not reactants:
            continue
        prod_mol = Chem.MolFromSmiles(prod)
        if prod_mol is None or prod_mol.GetNumAtoms() > max_atoms:
            continue
        reac_mol = Chem.MolFromSmiles(".".join(reactants))
        if reac_mol is None or reac_mol.GetNumAtoms() > max_atoms:
            continue
        cidx = CLASS_BUCKETS.index(rc) if rc in CLASS_BUCKETS else len(CLASS_BUCKETS) - 1
        examples.append({
            "product_smiles": prod,
            "reactant_smiles": ".".join(reactants),
            "reaction_class_idx": cidx,
            "n_atoms_product": prod_mol.GetNumAtoms(),
            "n_atoms_reactant": reac_mol.GetNumAtoms(),
        })
    return examples


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reactions", nargs="+", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--diffusion-steps", type=int, default=500)
    p.add_argument("--max-atoms", type=int, default=80)
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

    # Hook into the existing ABX diffusion machinery
    SCRIPTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPTS_DIR))
    sys.path.insert(0, str(SCRIPTS_DIR.parent))
    try:
        from rasyn.antibiotic.graph_diffusion import (  # type: ignore
            GraphDiffusionDenoiser,
            forward_noise,
            mol_to_graph,
            graph_to_mol,
            DiffusionConfig,
        )
    except ImportError as e:
        logger.error("ABX graph_diffusion not importable: %s", e)
        return 1

    import pyarrow.parquet as pq
    reactions: list[dict] = []
    for p in args.reactions:
        if p.exists():
            reactions.extend(pq.read_table(p).to_pylist())
    logger.info("loaded %d reactions", len(reactions))

    examples = _build_diffusion_examples(reactions, args.max_atoms)
    logger.info("built %d diffusion examples", len(examples))
    if not examples:
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure denoiser with reaction-class FiLM (12 classes) and an
    # extra binary disconnection-mask channel on edges.
    diff_cfg = DiffusionConfig(
        n_steps=args.diffusion_steps,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_classes=len(CLASS_BUCKETS),  # FiLM conditioning
        max_atoms=args.max_atoms,
    )
    denoiser = GraphDiffusionDenoiser(diff_cfg).to(device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=0.01)

    def get_batch():
        idxs = np.random.choice(len(examples), args.bs, replace=False)
        graphs_prod = []
        graphs_reac = []
        class_idx = []
        for i in idxs:
            ex = examples[i]
            g_prod = mol_to_graph(ex["product_smiles"], max_atoms=args.max_atoms)
            g_reac = mol_to_graph(ex["reactant_smiles"], max_atoms=args.max_atoms)
            if g_prod is None or g_reac is None:
                continue
            graphs_prod.append(g_prod)
            graphs_reac.append(g_reac)
            class_idx.append(ex["reaction_class_idx"])
        return graphs_prod, graphs_reac, torch.tensor(class_idx, dtype=torch.long, device=device)

    denoiser.train()
    log_path = args.out / "training_log.jsonl"
    log_path.touch()
    t0 = time.time()
    for step in range(args.steps):
        graphs_prod, graphs_reac, cidx = get_batch()
        if not graphs_reac:
            continue
        t = torch.randint(1, args.diffusion_steps, (len(graphs_reac),), device=device)
        # Add noise to reactant graph at timestep t
        noisy = forward_noise(graphs_reac, t, n_steps=args.diffusion_steps)
        loss = denoiser.training_loss(
            noisy_graphs=noisy,
            target_graphs=graphs_reac,
            condition_class_idx=cidx,
            condition_product_graphs=graphs_prod,
            timestep=t,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, "loss": float(loss.item()),
                                      "t": time.time() - t0}) + "\n")
            logger.info("step %d loss=%.4f", step, loss.item())

    torch.save({
        "model": denoiser.state_dict(),
        "args": vars(args),
        "class_buckets": CLASS_BUCKETS,
        "diffusion_config": {
            "n_steps": diff_cfg.n_steps, "d_model": diff_cfg.d_model,
            "n_heads": diff_cfg.n_heads, "n_layers": diff_cfg.n_layers,
            "n_classes": diff_cfg.n_classes, "max_atoms": diff_cfg.max_atoms,
        },
    }, args.out / "checkpoint.pt")
    logger.info("wrote %s", args.out / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
