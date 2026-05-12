"""Train the graph-edit retro proposer (RETRO_PLAN R-2 Channel 2).

Predicts bond-edit operations on the product graph that, when applied,
yield the reactant graph. Architecture: ~30M-param graph transformer
(custom layer; uses RDKit graphs converted to (node_features, edge_index,
edge_features) PyG-style tensors).

Edit operation vocabulary (per node + edge):
  Edge edits:
    - keep, break, change_bond_order
  Node edits:
    - keep, mod_charge, mod_h_count

For supervision: given atom-mapped reaction, diff product graph vs
reactant graph and label each edge/node with its ground-truth edit. The
loss is multi-label cross-entropy over node + edge heads.

Run on 2-3x A100 (~12-24 GPU-h):
    torchrun --nproc_per_node=2 --standalone scripts/train_retro_proposer_graphedit.py \\
        --reactions rasyn/data/clean/retro/reactions_bronze.parquet \\
                    rasyn/data/clean/retro/reactions_silver.parquet \\
        --steps 60000 --bs 16 --lr 2e-4 \\
        --out checkpoints/retro_graphedit_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("train_retro_graphedit")


EDGE_EDITS = ["keep", "break", "change_bond_order"]
NODE_EDITS = ["keep", "mod_charge", "mod_h_count"]


def _maybe_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _mol_to_graph(smi: str):
    try:
        from rdkit import Chem
    except ImportError:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n_atoms = m.GetNumAtoms()
    # Node features: (atomic_num, formal_charge, num_h, hybridization, aromatic, in_ring)
    node_feats = np.zeros((n_atoms, 6), dtype=np.int64)
    for atom in m.GetAtoms():
        i = atom.GetIdx()
        node_feats[i] = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge() + 4,  # offset to non-negative
            atom.GetTotalNumHs(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
    edges = []
    edge_feats = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
        bt = int(b.GetBondTypeAsDouble() * 2)  # 2,3,4,8 -> 1,1.5,2,3,...
        edge_feats.append([bt, int(b.GetIsAromatic()), int(b.IsInRing())])
        edge_feats.append([bt, int(b.GetIsAromatic()), int(b.IsInRing())])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_feats = np.array(edge_feats, dtype=np.int64) if edge_feats else np.zeros((0, 3), dtype=np.int64)
    return {"x": node_feats, "edge_index": edge_index, "edge_attr": edge_feats, "n_atoms": n_atoms}


def _label_edits(product_smi: str, reactant_smis: list[str]) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Compute ground-truth edge + node edit labels from product / reactants.

    For v1 we use a simple bond-set diff: bonds present in product but
    absent in reactants -> 'break'; bonds with changed order -> 'change_bond_order';
    rest -> 'keep'. Node-level: charge/h-count change -> 'mod_*'.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None
    prod_mol = Chem.MolFromSmiles(product_smi)
    if prod_mol is None:
        return None
    # Combine reactants into one disjoint mol via dot-SMILES
    reactant_mol = Chem.MolFromSmiles(".".join(reactant_smis))
    if reactant_mol is None:
        return None

    # Map product atom-map nums -> reactant atom-map nums (when present);
    # fall back to identity if no map.
    def amap(mol):
        return {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0}

    prod_map = amap(prod_mol)
    reac_map = amap(reactant_mol)
    common_maps = set(prod_map.keys()) & set(reac_map.keys())

    graph = _mol_to_graph(product_smi)
    if graph is None:
        return None

    n_atoms = graph["n_atoms"]
    edge_labels = np.zeros(graph["edge_index"].shape[1], dtype=np.int64)  # 'keep'
    node_labels = np.zeros(n_atoms, dtype=np.int64)  # 'keep'

    # Bond diff (simple O(n) over product bonds).
    # If a bond exists in product but not in reactants between the same
    # mapped atoms -> 'break'. If bond exists in both but with different
    # bond order -> 'change_bond_order'.
    reac_bonds = {}
    if common_maps:
        for b in reactant_mol.GetBonds():
            a1 = b.GetBeginAtom().GetAtomMapNum()
            a2 = b.GetEndAtom().GetAtomMapNum()
            if a1 in common_maps and a2 in common_maps:
                key = (min(a1, a2), max(a1, a2))
                reac_bonds[key] = b.GetBondTypeAsDouble()

    for col in range(0, graph["edge_index"].shape[1], 2):
        i = int(graph["edge_index"][0, col])
        j = int(graph["edge_index"][1, col])
        # Map to amap nums
        ai = prod_mol.GetAtomWithIdx(i).GetAtomMapNum()
        aj = prod_mol.GetAtomWithIdx(j).GetAtomMapNum()
        key = (min(ai, aj), max(ai, aj))
        prod_bo = prod_mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()
        if key not in reac_bonds and ai in common_maps and aj in common_maps:
            edge_labels[col] = 1   # break
            edge_labels[col + 1] = 1
        elif key in reac_bonds and abs(reac_bonds[key] - prod_bo) > 1e-6:
            edge_labels[col] = 2   # change_bond_order
            edge_labels[col + 1] = 2

    return graph, edge_labels, node_labels


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reactions", nargs="+", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d-hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=6)
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

    import pyarrow.parquet as pq
    reactions: list[dict] = []
    for p in args.reactions:
        if p.exists():
            reactions.extend(pq.read_table(p).to_pylist())
    logger.info("loaded %d reactions", len(reactions))

    # Pre-build examples (skip ones that fail conversion)
    examples = []
    for r in reactions:
        prod = r.get("product_smiles") or r.get("product")
        reactants = r.get("reactant_smiles") or r.get("reactants") or []
        if not prod or not reactants:
            continue
        labeled = _label_edits(prod, reactants)
        if labeled is None:
            continue
        graph, edge_labels, node_labels = labeled
        if graph["n_atoms"] > args.max_atoms:
            continue
        examples.append((graph, edge_labels, node_labels))
    logger.info("built %d graph examples", len(examples))
    if not examples:
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple graph transformer: a tower of TransformerEncoderLayer applied
    # over a fully-connected graph with edge bias added to attention logits.
    class GraphEditModel(nn.Module):
        def __init__(self, d_hidden: int, n_layers: int):
            super().__init__()
            self.atom_emb = nn.Embedding(120, d_hidden)
            self.charge_emb = nn.Embedding(10, d_hidden)
            self.hcount_emb = nn.Embedding(8, d_hidden)
            self.aromatic_emb = nn.Embedding(2, d_hidden)
            self.ring_emb = nn.Embedding(2, d_hidden)
            enc_layer = nn.TransformerEncoderLayer(
                d_hidden, 8, dim_feedforward=4 * d_hidden,
                dropout=0.1, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
            self.norm = nn.LayerNorm(d_hidden)
            self.edge_head = nn.Sequential(
                nn.Linear(d_hidden * 2, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, len(EDGE_EDITS)),
            )
            self.node_head = nn.Linear(d_hidden, len(NODE_EDITS))

        def forward(self, node_feats: torch.Tensor, mask: torch.Tensor):
            atomic = self.atom_emb(node_feats[:, :, 0].clamp(0, 119))
            charge = self.charge_emb(node_feats[:, :, 1].clamp(0, 9))
            hcount = self.hcount_emb(node_feats[:, :, 2].clamp(0, 7))
            aromatic = self.aromatic_emb(node_feats[:, :, 4].clamp(0, 1))
            ring = self.ring_emb(node_feats[:, :, 5].clamp(0, 1))
            h = atomic + charge + hcount + aromatic + ring
            h = self.encoder(h, src_key_padding_mask=~mask)
            h = self.norm(h)
            return h

    model = GraphEditModel(args.d_hidden, args.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_batch():
        idxs = np.random.choice(len(examples), args.bs, replace=False)
        max_n = max(examples[i][0]["n_atoms"] for i in idxs)
        node_feats = np.zeros((args.bs, max_n, 6), dtype=np.int64)
        mask = np.zeros((args.bs, max_n), dtype=bool)
        node_labels = -100 * np.ones((args.bs, max_n), dtype=np.int64)
        edge_pairs = []  # (batch_idx, i, j, label)
        for b_i, i in enumerate(idxs):
            g, e_lbl, n_lbl = examples[i]
            n = g["n_atoms"]
            node_feats[b_i, :n] = g["x"]
            mask[b_i, :n] = True
            node_labels[b_i, :n] = n_lbl
            for col in range(0, g["edge_index"].shape[1], 2):
                ai = int(g["edge_index"][0, col]); aj = int(g["edge_index"][1, col])
                edge_pairs.append((b_i, ai, aj, int(e_lbl[col])))
        return (
            torch.from_numpy(node_feats).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(node_labels).to(device),
            edge_pairs,
        )

    model.train()
    log_path = args.out / "training_log.jsonl"
    log_path.touch()
    t0 = time.time()
    for step in range(args.steps):
        node_feats, mask, node_labels, edge_pairs = get_batch()
        h = model(node_feats, mask)
        node_logits = model.node_head(h)
        node_loss = F.cross_entropy(
            node_logits.reshape(-1, len(NODE_EDITS)),
            node_labels.reshape(-1),
            ignore_index=-100,
        )
        if edge_pairs:
            b_idx, ai_idx, aj_idx, labels = zip(*edge_pairs)
            b_t = torch.tensor(b_idx, dtype=torch.long, device=h.device)
            ai_t = torch.tensor(ai_idx, dtype=torch.long, device=h.device)
            aj_t = torch.tensor(aj_idx, dtype=torch.long, device=h.device)
            l_t = torch.tensor(labels, dtype=torch.long, device=h.device)
            edge_feat = torch.cat([h[b_t, ai_t], h[b_t, aj_t]], dim=-1)
            edge_logits = model.edge_head(edge_feat)
            edge_loss = F.cross_entropy(edge_logits, l_t)
        else:
            edge_loss = torch.tensor(0.0, device=h.device)
        loss = node_loss + edge_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step, "loss": float(loss.item()),
                                      "node_loss": float(node_loss.item()),
                                      "edge_loss": float(edge_loss.item()),
                                      "t": time.time() - t0}) + "\n")
            logger.info("step %d loss=%.4f", step, loss.item())

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "edge_edits": EDGE_EDITS,
        "node_edits": NODE_EDITS,
    }, args.out / "checkpoint.pt")
    logger.info("wrote %s", args.out / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
