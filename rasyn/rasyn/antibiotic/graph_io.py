"""SMILES ↔ molecular-graph tensor IO for discrete graph diffusion (spec §5.1).

Per spec §5.1 the diffusion model denoises 2D molecular graphs with
categorical node and edge types. This module provides the deterministic
RDKit-based round-trip between canonical SMILES and the (node_types,
edge_types, node_mask) tensors the diffusion model operates on.

Vocabularies are CLOSED and ordered. The trailing entry `ABSORBED` is the
absorbing state used by the D3PM forward process; the trailing entry
`PAD` masks padding atoms beyond the molecule's true node count.

Hard caps:
  MAX_ATOMS = 40 — covers ~99% of antibacterial drugs (rifampin and a
                    few macrocycles overflow and get silently dropped
                    during dataset prep — counted in graph_io stats).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------- vocabularies ----------

# Top-12 atoms covering >99% of antibacterial chemistry + UNK + ABSORBED + PAD.
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "*", "UNK", "ABSORBED", "PAD"]
ATOM_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
N_ATOM_TYPES = len(ATOM_TYPES)
ATOM_PAD = ATOM_IDX["PAD"]
ATOM_ABSORBED = ATOM_IDX["ABSORBED"]
ATOM_UNK = ATOM_IDX["UNK"]

# RDKit BondType -> token. NONE means "no edge between atoms i and j".
BOND_TYPES = ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "ABSORBED", "PAD"]
BOND_IDX = {b: i for i, b in enumerate(BOND_TYPES)}
N_BOND_TYPES = len(BOND_TYPES)
BOND_NONE = BOND_IDX["NONE"]
BOND_PAD = BOND_IDX["PAD"]
BOND_ABSORBED = BOND_IDX["ABSORBED"]

MAX_ATOMS = 40


@dataclass(frozen=True)
class MolGraph:
    """Padded molecular graph tensors.

    node_types: int8[MAX_ATOMS] — ATOM_IDX values. Positions >= n_atoms are ATOM_PAD.
    edge_types: int8[MAX_ATOMS, MAX_ATOMS] — symmetric BOND_IDX values.
                Diagonal and out-of-mol positions are BOND_PAD.
    node_mask:  bool[MAX_ATOMS] — True for the n_atoms real atoms, False for padding.
    n_atoms:    int — actual atom count (used for stats / loss masking).
    """
    node_types: np.ndarray
    edge_types: np.ndarray
    node_mask: np.ndarray
    n_atoms: int


def _rdkit_bond_token(bond_obj) -> int:
    from rdkit.Chem import BondType
    bt = bond_obj.GetBondType()
    if bt == BondType.SINGLE:
        return BOND_IDX["SINGLE"]
    if bt == BondType.DOUBLE:
        return BOND_IDX["DOUBLE"]
    if bt == BondType.TRIPLE:
        return BOND_IDX["TRIPLE"]
    if bt == BondType.AROMATIC:
        return BOND_IDX["AROMATIC"]
    return BOND_IDX["SINGLE"]  # OTHER coerced to SINGLE


def smiles_to_graph(smi: str) -> MolGraph | None:
    """Canonical SMILES → padded MolGraph. Returns None if invalid or too large."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n = m.GetNumAtoms()
    if n == 0 or n > MAX_ATOMS:
        return None
    nodes = np.full(MAX_ATOMS, ATOM_PAD, dtype=np.int8)
    edges = np.full((MAX_ATOMS, MAX_ATOMS), BOND_PAD, dtype=np.int8)
    for i in range(n):
        sym = m.GetAtomWithIdx(i).GetSymbol()
        nodes[i] = ATOM_IDX.get(sym, ATOM_UNK)
    # Initialize within-molecule edges to NONE.
    edges[:n, :n] = BOND_NONE
    np.fill_diagonal(edges[:n, :n], BOND_PAD)  # self-loops not modeled
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = _rdkit_bond_token(b)
        edges[i, j] = t
        edges[j, i] = t
    mask = np.zeros(MAX_ATOMS, dtype=bool)
    mask[:n] = True
    return MolGraph(node_types=nodes, edge_types=edges, node_mask=mask, n_atoms=n)


def graph_to_rdkit_mol(node_types: np.ndarray, edge_types: np.ndarray, node_mask: np.ndarray):
    """(node_types, edge_types, node_mask) → RDKit RWMol with explicit valence sanitization.

    Returns the RDKit mol on success or None if construction fails.
    Used by the diffusion sampler to convert denoised graphs back to SMILES.
    """
    from rdkit import Chem
    from rdkit.Chem import BondType

    bond_lookup = {
        BOND_IDX["SINGLE"]:    BondType.SINGLE,
        BOND_IDX["DOUBLE"]:    BondType.DOUBLE,
        BOND_IDX["TRIPLE"]:    BondType.TRIPLE,
        BOND_IDX["AROMATIC"]:  BondType.AROMATIC,
    }
    mol = Chem.RWMol()
    idx_map = {}
    for i in range(len(node_mask)):
        if not node_mask[i]:
            continue
        t = int(node_types[i])
        if t in (ATOM_PAD, ATOM_ABSORBED):
            continue
        sym = ATOM_TYPES[t]
        if sym in ("UNK", "PAD", "ABSORBED"):
            continue
        a = Chem.Atom(sym if sym != "*" else 0)
        idx_map[i] = mol.AddAtom(a)
    n = len(idx_map)
    if n < 2:
        return None
    keys = list(idx_map.keys())
    for ii in range(len(keys)):
        for jj in range(ii + 1, len(keys)):
            i, j = keys[ii], keys[jj]
            t = int(edge_types[i, j])
            bt = bond_lookup.get(t)
            if bt is not None:
                mol.AddBond(idx_map[i], idx_map[j], bt)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def graph_to_smiles(node_types: np.ndarray, edge_types: np.ndarray, node_mask: np.ndarray) -> str | None:
    from rdkit import Chem
    mol = graph_to_rdkit_mol(node_types, edge_types, node_mask)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ---------- batch helpers ----------

def stack_graphs(graphs: list[MolGraph]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a list of MolGraph into (B,N), (B,N,N), (B,N) arrays."""
    if not graphs:
        return (
            np.zeros((0, MAX_ATOMS), dtype=np.int8),
            np.zeros((0, MAX_ATOMS, MAX_ATOMS), dtype=np.int8),
            np.zeros((0, MAX_ATOMS), dtype=bool),
        )
    nodes = np.stack([g.node_types for g in graphs])
    edges = np.stack([g.edge_types for g in graphs])
    masks = np.stack([g.node_mask for g in graphs])
    return nodes, edges, masks
