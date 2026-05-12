"""Discrete graph diffusion for antibiotic discovery (spec §5.1, §11.1–§11.6).

DiGress-style discrete denoising diffusion over (atom_type, bond_type)
categoricals with an absorbing-state forward process.

Forward process (q):
  For each timestep t in [1..T], each node and each edge independently
  transitions to the ABSORBED state with probability β_t. Otherwise it
  retains its current category.

  Q_t = (1 - β_t) * I  +  β_t * 1_a   where 1_a is the absorbing column.

  Cumulative: ᾱ_t = ∏_{s<=t}(1-β_s). At time t, original symbol survives
  with probability ᾱ_t, otherwise absorbed.

Reverse process (p_θ):
  At each step, a graph-transformer denoiser predicts the categorical
  distribution over CLEAN node and edge types conditional on the noisy
  graph. Reverse-step distribution is then computed analytically from
  p_θ and the absorbing posterior (D3PM Eq. 3, absorbing case).

Conditioning (spec §11.3–§11.5):
  Global conditioning vector built from organism (11) + spectrum (4) +
  antibacterial label (4) + selectivity label (3) one-hots. Vector
  projected and broadcast to every node embedding. Classifier-free
  guidance: 10% of training steps drop conditioning (replaced with the
  unconditional UNCOND tokens).

Loss (spec §16):
  L_t = α_node * CE_node + α_edge * CE_edge   (uniformly sampled t)

Reference: Vignac, Krawczuk, Siraudin et al. 2023 "DiGress: Discrete
Denoising Diffusion for Graph Generation" (ICLR 2023, arXiv:2209.14734).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasyn.antibiotic.graph_io import (
    ATOM_ABSORBED,
    ATOM_PAD,
    BOND_ABSORBED,
    BOND_NONE,
    BOND_PAD,
    MAX_ATOMS,
    N_ATOM_TYPES,
    N_BOND_TYPES,
)


# ============ conditioning vocab (must match spec §7.1 + schemas.py) ============

ORGANISM_LIST = [
    "E.coli", "S.aureus", "MRSA", "K.pneumoniae", "A.baumannii",
    "P.aeruginosa", "N.gonorrhoeae", "MTB", "C.difficile", "H.pylori", "unknown",
]
SPECTRUM_LIST = [
    "broad_spectrum_or_general_antibacterial",
    "pathogen_specific",
    "target_pathogen_specific_or_selective",
    "unknown",
]
ANTIBAC_LIST = ["active", "weak", "inactive", "unknown"]
SELECT_LIST = ["selective", "non_selective", "unknown"]

N_ORG, N_SPEC, N_AB, N_SEL = len(ORGANISM_LIST), len(SPECTRUM_LIST), len(ANTIBAC_LIST), len(SELECT_LIST)
COND_DIM = N_ORG + N_SPEC + N_AB + N_SEL  # 22

ORG_IDX = {o: i for i, o in enumerate(ORGANISM_LIST)}
SPEC_IDX = {s: i for i, s in enumerate(SPECTRUM_LIST)}
AB_IDX = {a: i for i, a in enumerate(ANTIBAC_LIST)}
SEL_IDX = {s: i for i, s in enumerate(SELECT_LIST)}


def build_condition_vector(
    organism: str | None = None,
    spectrum: str | None = None,
    antibacterial: str | None = None,
    selectivity: str | None = None,
) -> torch.Tensor:
    """Return a (COND_DIM,) float tensor. None → 'unknown' bucket."""
    v = torch.zeros(COND_DIM)
    v[ORG_IDX.get(organism or "unknown", ORG_IDX["unknown"])] = 1.0
    v[N_ORG + SPEC_IDX.get(spectrum or "unknown", SPEC_IDX["unknown"])] = 1.0
    v[N_ORG + N_SPEC + AB_IDX.get(antibacterial or "unknown", AB_IDX["unknown"])] = 1.0
    v[N_ORG + N_SPEC + N_AB + SEL_IDX.get(selectivity or "unknown", SEL_IDX["unknown"])] = 1.0
    return v


# ============ noise schedule ============

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021), adapted to absorbing diffusion."""
    t = torch.arange(T + 1, dtype=torch.float32) / T
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=1e-4, max=0.999)


class AbsorbingDiffusion:
    """Schedule + forward q-sample for absorbing-state discrete diffusion."""

    def __init__(self, T: int = 500, device: torch.device | str = "cpu"):
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alpha_bar = torch.cumprod(1.0 - self.betas, dim=0)  # ᾱ_t

    def q_sample(
        self,
        node_clean: torch.LongTensor,
        edge_clean: torch.LongTensor,
        t: torch.LongTensor,
        node_mask: torch.BoolTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Sample (x_t_node, x_t_edge) from q(x_t | x_0). Off-mask positions
        retain PAD; absorbed positions get ABSORBED token."""
        B, N = node_clean.shape
        ab_t = self.alpha_bar[t].view(B, 1)  # (B,1)
        # Nodes: with probability ᾱ_t keep clean, else ABSORBED.
        keep_node = torch.rand(B, N, device=node_clean.device) < ab_t
        node_t = torch.where(keep_node, node_clean, torch.full_like(node_clean, ATOM_ABSORBED))
        # Off-mask positions stay PAD regardless of noise.
        node_t = torch.where(node_mask, node_t, torch.full_like(node_t, ATOM_PAD))

        # Edges: same, applied symmetrically. Sample upper-triangular noise then mirror.
        ab_te = ab_t.view(B, 1, 1)
        keep_edge_upper = torch.rand(B, N, N, device=edge_clean.device) < ab_te
        keep_edge_upper = torch.triu(keep_edge_upper, diagonal=1)
        keep_edge = keep_edge_upper | keep_edge_upper.transpose(-1, -2)
        # Diagonal: always preserved (not modeled).
        diag = torch.eye(N, dtype=torch.bool, device=edge_clean.device).unsqueeze(0).expand(B, -1, -1)
        keep_edge = keep_edge | diag
        edge_t = torch.where(keep_edge, edge_clean, torch.full_like(edge_clean, BOND_ABSORBED))
        # Off-mask positions stay PAD.
        edge_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        edge_t = torch.where(edge_mask, edge_t, torch.full_like(edge_t, BOND_PAD))
        return node_t, edge_t


# ============ denoiser: graph transformer ============

class GraphTransformerLayer(nn.Module):
    """Self-attention over nodes with edge-feature bias added to attention logits.

    Inspired by the DiGress denoiser and Graph Transformer (Dwivedi & Bresson).
    Each layer:
      - linear projects nodes and edges
      - attention(Q,K) + edge_bias → attention weights
      - aggregate values per node
      - simultaneously updates edges from outer product of node updates
    """

    def __init__(self, d_node: int = 256, d_edge: int = 64, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.n_heads = n_heads
        self.dk = d_node // n_heads

        self.q = nn.Linear(d_node, d_node)
        self.k = nn.Linear(d_node, d_node)
        self.v = nn.Linear(d_node, d_node)
        self.e2bias = nn.Linear(d_edge, n_heads)
        self.o_node = nn.Linear(d_node, d_node)

        self.ff_node = nn.Sequential(
            nn.Linear(d_node, d_node * 4), nn.GELU(), nn.Linear(d_node * 4, d_node),
        )
        self.ln_node1 = nn.LayerNorm(d_node)
        self.ln_node2 = nn.LayerNorm(d_node)

        # Edge update from concatenated [x_i, x_j, e_ij]
        self.edge_update = nn.Sequential(
            nn.Linear(2 * d_node + d_edge, d_edge * 2), nn.GELU(), nn.Linear(d_edge * 2, d_edge),
        )
        self.ln_edge = nn.LayerNorm(d_edge)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,       # (B, N, d_node)
        e: torch.Tensor,       # (B, N, N, d_edge)
        mask: torch.Tensor,    # (B, N) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        H = self.n_heads
        Dk = self.dk

        q = self.q(x).view(B, N, H, Dk).transpose(1, 2)  # (B,H,N,Dk)
        k = self.k(x).view(B, N, H, Dk).transpose(1, 2)
        v = self.v(x).view(B, N, H, Dk).transpose(1, 2)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(Dk)  # (B,H,N,N)
        attn = attn + self.e2bias(e).permute(0, 3, 1, 2)               # (B,H,N,N)
        attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).reshape(B, N, -1)
        x_upd = self.ln_node1(x + self.dropout(self.o_node(out)))
        x_upd = self.ln_node2(x_upd + self.dropout(self.ff_node(x_upd)))

        # Edge update
        xi = x_upd.unsqueeze(2).expand(-1, -1, N, -1)
        xj = x_upd.unsqueeze(1).expand(-1, N, -1, -1)
        e_in = torch.cat([xi, xj, e], dim=-1)
        e_upd = self.ln_edge(e + self.dropout(self.edge_update(e_in)))
        return x_upd, e_upd


class GraphDenoiser(nn.Module):
    """Predict clean (node_logits, edge_logits) from noisy graph + timestep + condition.

    Inputs:
      node_t: (B, N) long — noisy node tokens
      edge_t: (B, N, N) long — noisy edge tokens
      t: (B,) long — timestep (1..T)
      cond: (B, COND_DIM) float — global conditioning (zero vector for unconditional)
      node_mask: (B, N) bool

    Outputs:
      node_logits: (B, N, N_ATOM_TYPES_OUT) — predicted CLEAN node distribution
      edge_logits: (B, N, N, N_BOND_TYPES_OUT) — predicted CLEAN edge distribution

    N_*_TYPES_OUT excludes the ABSORBED and PAD tokens (we don't predict them
    as a clean state; padding is determined by the static node_mask).
    """

    N_ATOM_OUT = N_ATOM_TYPES - 2   # exclude ABSORBED + PAD
    N_BOND_OUT = N_BOND_TYPES - 2   # exclude ABSORBED + PAD

    def __init__(
        self,
        d_node: int = 256,
        d_edge: int = 64,
        n_heads: int = 8,
        n_layers: int = 6,
        T: int = 500,
        cond_dim: int = COND_DIM,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.T = T

        self.node_emb = nn.Embedding(N_ATOM_TYPES, d_node)
        self.edge_emb = nn.Embedding(N_BOND_TYPES, d_edge)
        self.t_emb = nn.Embedding(T + 1, d_node)
        self.cond_proj = nn.Linear(cond_dim, d_node)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_node, d_edge, n_heads) for _ in range(n_layers)
        ])

        self.node_head = nn.Linear(d_node, self.N_ATOM_OUT)
        self.edge_head = nn.Linear(d_edge, self.N_BOND_OUT)

    def forward(
        self,
        node_t: torch.LongTensor,
        edge_t: torch.LongTensor,
        t: torch.LongTensor,
        cond: torch.Tensor,
        node_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = node_t.shape
        x = self.node_emb(node_t)                                          # (B,N,d_node)
        x = x + self.t_emb(t).unsqueeze(1) + self.cond_proj(cond).unsqueeze(1)
        e = self.edge_emb(edge_t)                                          # (B,N,N,d_edge)
        for layer in self.layers:
            x, e = layer(x, e, node_mask)
        return self.node_head(x), self.edge_head(e)


# ============ loss ============

def diffusion_loss(
    denoiser: GraphDenoiser,
    diffusion: AbsorbingDiffusion,
    node_clean: torch.LongTensor,
    edge_clean: torch.LongTensor,
    node_mask: torch.BoolTensor,
    cond: torch.Tensor,
    *,
    alpha_node: float = 1.0,
    alpha_edge: float = 0.5,
    cfg_drop_prob: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """L_t = α_node * CE_node + α_edge * CE_edge with classifier-free guidance dropout.

    Loss is computed only on positions that were ACTUALLY corrupted by the
    forward process (D3PM hybrid loss restricted to the absorbing set). This
    follows DiGress §3.3 — predicting clean values at non-absorbed positions
    is trivial (they didn't change) so we down-weight them by masking out.
    """
    B, N = node_clean.shape
    device = node_clean.device

    # Classifier-free guidance: drop cond on a fraction of the batch.
    if cfg_drop_prob > 0:
        drop = torch.rand(B, 1, device=device) < cfg_drop_prob
        cond = torch.where(drop, torch.zeros_like(cond), cond)

    t = torch.randint(1, diffusion.T + 1, (B,), device=device)
    node_t, edge_t = diffusion.q_sample(node_clean, edge_clean, t - 1, node_mask)

    node_logits, edge_logits = denoiser(node_t, edge_t, t, cond, node_mask)

    # Restrict node CE to nodes that were absorbed AND on-mask AND clean is a predictable type.
    node_absorbed = (node_t == ATOM_ABSORBED) & node_mask
    # Unwrap DDP if present (DDP doesn't expose class attrs).
    _d = denoiser.module if hasattr(denoiser, "module") else denoiser
    valid_clean = node_clean < _d.N_ATOM_OUT  # predictable clean tokens
    node_target_mask = node_absorbed & valid_clean
    if node_target_mask.any():
        n_logits = node_logits[node_target_mask]
        n_tgt = node_clean[node_target_mask].long()
        l_node = F.cross_entropy(n_logits, n_tgt)
    else:
        l_node = node_logits.new_zeros(())

    # Edge CE on upper-triangular absorbed edges.
    edge_mask_2d = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).expand(B, -1, -1)
    edge_absorbed = (edge_t == BOND_ABSORBED) & edge_mask_2d & upper
    edge_valid = edge_clean < _d.N_BOND_OUT
    edge_target_mask = edge_absorbed & edge_valid
    if edge_target_mask.any():
        e_logits = edge_logits[edge_target_mask]
        e_tgt = edge_clean[edge_target_mask].long()
        l_edge = F.cross_entropy(e_logits, e_tgt)
    else:
        l_edge = edge_logits.new_zeros(())

    loss = alpha_node * l_node + alpha_edge * l_edge
    stats = {
        "loss": float(loss.item()),
        "l_node": float(l_node.item()) if torch.is_tensor(l_node) else float(l_node),
        "l_edge": float(l_edge.item()) if torch.is_tensor(l_edge) else float(l_edge),
        "n_node_targets": int(node_target_mask.sum().item()),
        "n_edge_targets": int(edge_target_mask.sum().item()),
    }
    return loss, stats


# ============ sampling: guided reverse process ============

@torch.no_grad()
def sample_graphs(
    denoiser: GraphDenoiser,
    diffusion: AbsorbingDiffusion,
    *,
    cond: torch.Tensor,
    n_atoms_per_sample: torch.LongTensor,
    guidance_scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]:
    """Generate B graphs with the reverse process (§11.6 guided sampling).

    Algorithm:
      1. Initialize fully-absorbed graphs (all nodes = ABSORBED, all edges = ABSORBED).
      2. For t = T..1:
           predict p_θ(x_0 | x_t, cond) and p_θ(x_0 | x_t, ∅).
           Combine via classifier-free guidance:
              ε̂ = (1 + w) * cond_logits − w * uncond_logits
           Sample a clean prediction x̂_0 from softmax(ε̂).
           Set x_{t-1} = x̂_0 with probability (α_{t-1} - α_t)/(1 - α_t),
                         else keep absorbed.
      3. Return final (nodes, edges, mask).

    n_atoms_per_sample lets the caller request graph size per sample (used by
    Channel E to grow molecules of similar size to a seed fragment).
    """
    B = cond.shape[0]
    N = MAX_ATOMS
    node_t = torch.full((B, N), ATOM_ABSORBED, dtype=torch.long, device=device)
    edge_t = torch.full((B, N, N), BOND_ABSORBED, dtype=torch.long, device=device)
    node_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    for b in range(B):
        node_mask[b, : int(n_atoms_per_sample[b])] = True
    # Off-mask positions are PAD
    node_t = torch.where(node_mask, node_t, torch.full_like(node_t, ATOM_PAD))
    edge_mask_2d = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    edge_t = torch.where(edge_mask_2d, edge_t, torch.full_like(edge_t, BOND_PAD))

    null_cond = torch.zeros_like(cond)
    for step in range(diffusion.T, 0, -1):
        tt = torch.full((B,), step, dtype=torch.long, device=device)
        cond_node_logits, cond_edge_logits = denoiser(node_t, edge_t, tt, cond, node_mask)
        if guidance_scale != 0.0:
            null_node_logits, null_edge_logits = denoiser(node_t, edge_t, tt, null_cond, node_mask)
            node_logits = (1.0 + guidance_scale) * cond_node_logits - guidance_scale * null_node_logits
            edge_logits = (1.0 + guidance_scale) * cond_edge_logits - guidance_scale * null_edge_logits
        else:
            node_logits, edge_logits = cond_node_logits, cond_edge_logits

        # Categorical sample of clean prediction at every position.
        node_probs = F.softmax(node_logits, dim=-1)
        edge_probs = F.softmax(edge_logits, dim=-1)
        flat_n = node_probs.reshape(-1, node_probs.size(-1))
        flat_e = edge_probs.reshape(-1, edge_probs.size(-1))
        node_pred = torch.multinomial(flat_n, 1).view(B, N)
        edge_pred = torch.multinomial(flat_e, 1).view(B, N, N)
        # Symmetrize edges
        edge_pred = torch.triu(edge_pred, diagonal=1)
        edge_pred = edge_pred + edge_pred.transpose(-1, -2)

        # Reverse step: with probability r reveal predicted clean, else stay absorbed.
        if step > 1:
            r = (diffusion.alpha_bar[step - 2] - diffusion.alpha_bar[step - 1]) / (
                1.0 - diffusion.alpha_bar[step - 1] + 1e-8
            )
        else:
            r = torch.tensor(1.0, device=device)  # final reveal
        reveal_node = (torch.rand(B, N, device=device) < r) & (node_t == ATOM_ABSORBED) & node_mask
        node_t = torch.where(reveal_node, node_pred, node_t)
        reveal_edge_upper = (torch.rand(B, N, N, device=device) < r) & (edge_t == BOND_ABSORBED) & edge_mask_2d
        reveal_edge_upper = torch.triu(reveal_edge_upper, diagonal=1)
        reveal_edge = reveal_edge_upper | reveal_edge_upper.transpose(-1, -2)
        edge_t = torch.where(reveal_edge, edge_pred, edge_t)

    return node_t, edge_t, node_mask
