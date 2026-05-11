"""Train the organism-conditioned antibiotic multi-head ranker (spec §12).

Architecture:
  Stage-1 200M backbone encoder (frozen or finetune) on candidate SMILES
  -> mean-pooled candidate embedding
  -> concat with [organism one-hot, gram-type one-hot, spectrum-goal one-hot]
  -> MLP trunk
  -> 12 heads:
       antibacterial_score          (regression / binary)
       organism_specific_score      (regression)
       selectivity_score            (regression)
       cytotoxicity_risk            (binary)
       hemolysis_risk               (binary)
       artifact_risk                (binary)
       known_antibiotic_similarity_penalty  (regression)
       training_active_similarity_penalty   (regression)
       novelty_score                (regression)
       synthesizability_score       (regression)
       uncertainty_score            (regression)
       failure_mode_probabilities   (5-way softmax)

Training data: antibiotic_ranking_tasks.parquet (Phase ABX-3 output).
Loss: weighted sum of BCE / MSE / cross-entropy.

Run (8x A100 DDP):
    torchrun --nproc_per_node=8 --standalone scripts/train_abx_ranker.py \\
        --pretrain rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --tasks rasyn/data/clean/antibiotic/antibiotic_ranking_tasks.parquet \\
        --facts rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --steps 5000 --bs 32 --seed 42 \\
        --out rasyn/data/clean/abx_ranker_seed42
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


# Vocabularies (one-hot encoded conditioning input)
ORGANISM_LIST = [
    "E.coli", "S.aureus", "MRSA", "K.pneumoniae", "A.baumannii", "P.aeruginosa",
    "N.gonorrhoeae", "M.tuberculosis", "C.difficile", "broad_spectrum", "unknown",
]
GRAM_LIST = ["Gram-positive", "Gram-negative", "atypical", "unknown"]
SPECTRUM_LIST = [
    "broad_spectrum_or_general_antibacterial", "pathogen_specific",
    "target_pathogen_specific_or_selective", "unknown",
]
FAILURE_MODES = ["inactive", "cytotoxic", "artifact", "organism_mismatch", "known_control_only"]

CONDITION_DIM = len(ORGANISM_LIST) + len(GRAM_LIST) + len(SPECTRUM_LIST)  # = 11+4+4 = 19


def condition_vector(organism: str, gram: str, spectrum: str) -> np.ndarray:
    v = np.zeros(CONDITION_DIM, dtype=np.float32)
    v[ORGANISM_LIST.index(organism) if organism in ORGANISM_LIST else ORGANISM_LIST.index("unknown")] = 1.0
    offset = len(ORGANISM_LIST)
    v[offset + (GRAM_LIST.index(gram) if gram in GRAM_LIST else GRAM_LIST.index("unknown"))] = 1.0
    offset += len(GRAM_LIST)
    v[offset + (SPECTRUM_LIST.index(spectrum) if spectrum in SPECTRUM_LIST else SPECTRUM_LIST.index("unknown"))] = 1.0
    return v


# ----------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------

def tokenize(smi: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    UNK = 1
    ids = [VOCAB.get(c, UNK) for c in smi[:max_len]]
    n = len(ids)
    ids = ids + [PAD] * (max_len - n)
    attn = np.zeros(max_len, dtype=bool)
    attn[:n] = True
    return np.asarray(ids, dtype=np.int64), attn


# ----------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------

class ABXRankerDataset(Dataset):
    """Per-pair candidate dataset constructed from ranking_tasks + facts.

    Each item: (candidate_smiles, organism, gram, spectrum, labels...).
    Pairs are sampled randomly across ranking tasks; supports listwise sampling
    in a later version.
    """

    def __init__(self, rows: list[dict], max_len: int = 128):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids, mask = tokenize(r["candidate_smiles"] or "C", self.max_len)
        cond = condition_vector(r.get("organism", "unknown"),
                                 r.get("gram_type", "unknown"),
                                 r.get("spectrum_goal", "unknown"))
        # Labels
        ab_label = 1.0 if r.get("activity_label") == "active" else 0.0
        cytotox_label = 1.0 if r.get("is_cytotoxic", False) else 0.0
        # Failure mode
        if ab_label == 0.0:
            fm = "inactive"
        elif cytotox_label == 1.0:
            fm = "cytotoxic"
        else:
            fm = "inactive"  # default fallback
        # Map to discovery label
        disc = r.get("discovery_label") or "unknown"
        if disc == "active_toxic":
            fm = "cytotoxic"
        fm_id = FAILURE_MODES.index(fm) if fm in FAILURE_MODES else FAILURE_MODES.index("inactive")
        return (
            torch.from_numpy(ids),
            torch.from_numpy(mask),
            torch.from_numpy(cond),
            torch.tensor(ab_label, dtype=torch.float32),
            torch.tensor(cytotox_label, dtype=torch.float32),
            torch.tensor(fm_id, dtype=torch.long),
        )


# ----------------------------------------------------------------
# Model
# ----------------------------------------------------------------

class ABXMultiHeadRanker(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 1024, n_heads: int = 16,
                 n_layers: int = 16, max_len: int = 128, cond_dim: int = CONDITION_DIM):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.LayerNorm(d_model), nn.Dropout(0.1),
        )
        # Heads
        self.head_antibacterial = nn.Linear(d_model, 1)
        self.head_organism_specific = nn.Linear(d_model, 1)
        self.head_selectivity = nn.Linear(d_model, 1)
        self.head_cytotox = nn.Linear(d_model, 1)
        self.head_hemolysis = nn.Linear(d_model, 1)
        self.head_artifact = nn.Linear(d_model, 1)
        self.head_known_pen = nn.Linear(d_model, 1)
        self.head_train_pen = nn.Linear(d_model, 1)
        self.head_novelty = nn.Linear(d_model, 1)
        self.head_synth = nn.Linear(d_model, 1)
        self.head_uncertainty = nn.Linear(d_model, 1)
        self.head_failure_modes = nn.Linear(d_model, len(FAILURE_MODES))
        self.max_len = max_len

    def encode(self, ids, mask):
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand_as(ids)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, ids, mask, cond):
        emb = self.encode(ids, mask)
        cond_h = self.cond_proj(cond)
        h = self.trunk(torch.cat([emb, cond_h], dim=-1))
        return {
            "antibacterial": torch.sigmoid(self.head_antibacterial(h)).squeeze(-1),
            "organism_specific": self.head_organism_specific(h).squeeze(-1),
            "selectivity": self.head_selectivity(h).squeeze(-1),
            "cytotox": torch.sigmoid(self.head_cytotox(h)).squeeze(-1),
            "hemolysis": torch.sigmoid(self.head_hemolysis(h)).squeeze(-1),
            "artifact": torch.sigmoid(self.head_artifact(h)).squeeze(-1),
            "known_pen": self.head_known_pen(h).squeeze(-1),
            "train_pen": self.head_train_pen(h).squeeze(-1),
            "novelty": self.head_novelty(h).squeeze(-1),
            "synth": self.head_synth(h).squeeze(-1),
            "uncertainty": self.head_uncertainty(h).squeeze(-1),
            "failure_modes": self.head_failure_modes(h),
        }


def load_pretrain(model: ABXMultiHeadRanker, ckpt_path: Path, log=None) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    msd = model.state_dict()
    n = 0
    for k, v in sd.items():
        if k.startswith("lm_head"):
            continue
        if k in msd and msd[k].shape == v.shape:
            msd[k] = v
            n += 1
    model.load_state_dict(msd)
    if log:
        log(f"Loaded {n} backbone tensors from {ckpt_path}")
    return n


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=2))
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_per_pair_rows(tasks_df: pd.DataFrame, facts_df: pd.DataFrame) -> list[dict]:
    """Flatten ranking tasks into per-candidate rows."""
    smi_by_ik = {}
    if "inchi_key" in facts_df.columns and "canonical_smiles" in facts_df.columns:
        for _, r in facts_df.dropna(subset=["inchi_key", "canonical_smiles"]).iterrows():
            smi_by_ik[r["inchi_key"]] = r["canonical_smiles"]

    rows = []
    for _, task in tasks_df.iterrows():
        org = task.get("organism", "unknown")
        # Gram + spectrum default per organism
        gram = "Gram-negative" if org in ("E.coli", "K.pneumoniae", "A.baumannii", "P.aeruginosa", "N.gonorrhoeae") else (
            "Gram-positive" if org in ("S.aureus", "MRSA", "M.tuberculosis", "C.difficile") else "unknown"
        )
        spectrum = "broad_spectrum_or_general_antibacterial" if org == "broad_spectrum" else "pathogen_specific"
        def _as_list(v):
            if v is None: return []
            try:
                return list(v)
            except TypeError:
                return []
        cand_iks = _as_list(task.get("candidate_inchi_keys"))
        ab_labels = _as_list(task.get("antibacterial_labels"))
        disc_labels = _as_list(task.get("discovery_labels"))
        for i, ik in enumerate(cand_iks):
            if ik is None or (isinstance(ik, float) and ik != ik):  # NaN
                continue
            smi = smi_by_ik.get(ik)
            if not smi:
                continue
            rows.append({
                "candidate_smiles": smi,
                "organism": org,
                "gram_type": gram,
                "spectrum_goal": spectrum,
                "activity_label": ab_labels[i] if i < len(ab_labels) else "unknown",
                "discovery_label": disc_labels[i] if i < len(disc_labels) else "unknown",
                "is_cytotoxic": (disc_labels[i] == "active_toxic") if i < len(disc_labels) else False,
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=500)
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
        log(f"Loading tasks {args.tasks}")
    tasks_df = pd.read_parquet(args.tasks)
    facts_df = pd.read_parquet(args.facts)
    rows = build_per_pair_rows(tasks_df, facts_df)
    log(f"  per-pair rows: {len(rows):,}")
    if len(rows) < 1000:
        raise SystemExit(f"Too few training rows ({len(rows)}); aborting.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rows))
    val_n = max(512, int(len(rows) * args.val_frac))
    val_idx = set(perm[:val_n].tolist())
    tr_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val_rows = [rows[i] for i in val_idx]
    tr_ds = ABXRankerDataset(tr_rows, max_len=args.max_len)
    val_ds = ABXRankerDataset(val_rows, max_len=args.max_len)
    sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, shuffle=(sampler is None),
                       num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs * 2, num_workers=2, pin_memory=True)
    log(f"World={world_size} | tr={len(tr_ds):,} val={len(val_ds):,}")

    ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    model = ABXMultiHeadRanker(
        VOCAB_SIZE,
        d_model=cargs.get("d_model", 1024),
        n_heads=cargs.get("n_heads", 16),
        n_layers=cargs.get("n_layers", 16),
        max_len=cargs.get("max_len", args.max_len),
    ).to(device)
    if is_main:
        load_pretrain(model, args.pretrain, log)
    if dist.is_initialized():
        dist.barrier()
    if not is_main:
        load_pretrain(model, args.pretrain, None)
    if world_size > 1:
        # find_unused_parameters=True because 12 heads but only 3 contribute to loss
        # in this v1 (antibacterial, cytotox, failure_modes). Other heads (selectivity,
        # hemolysis, artifact, novelty, etc.) have no labels yet.
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
        )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        n = 0; ab_correct = 0; cyto_correct = 0; fm_correct = 0
        for batch in val_dl:
            ids, mask, cond, ab, cyto, fm = [b.to(device, non_blocking=True) for b in batch]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, mask, cond)
            ab_correct += ((out["antibacterial"].float() >= 0.5) == (ab >= 0.5)).sum().item()
            cyto_correct += ((out["cytotox"].float() >= 0.5) == (cyto >= 0.5)).sum().item()
            fm_correct += (out["failure_modes"].float().argmax(-1) == fm).sum().item()
            n += ab.size(0)
        model.train()
        return {
            "antibacterial_acc": ab_correct / max(1, n),
            "cytotox_acc": cyto_correct / max(1, n),
            "failure_mode_acc": fm_correct / max(1, n),
            "n_val": n,
        }

    if is_main:
        log(f"Starting ABX ranker training: {args.steps} steps")
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
            ids, mask, cond, ab, cyto, fm = [b.to(device, non_blocking=True) for b in batch]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, mask, cond)
            ab_loss = F.binary_cross_entropy(out["antibacterial"].float(), ab)
            cyto_loss = F.binary_cross_entropy(out["cytotox"].float(), cyto)
            fm_loss = F.cross_entropy(out["failure_modes"].float(), fm)
            loss = ab_loss + 0.5 * cyto_loss + 0.3 * fm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 50 == 0 and is_main:
                log(f"step {step}/{args.steps} loss={loss.item():.4f} ab={ab_loss.item():.3f} cyto={cyto_loss.item():.3f} fm={fm_loss.item():.3f}",
                    step=step, loss=float(loss.item()))
            if step % args.val_every == 0 and is_main:
                metrics = evaluate()
                log(f"step {step} val: {metrics}", val=metrics)
                (args.out / "per_class_metrics.json").write_text(json.dumps(metrics, indent=2))
            if step % args.ckpt_every == 0 and is_main:
                _save(model, args, step)
                log(f"Saved checkpoint at step {step}")
            if step >= args.steps:
                break
        epoch += 1

    if is_main:
        metrics = evaluate()
        (args.out / "per_class_metrics.json").write_text(json.dumps(metrics, indent=2))
        _save(model, args, step)
        log("FINAL", val=metrics, total_seconds=time.time() - t0)
    cleanup_distributed()


def _save(model, args, step):
    sd = (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model).state_dict()
    torch.save({
        "step": step, "model": sd, "args": vars(args),
        "organism_list": ORGANISM_LIST, "gram_list": GRAM_LIST,
        "spectrum_list": SPECTRUM_LIST, "failure_modes": FAILURE_MODES,
        "vocab_size": VOCAB_SIZE, "cond_dim": CONDITION_DIM,
    }, args.out / "checkpoint.pt")


if __name__ == "__main__":
    sys.exit(main())
