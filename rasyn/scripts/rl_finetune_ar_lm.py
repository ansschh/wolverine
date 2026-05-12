"""RL fine-tune an autoregressive SMILES LM toward ranker reward.

Algorithm: REINFORCE with a learned baseline (running mean of rewards).
At each iteration:
  1. Sample N SMILES from the current LM (temperature 1.0 + top-p 0.95).
  2. RDKit-validate each. Invalid ones get reward = -1 (penalty).
  3. For valid ones, compute reward via the v4 ranker:
       r = ab - 0.5*cyto - 0.3*art
       + λ_novelty * (1 - max_tan_to_training_actives)
       - λ_memorization * (1 if any_train_active_at_Tan>=0.95 else 0)
  4. Subtract running-mean baseline → advantage.
  5. Loss: -E[advantage * log p(SMILES | θ)] → grad descent on θ.

Per Stokes-style design: we keep a KL-divergence anchor to the original
LM (a reference copy frozen) to prevent the policy from collapsing to a
single high-reward molecule.

Run on 5x A100:
    torchrun --nproc_per_node=5 --standalone scripts/rl_finetune_ar_lm.py \\
        --base-lm rasyn/data/clean/smiles_ar_lm_200m/checkpoint.pt \\
        --ranker rasyn/data/clean/abx_ranker_v4_seed42/checkpoint.pt \\
        --facts rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --organism E.coli --gram Gram-negative \\
        --spectrum broad_spectrum_or_general_antibacterial \\
        --iterations 500 --samples-per-iter 64 \\
        --kl-coef 0.1 \\
        --out rasyn/data/clean/smiles_ar_lm_rl_ecoli
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

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from train_smiles_ar_lm import (  # noqa: E402
    ARSMILESLM, VOCAB, VOCAB_SIZE, PAD, BOS, EOS, UNK, tokenize_with_bos_eos,
)
from train_abx_ranker_v4 import ABXMultiHeadRankerV4  # noqa: E402
from train_abx_ranker import condition_vector, tokenize as ranker_tok  # noqa: E402
# The ranker uses h200_smiles_lm_pretrain VOCAB (different from the AR LM vocab).
# Import that vocab size separately so we can construct the ranker with matching dims.
from h200_smiles_lm_pretrain import VOCAB_SIZE as RANKER_VOCAB_SIZE  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=dt.timedelta(hours=2))
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_base_lm(ckpt_path: Path, device) -> ARSMILESLM:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    model = ARSMILESLM(
        d_model=cargs.get("d_model", 1024),
        n_heads=cargs.get("n_heads", 16),
        n_layers=cargs.get("n_layers", 16),
        max_len=cargs.get("max_len", 128),
    ).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    return model


def load_ranker(ckpt_path: Path, device) -> ABXMultiHeadRankerV4:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ckpt.get("args", {})
    # Ranker uses h200_smiles_lm_pretrain vocab (43 tokens), NOT the AR LM vocab.
    model = ABXMultiHeadRankerV4(
        RANKER_VOCAB_SIZE, d_model=1024, n_heads=16, n_layers=16,
        max_len=cargs.get("max_len", 128),
    ).to(device)
    sd = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def validate_smiles(smi: str) -> str | None:
    """RDKit-validate; return canonical SMILES if valid, None otherwise."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m is None or m.GetNumAtoms() < 3:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def _morgan_fp(smi: str, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=n_bits) if m else None


def compute_rewards(smiles_list: list[str], ranker: ABXMultiHeadRankerV4,
                     organism: str, gram: str, spectrum: str,
                     active_fps, device, max_len: int = 128, bs: int = 32) -> np.ndarray:
    """Score N SMILES with the ranker; combine into reward + novelty bonus.
    Invalid SMILES get -1 reward."""
    from rdkit.DataStructs import TanimotoSimilarity
    cond = condition_vector(organism, gram, spectrum)
    cond_t = torch.from_numpy(cond).to(device)

    rewards = np.full(len(smiles_list), -1.0, dtype=np.float32)
    valid_indices: list[int] = []
    canonical_smiles: list[str] = []
    for i, s in enumerate(smiles_list):
        cs = validate_smiles(s)
        if cs:
            valid_indices.append(i)
            canonical_smiles.append(cs)

    if not valid_indices:
        return rewards

    # Score valid ones
    for i in range(0, len(valid_indices), bs):
        chunk_idx = valid_indices[i:i+bs]
        chunk_smi = canonical_smiles[i:i+bs]
        ids_list, mask_list = [], []
        for s in chunk_smi:
            ids, mask = ranker_tok(s, max_len)
            ids_list.append(ids); mask_list.append(mask)
        ids_t = torch.from_numpy(np.stack(ids_list)).long().to(device)
        mask_t = torch.from_numpy(np.stack(mask_list)).bool().to(device)
        cond_b = cond_t.unsqueeze(0).expand(len(chunk_smi), -1)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = ranker(ids_t, mask_t, cond_b)
        ab = out["antibacterial"].float().cpu().numpy()
        cy = out["cytotox"].float().cpu().numpy()
        ar = out["artifact"].float().cpu().numpy()
        for j, (idx, smi) in enumerate(zip(chunk_idx, chunk_smi)):
            base = float(ab[j]) - 0.5 * float(cy[j]) - 0.3 * float(ar[j])
            # Novelty: 1 - max Tan to training actives
            novelty = 1.0
            if active_fps:
                fp = _morgan_fp(smi)
                if fp is not None:
                    max_tan = max(TanimotoSimilarity(fp, af) for af in active_fps)
                    novelty = 1.0 - max_tan
            rewards[idx] = base + 0.3 * novelty
    return rewards


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-lm", type=Path, required=True)
    p.add_argument("--ranker", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--organism", default="E.coli")
    p.add_argument("--gram", default="Gram-negative")
    p.add_argument("--spectrum", default="broad_spectrum_or_general_antibacterial")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--samples-per-iter", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-coef", type=float, default=0.1,
                   help="KL-divergence anchor coefficient to keep policy near reference")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-ref-actives", type=int, default=500,
                   help="How many training-active fingerprints to use for novelty calc")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "rl_log.jsonl"

    def log(msg, **extra):
        if is_main:
            print(f"[{time.strftime('%H:%M:%S')}] [r{rank}] {msg}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"t": time.time(), "msg": str(msg), **extra}) + "\n")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")

    log(f"Loading base LM {args.base_lm}")
    policy = load_base_lm(args.base_lm, device)
    reference = load_base_lm(args.base_lm, device)  # frozen anchor
    for p_ in reference.parameters():
        p_.requires_grad = False

    log(f"Loading ranker {args.ranker}")
    ranker = load_ranker(args.ranker, device)

    log(f"Loading training-active fingerprints (organism={args.organism})")
    facts_df = pd.read_parquet(args.facts)
    actives = facts_df[(facts_df["organism"] == args.organism) &
                       (facts_df["activity_label"] == "active")]
    active_smiles = actives["canonical_smiles"].dropna().astype(str).unique().tolist()[:args.n_ref_actives]
    active_fps = [_morgan_fp(s) for s in active_smiles if _morgan_fp(s) is not None]
    log(f"  built {len(active_fps)} active fingerprints")

    if world_size > 1:
        policy = nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank])

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    # Running baseline
    baseline = 0.0
    baseline_momentum = 0.9

    t0 = time.time()
    samples_buffer: list[tuple[str, float]] = []  # (smi, reward) — for logging

    for iteration in range(args.iterations):
        policy.train()
        # 1. Sample SMILES from current policy.
        with torch.no_grad():
            pol_unwrap = policy.module if isinstance(policy, nn.parallel.DistributedDataParallel) else policy
            smiles = pol_unwrap.sample(
                args.samples_per_iter, max_len=args.max_len,
                temperature=args.temperature, top_p=args.top_p, device=device,
            )

        # 2. Score with ranker → reward.
        rewards = compute_rewards(smiles, ranker, args.organism, args.gram, args.spectrum,
                                    active_fps, device, max_len=args.max_len)
        mean_r = float(rewards.mean())
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * mean_r
        advantages = rewards - baseline

        # 3. Re-tokenize SMILES so we can compute log-prob.
        n_keep = sum(1 for r in rewards if r > -1.0)  # number valid
        ids_list = []
        for smi in smiles:
            ids, _ = tokenize_with_bos_eos(smi, args.max_len)
            ids_list.append(ids)
        ids_t = torch.tensor(np.stack(ids_list), dtype=torch.long, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        # 4. Forward + log-prob of generated sequences.
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = policy(ids_t)             # (B, T, V)
            ref_logits = reference(ids_t)      # KL anchor
        # log p(token_{t+1} | tokens_{0..t}) — shift by 1
        log_probs = F.log_softmax(logits.float()[:, :-1], dim=-1)
        ref_log_probs = F.log_softmax(ref_logits.float()[:, :-1], dim=-1)
        tgt = ids_t[:, 1:]
        pad_mask = (tgt != PAD).float()
        # Gather log-prob at the actually-generated token
        tok_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        ref_lp = ref_log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # Sequence log-prob = sum over non-pad positions
        seq_lp = (tok_lp * pad_mask).sum(dim=-1)
        ref_seq_lp = (ref_lp * pad_mask).sum(dim=-1)
        # Policy gradient loss (REINFORCE): - advantage * log-prob
        pg_loss = -(adv_t * seq_lp).mean()
        # KL anchor
        kl = (tok_lp - ref_lp) * pad_mask
        kl_loss = (args.kl_coef * kl.sum(dim=-1)).mean()
        loss = pg_loss + kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()

        if is_main:
            log(f"iter {iteration}/{args.iterations} mean_r={mean_r:.4f} baseline={baseline:.4f} "
                f"adv={float(adv_t.mean()):.4f} pg_loss={float(pg_loss):.4f} kl={float(kl_loss):.4f} "
                f"n_valid={n_keep}/{args.samples_per_iter}",
                iter=iteration, mean_reward=mean_r, baseline=baseline,
                n_valid=int(n_keep))

            # Save best-of-iter samples
            for s, r in zip(smiles, rewards):
                if r > 0.5:
                    samples_buffer.append((s, float(r)))
                    if len(samples_buffer) > 500:
                        samples_buffer.sort(key=lambda x: -x[1])
                        samples_buffer = samples_buffer[:300]

            if iteration % args.ckpt_every == 0 and iteration > 0:
                _save(policy, args, iteration, samples_buffer)

    if is_main:
        _save(policy, args, args.iterations, samples_buffer)
        log("FINAL", total_seconds=time.time() - t0)
    cleanup_ddp()
    return 0


def _save(model, args, iteration, samples_buffer):
    pol = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    sd = pol.state_dict()
    torch.save({
        "iteration": iteration, "model": sd, "args": vars(args),
        "framework": "ar_smiles_lm_rl_finetuned",
    }, args.out / "checkpoint.pt")
    # Top samples (sorted by reward)
    sorted_samples = sorted(samples_buffer, key=lambda x: -x[1])[:100]
    (args.out / "top_samples.json").write_text(json.dumps([
        {"smiles": s, "reward": r} for s, r in sorted_samples
    ], indent=2))


if __name__ == "__main__":
    sys.exit(main())
