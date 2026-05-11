"""Train 3 ABX ranker seeds (42, 43, 44) for ensemble inference (spec §12, §18.9).

Sequential orchestrator — each seed produces its own checkpoint directory.
At inference, `run_abx_sealed_cases.py` loads all available seed ckpts and
averages their outputs to get ensemble scores + variance for uncertainty.

Run on a single 8x A100 pod (sequential to share GPU memory):
    cd ~/wolverine/rasyn && source .venv/bin/activate
    python scripts/train_abx_ranker_multiseed.py \\
        --pretrain rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --tasks rasyn/data/clean/antibiotic/antibiotic_ranking_tasks.parquet \\
        --facts rasyn/data/clean/antibiotic/antibacterial_assay_facts.parquet \\
        --seeds 42,43,44 --steps 4000 --bs 32 \\
        --out-base rasyn/data/clean
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--nproc-per-node", type=int, default=8)
    p.add_argument("--out-base", type=Path, required=True)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        out_dir = args.out_base / f"abx_ranker_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"=== TRAINING seed={seed} -> {out_dir} ===")
        cmd = [
            "torchrun", f"--nproc_per_node={args.nproc_per_node}", "--standalone",
            "scripts/train_abx_ranker.py",
            "--pretrain", str(args.pretrain),
            "--tasks", str(args.tasks),
            "--facts", str(args.facts),
            "--steps", str(args.steps),
            "--bs", str(args.bs),
            "--lr", str(args.lr),
            "--seed", str(seed),
            "--out", str(out_dir),
        ]
        _log(" ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            _log(f"  seed={seed} FAILED rc={rc}; aborting")
            return rc
    _log("\nAll seeds trained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
