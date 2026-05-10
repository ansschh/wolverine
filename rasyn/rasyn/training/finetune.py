"""Stage-3 per-rescue-mode finetuning entry point.

Three parallel finetune jobs (one per rescue mode), each picking up the
Stage-2 main-trained checkpoint and specialising on:
  - direct_analog (preserves direct potency comparison)
  - prodrug_exposure (uses rule-based delivery proxy + active-species score,
                      not direct potency)
  - active_metabolite (similar to direct but with metabolite-aware features)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinetuneConfig:
    base_checkpoint: Path
    rescue_mode: str  # one of: direct, prodrug, active_metabolite
    learning_rate: float = 5e-5
    batch_size: int = 64
    epochs: int = 5
    seed: int = 42
    out_checkpoint: Path = Path("rasyn/data/clean/checkpoints/ranker_finetuned.pt")


def finetune(cfg: FinetuneConfig):
    raise NotImplementedError(
        "Stage-3 finetune. Activates after Stage-2 main training produces a "
        "checkpoint. Three modes run in parallel on dedicated GPU groups."
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--mode", required=True, choices=["direct", "prodrug", "active_metabolite"])
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    cfg = FinetuneConfig(
        base_checkpoint=args.base_checkpoint,
        rescue_mode=args.mode,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    finetune(cfg)


if __name__ == "__main__":
    main()
