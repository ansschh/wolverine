"""Stage-2 main training entry point for the rescue ranker.

Wires:
  RescuePairDataset -> DataLoader -> ConcatMLPRanker -> multi-task loss -> AdamW
  + rank-loss (BCE on rescue_score) + cross-entropy on rescue_label
  + multi-label BCE on failure_mode + cross-entropy on retention/improvement.

Single-process for now; FSDP/DDP wrap is added when L3 is green.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 10
    weight_decay: float = 1e-2
    seed: int = 42
    aux_weights: dict = None  # type: ignore[assignment]
    dataset_parquet: Path = Path("rasyn/data/clean/rescue_pairs.parquet")
    checkpoint_out: Path = Path("rasyn/data/clean/checkpoints/ranker.pt")


def train(cfg: TrainConfig):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from rasyn.ranker.torch_ranker import build_concat_mlp_ranker
    from rasyn.training.datasets import RescuePairDataset, load_parquet, torch_dataset

    torch.manual_seed(cfg.seed)
    df = load_parquet(cfg.dataset_parquet)  # noqa: F841 - placeholder until clean parquet ships
    raise NotImplementedError(
        "Stage-2 training entry point. Activates after Phase A-4 produces a clean rescue-pair parquet "
        "and Layer-2 verification picks the ranker architecture. See PLAN.md §8."
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", type=Path, default=TrainConfig.dataset_parquet)
    p.add_argument("--out", type=Path, default=TrainConfig.checkpoint_out)
    args = p.parse_args(argv)
    cfg = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.bs,
        epochs=args.epochs,
        seed=args.seed,
        dataset_parquet=args.data,
        checkpoint_out=args.out,
    )
    train(cfg)


if __name__ == "__main__":
    main()
