"""Stage-1 multi-task pretraining entry point.

Trains the molecular encoder backbone on decontaminated ChEMBL + TDC
auxiliary endpoints. Mask-style + property-prediction multi-task. Per
PLAN.md §17.B with 32-GPU access we target ~500M-1B params on the H100s.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class PretrainConfig:
    backbone: str = "transformer"  # 'gnn' | 'transformer' | 'fingerprint_mlp'
    hidden_dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    seq_max_len: int = 256
    learning_rate: float = 1e-4
    batch_size: int = 256
    steps: int = 200_000
    warmup_steps: int = 4_000
    seed: int = 42
    use_fsdp: bool = True  # ON for >1 GPU
    aux_loss_weights: dict = None  # type: ignore[assignment]


def pretrain(cfg: PretrainConfig):
    import torch  # noqa: F401  (sanity import)

    raise NotImplementedError(
        "Stage-1 pretraining entry point. Activates after Phase A produces "
        "a decontaminated molecule corpus + auxiliary ADMET labels, and the "
        "encoder family is locked at Layer-2/Layer-3. See PLAN.md §17.B for "
        "the 16x H100 + 16x A100 cluster partition."
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["gnn", "transformer", "fingerprint_mlp"], default="transformer")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    cfg = PretrainConfig(
        backbone=args.backbone,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        learning_rate=args.lr,
        batch_size=args.bs,
        steps=args.steps,
        seed=args.seed,
    )
    pretrain(cfg)


if __name__ == "__main__":
    main()
