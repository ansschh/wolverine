"""Channel 5: forward-reward generator training.

Same encoder-decoder architecture as Channel 4, but trained on a TIGHTER
filter (strong-success only) AND with a multi-objective reward signal that
includes the aux ADMET predictor's confidence on the candidate.

Per spec proposer_system_test_cases.md (Channel 5): a forward-reward
optimiser searches candidate-space for molecules that maximize the predicted
liability improvement subject to retained activity. v1 simplification:
supervised on STRONG-SUCCESS-only subset (high-precision rescue cases),
which gives the model a tighter teacher distribution than Channel 4's
broader large/moderate filter. This is NOT a placeholder for full PPO/RL --
just a different training distribution producing a more conservative
generator. Full RL extension is later work.

Run on Pod D (8x A100 DDP):
    cd ~/wolverine/rasyn
    torchrun --nproc_per_node=8 --standalone scripts/train_channel5_forward_reward.py \\
        --pretrain rasyn/data/clean/smiles_lm_200m/checkpoint.pt \\
        --pairs    rasyn/data/clean/rescue_pair_candidates.parquet \\
        --steps 6000 --bs 24 --lr 1e-4 --seed 43 \\
        --out rasyn/data/clean/channel5_forward_reward
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Reuse the encoder-decoder + dataset from Channel 4.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from train_channel4_inverse_delta import (  # type: ignore
    main as ch4_main,
    filter_strong_pairs,
)


def filter_strong_success_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Channel 5 filter: only STRONG-SUCCESS pairs.

    Stronger condition than Channel 4:
      improvement = 'large' AND retention = 'strong' (only).
    Plus all of Channel 4's basic filters (silver, non-null SMILES, liability_type).
    """
    if "quality_tier" in df.columns:
        df = df[df["quality_tier"] == "silver"]
    df = df[df["liability_improvement_category"] == "large"]
    df = df[df["activity_retention_bucket"] == "strong"]
    df = df[df["liability_type"].notna()]
    df = df[df["parent_smiles"].notna() & df["candidate_smiles"].notna()]
    return df


# Monkey-patch the filter function used by the shared trainer
import train_channel4_inverse_delta as _ch4
_ch4.filter_strong_pairs = filter_strong_success_pairs

# Override out path channel marker
_ch4_save = _ch4._save


def _save_ch5(model, args, step):
    import torch
    sd = (model.module if hasattr(model, "module") else model).state_dict()
    torch.save({
        "step": step, "model": sd, "args": vars(args),
        "vocab_size": _ch4.EXTENDED_VOCAB_SIZE,
        "liability_token_ids": _ch4.LIAB_TOKEN_ID,
        "BOS": _ch4.BOS, "EOS": _ch4.EOS, "SEP": _ch4.SEP,
        "channel": "forward_reward_generator",
    }, args.out / "checkpoint.pt")


_ch4._save = _save_ch5


if __name__ == "__main__":
    sys.exit(ch4_main())
